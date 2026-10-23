"""Audit the completed paired pilot and plot its fixed-probe learning curves."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import verified_source
from learning.vla.pickup_evaluation import audit_pickup_trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    plan = json.loads((args.campaign / "plan.json").read_text())
    completion = json.loads((args.campaign / "completion.json").read_text())
    if not completion["completed"] or not completion["training_comparison_completed"]:
        raise ValueError("Complete paired training/control pilot required")
    for name, source in plan["sources"].items():
        verified_source(name, source["sha256"])
    if sha256(args.campaign / "anchor_index.json") != plan["index_sha256"]:
        raise ValueError("Sampling index changed")
    rows, checks, updates = {}, {}, {}
    for mode in plan["modes_in_fixed_order"]:
        folder = args.campaign / ("training_"+mode)
        report = json.loads((folder / "report.json").read_text())
        events = list(map(json.loads, (folder / "events.jsonl").read_text().splitlines()))
        updates[mode] = [e for e in events if e["type"] == "update"]
        checks[mode] = [e for e in events if e["type"] == "evaluation"]
        if (not report["completed"] or report["updates"] != plan["steps_per_arm"]
                or len(updates[mode]) != plan["steps_per_arm"]
                or report["binding"]["base_weights_sha256"] != plan["base_weights_sha256"]
                or not all(np.isfinite(e[k]) for e in updates[mode] for k in ("loss", "grad_norm", "lr"))):
            raise ValueError("Unmatched/nonfinite training comparison")
        audit = json.loads((args.campaign / ("training_"+mode+"_audit.json")).read_text())
        if sha256(Path(report["final_checkpoint"]) / "pretrained_model/model.safetensors") != audit["final_model_sha256"]:
            raise ValueError("Final weights changed")
        categories = {}
        phase_rows = checks[mode][-1]["scores"]["train_phase"]["phase_rows"]
        for row in phase_rows:
            categories.setdefault(row["category"], []).append(row["prefix_errors"]["25"]["gripper_mae_mm"])
        rows[mode] = {"updates": len(updates[mode]), "training_audit": audit,
            "initial_validation_native_loss": checks[mode][0]["scores"]["validation"]["mean"],
            "final_validation_native_loss": checks[mode][-1]["scores"]["validation"]["mean"],
            "final_validation_actual8_loss": checks[mode][-1]["scores"]["validation"]["actual_8_channel_mean"],
            "final_train_phase_first25_gripper_mae_mm": {k:float(np.mean(v)) for k,v in categories.items()}}
    # Check both arms really started from the same pre-update fixed predictions.
    initial_a = checks["uniform"][0]["scores"]
    initial_b = checks["skip-survey"][0]["scores"]
    if initial_a != initial_b:
        raise ValueError("Arms did not start with identical fixed-probe results")
    controls = []
    for row in completion["controls"]:
        folder = Path(row["directory"])
        report = json.loads((folder / "report.json").read_text())
        with np.load(folder / "trajectory.npz", allow_pickle=False) as trace:
            metrics = audit_pickup_trace(report, trace, json.loads((folder / "diagnostics.json").read_text()))
        if any(row[key] != value for key, value in metrics.items()):
            raise ValueError("Saved control summary differs from physical trace")
        with np.load(folder / "tcp_path.npz", allow_pickle=False) as data:
            tcp, reference = data["actual_tcp"], data["expert_preclose_tcp"]
            for measurement in row["early_approach_tcp"]:
                i = measurement["command"]
                distance = np.linalg.norm(tcp[i]-reference, axis=1)
                if (abs(float(distance.min()*1000)-measurement["nearest_expert_path_mm"]) > 1e-7
                        or int(distance.argmin()) != measurement["nearest_expert_command"]):
                    raise ValueError("Approach-path summary differs from saved FK")
        controls.append(row)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for mode, color in (("uniform", "#27699a"), ("skip-survey", "#c46135")):
        u, c = updates[mode], checks[mode]
        values = np.array([e["loss"] for e in u]); width = min(50, len(values))
        axes[0, 0].plot([e["step"] for e in u][width-1:], np.convolve(values, np.ones(width)/width, "valid"), label=mode, color=color)
        axes[0, 1].plot([e["step"] for e in c], [e["scores"]["validation"]["actual_8_channel_mean"] for e in c], "o-", label=mode, color=color)
        for category, linestyle in (("first_active", "-"), ("approach", "--")):
            values = [np.mean([r["prefix_errors"]["25"]["gripper_mae_mm"] for r in e["scores"]["train_phase"]["phase_rows"] if r["category"] == category]) for e in c]
            axes[1, 0].plot([e["step"] for e in c], values, linestyle, marker="o", color=color, label=mode+" / "+category)
    axes[0, 0].set_ylabel("Native training loss, 50-update mean")
    axes[0, 1].set_ylabel("Fixed validation probe loss, actual 8 channels")
    axes[1, 0].set_ylabel("Training phase first-25 finger MAE (mm)")
    for axis in axes.flat[:3]:
        axis.set_xlabel("Optimizer updates"); axis.grid(alpha=.2); axis.legend(fontsize=8)
    axes[1, 1].axis("off")
    text = ["Physical pickup on training-reset control:"]
    text.extend(f"{r['variant']}: {r['outcome']} ({r['steps']} commands)" for r in controls)
    text.extend(["", "Held-out evaluation:"]+[f"{k}: {v['state']}" for k,v in completion["heldout"].items()])
    text.extend(["", "No unseen-scene or sim-to-real claim.", "One-seed pilot; no prefix or checkpoint sweep."])
    axes[1, 1].text(.02, .95, "\n".join(text), transform=axes[1, 1].transAxes, va="top", fontsize=10)
    fig.suptitle("SmolVLA: initial-wait sampling ablation | matched 1,000-update continuations\n"
                 "Only eligible training anchors change; native objective and physical guards unchanged", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    args.output.mkdir(parents=True, exist_ok=False)
    fig.savefig(args.output / "sampling_comparison.png", dpi=160)
    fig.savefig(args.output / "sampling_comparison.pdf")
    plt.close(fig)
    summary = {"completed": True, "variants": rows, "controls": controls, "heldout": completion["heldout"],
        "initial_fixed_probes_identical": True, "completion_sha256": sha256(args.campaign / "completion.json"),
        "limits": plan["limits"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
