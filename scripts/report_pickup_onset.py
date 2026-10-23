"""Audit onset outputs and distinguish timing ambiguity from feedback/execution errors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.pickup_diagnosis import unfold_prefixes
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import verified_source, archive_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnosis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.diagnosis / "report.json").read_text())
    plan = json.loads((args.diagnosis / "plan.json").read_text())
    repeat = json.loads((args.diagnosis / "repeatability/report.json").read_text())
    if (not report["completed"] or not repeat["completed"] or report["elapsed_s"] > 900
            or repeat["elapsed_s"] > 300 or report["weights_sha256"] != repeat["weights_sha256"]
            or sha256(Path(plan["checkpoint"]) / "model.safetensors") != report["weights_sha256"]):
        raise ValueError("Complete bounded diagnostics on the same final checkpoint required")
    for name, record in plan["implementation"].items():
        verified_source(name, record["sha256"])
    verified_source("scripts/probe_pickup_repeatability.py", repeat["implementation"]["sha256"])
    with np.load(args.diagnosis / "repeatability/predictions.npz", allow_pickle=False) as arrays:
        for row in repeat["rows"]:
            prefix = f"{row['episode']}_{row['precision']}_{row['seed']}_"
            for kind in ("recorded", "rendered"):
                error = float(np.abs(arrays[prefix+kind+"_a"]-arrays[prefix+kind+"_b"]).max())
                if error != row[kind+"_repeat_max_abs"] or not row["identical_"+kind+"_input_hashes"]:
                    raise ValueError("Repeatability summary differs from stored predictions")
    collection = ROOT / "data/drive_v2c30_20261012/merged_all_v1"
    receipt_path = ROOT / "data/drive_v2c30_20261012/export_all_v1/openarm_export.json"
    receipt = json.loads(receipt_path.read_text())
    survey_rows = []
    for record in receipt["datasets"]["train"]["episodes"]:
        with np.load(collection / record["episode"] / "episode.npz", allow_pickle=False) as source:
            phase, state = source["phase"], source["observation_state"]
            n = int(np.flatnonzero(phase != "survey")[0])
            survey_rows.append({"episode": record["episode"], "survey_commands": n,
                "state_max_change": float(np.abs(state[:n]-state[0]).max())})
    name = "train_20002"
    with np.load(collection / name / "episode.npz", allow_pickle=False) as source:
        actions, phases = source["action_applied"], source["phase"]
        survey_count = int(np.flatnonzero(phases != "survey")[0])
        future = np.stack([actions[i:i+50, 7]*1000 for i in range(survey_count)])
        close = int(np.flatnonzero(phases == "close")[0])
    main_row = next(row for row in report["rows"] if row["episode"] == name)
    profile = main_row["profiles"][0]
    predicted = np.asarray(profile["finger_prediction_mm"])
    with np.load(ROOT / "outputs/fp32_pickup_diagnosis_20261020" / name / "kinematics.npz", allow_pickle=False) as arrays:
        reference = arrays["expert_tcp"][:close+1].copy()
    early, trajectories = [], []
    for prefix in (1, 5, 25):
        with np.load(args.diagnosis / name / f"prefix_{prefix}_early.npz", allow_pickle=False) as data:
            trace = {k:data[k].copy() for k in data.files}
        chunks = json.loads((args.diagnosis / name / f"prefix_{prefix}_chunks.json").read_text())
        expanded = unfold_prefixes([c["anchor"] for c in chunks], [c["prediction"] for c in chunks], 200, prefix)
        if not np.array_equal(expanded, trace["action_requested"]) or trace["state"].shape != (201, 16):
            raise ValueError("Early prefix execution differs from declared commands")
        nearest = np.linalg.norm(trace["tcp"][:, None, :]-reference[None, :, :], axis=2)
        early.append({"prefix": prefix, "commands": 200,
            "minimum_finger_mm": float(trace["state"][:, 15].min()*1000),
            "final_finger_mm": float(trace["state"][-1, 15]*1000),
            "final_nearest_reference_command": int(nearest[-1].argmin()),
            "final_off_reference_path_mm": float(nearest[-1].min()*1000),
            "not_a_pickup_evaluation": True})
        trajectories.append((prefix, trace["state"][:, 15]*1000, nearest.min(axis=1)*1000))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(50)
    axes[0].fill_between(x, future.min(0), future.max(0), color="#bbc1c7", alpha=.5,
        label="Target range across 44 near-static survey observations")
    axes[0].plot(x, future[0], color="black", label="Actual target at reset")
    axes[0].plot(x, predicted, color="#c46135", label="Predicted at reset, seed1000")
    axes[0].set_xlabel("Future action offset in the 50-command chunk")
    axes[0].set_ylabel("Finger target (mm)")
    axes[0].set_title("Initial chunk closes ahead of the demonstration")
    axes[0].legend(fontsize=7)
    for prefix, finger, path in trajectories:
        axes[1].plot(np.arange(201), finger, label=f"Replan every {prefix} command(s)")
        axes[2].plot(np.arange(201), path, label=f"Prefix {prefix}")
    axes[1].axhline(35, color="black", linestyle=":", label="Nominal approach opening: 35 mm")
    axes[1].set_xlabel("Executed commands")
    axes[1].set_ylabel("Actual live finger travel (mm)")
    axes[1].set_title("More frequent feedback helps opening, not uniformly")
    axes[1].legend(fontsize=7)
    axes[2].set_xlabel("Executed commands")
    axes[2].set_ylabel("Nearest expert approach-path distance (mm)")
    axes[2].set_title("Short prefixes do not eliminate arm deviation")
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("Final FP32 VLA: onset diagnosis on the training-reset control\n"
        "No training or production-controller change; 200-command interventions are not pickup-success tests", fontsize=11)
    fig.tight_layout(rect=(0, .04, 1, .89))
    fig.text(.04, .02, "Shared explicit noise at common command anchors; changing prefix changes subsequent feedback/noise sampling. No best prefix is selected.", fontsize=8)
    args.output.mkdir(parents=True, exist_ok=False)
    fig.savefig(args.output / "onset_diagnosis.png", dpi=160)
    fig.savefig(args.output / "onset_diagnosis.pdf")
    plt.close(fig)
    native = ROOT / ".venv-smolvla/Lib/site-packages/lerobot/policies/smolvla/modeling_smolvla.py"
    summary = {"onset": report, "repeatability": repeat, "early_prefix_audit": early,
        "training_survey_audit": {"parents": len(survey_rows), "rows": survey_rows,
            "total_survey_commands": sum(r["survey_commands"] for r in survey_rows), "receipt_sha256": sha256(receipt_path)},
        "native_forward_source": archive_source(native.relative_to(ROOT)),
        "evidence_sha256": {"onset": sha256(args.diagnosis / "report.json"),
            "repeatability": sha256(args.diagnosis / "repeatability/report.json")}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"early_prefixes": early, "survey_parent_count": len(survey_rows),
        "survey_lengths": sorted(set(r["survey_commands"] for r in survey_rows))}, indent=2))


if __name__ == "__main__":
    main()
