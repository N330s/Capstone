"""Audit saved diagnostic predictions and plot expert-input versus live divergence."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.pickup_diagnosis import unfold_prefixes, command_errors
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import verified_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnosis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.diagnosis / "report.json").read_text())
    plan = json.loads((args.diagnosis / "plan.json").read_text())
    if not report["completed"] or len(report["rows"]) != 2 or report["elapsed_s"] > plan["max_seconds"]:
        raise ValueError("Complete bounded two-reset diagnostic required")
    for name, binding in plan["sources"].items():
        verified_source(name, binding["sha256"])
    if sha256(Path(plan["checkpoint"]) / "model.safetensors") != report["checkpoint_weights_sha256"]:
        raise ValueError("Diagnosed checkpoint changed")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    args.output.mkdir(parents=True, exist_ok=False)
    for col, row in enumerate(report["rows"]):
        name = row["episode"]
        source = ROOT / "data/drive_v2c30_20261012/merged_all_v1" / name / "episode.npz"
        if sha256(source) != row["source_data_sha256"]:
            raise ValueError("Source recording changed")
        with np.load(source, allow_pickle=False) as raw:
            expert = raw["action_applied"][:1200].copy()
            phases = raw["phase"][:1200].copy()
        live_dir = ROOT / "outputs/fp32_multiepisode_20261020/pickup_evaluation" / (name+"_policy")
        if sha256(live_dir / "report.json") != row["live_report_sha256"]:
            raise ValueError("Original live outcome changed")
        with np.load(live_dir / "trajectory.npz", allow_pickle=False) as raw:
            live = raw["action_requested"].copy()
            actual_finger = raw["observation_state"][:1200, 15].copy()
        with np.load(args.diagnosis / name / "recorded_predictions.npz", allow_pickle=False) as raw:
            teacher = unfold_prefixes(raw["anchors"], raw["predictions"], 1200)
            np.testing.assert_array_equal(teacher, raw["executed_prefix_commands"])
        if command_errors(teacher, expert, phases) != row["teacher"] or command_errors(live, expert, phases) != row["live"]:
            raise ValueError("Saved diagnostic error summary differs from predictions")
        with np.load(args.diagnosis / name / "kinematics.npz", allow_pickle=False) as raw:
            tcp_error = np.linalg.norm(raw["live_tcp"]-raw["expert_tcp"], axis=1)*1000
        j_teacher = np.abs(teacher[:, :7]-expert[:, :7]).mean(axis=1)*180/np.pi
        j_live = np.abs(live[:, :7]-expert[:, :7]).mean(axis=1)*180/np.pi
        x = np.arange(1200)
        axes[0, col].plot(x, expert[:, 7]*1000, color="black", label="Expert target")
        axes[0, col].plot(x, teacher[:, 7]*1000, color="#24885c", label="VLA: recorded state + RGB")
        axes[0, col].plot(x, live[:, 7]*1000, color="#c46135", label="VLA: live state + RGB")
        axes[0, col].plot(x, actual_finger*1000, color="#8666aa", alpha=.6, linestyle="--", label="Actual live finger travel")
        axes[0, col].set_ylabel("Finger travel / target (mm)")
        axes[0, col].set_title(name + (" | training control" if col == 0 else " | validation reset"))
        axes[1, col].plot(np.arange(len(tcp_error)), tcp_error, color="#c46135")
        axes[1, col].set_ylabel("Actual live TCP vs expert TCP (mm)\nat same command index")
        axes[2, col].plot(x, j_teacher, color="#24885c", label="Recorded-input prediction")
        axes[2, col].plot(x, j_live, color="#c46135", label="Live-input prediction")
        axes[2, col].set_ylabel("Mean absolute 7-joint target error (degrees)")
        close = row["source_first_close_command"]
        for axis in axes[:, col]:
            axis.axvline(close, color="black", linestyle=":", label="Expert close phase begins")
            axis.set_xlim(0, 850)
            axis.set_xlabel("Command index (not policy inference count)")
            axis.grid(alpha=.2)
        axes[0, col].legend(fontsize=8)
        axes[2, col].legend(fontsize=8)
        with (args.output / (name+"_comparison.csv")).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["command", "expert_phase", "expert_target_mm", "teacher_target_mm", "live_target_mm",
                             "live_finger_mm", "live_tcp_error_mm", "teacher_joint_mae_deg", "live_joint_mae_deg"])
            writer.writerows(zip(x, phases, expert[:, 7]*1000, teacher[:, 7]*1000, live[:, 7]*1000,
                                 actual_finger*1000, tcp_error[:1200], j_teacher, j_live))
    fig.suptitle("Final FP32 SmolVLA: recorded expert inputs versus live rollout\n"
        "Green uses externally supplied observations; it is not autonomous-policy success", fontsize=13)
    fig.tight_layout(rect=(0, .035, 1, .94))
    fig.text(.03, .012, "Same final checkpoint, seed 1000 and execution prefix 25. Physics and guards unchanged. Two-reset diagnostic, not generalization evidence.", fontsize=9)
    fig.savefig(args.output / "recorded_vs_live.png", dpi=160)
    fig.savefig(args.output / "recorded_vs_live.pdf")
    plt.close(fig)
    (args.output / "summary.json").write_text(json.dumps({**report,
        "diagnostic_report_sha256": sha256(args.diagnosis / "report.json"), "prediction_error_readback_passed": True}, indent=2))


if __name__ == "__main__":
    main()
