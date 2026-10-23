"""Compare actual closed-loop actions with predictions on recorded observations."""
import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/v2c30_rollout_diagnosis_20261013")
    result = parser.parse_args().output
    result.mkdir(exist_ok=False)
    source = ROOT / "outputs/v2c30_failure_diagnosis_20261013"
    diagnosis = json.loads((source / "recorded_observations_v2/report.json").read_text())
    replay = json.loads((source / "expert_replay/report.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    fig.subplots_adjust(left=.06, right=.98, top=.78, bottom=.22, wspace=.13)
    fig.suptitle("Why low validation loss did not produce a successful grasp", fontsize=21, weight="bold", y=.95)
    fig.text(.06, .86, "Recorded expert observations yield reasonable gripper predictions; closed-loop execution drifts toward premature closure.", fontsize=12)
    rows = []
    for ax, name in zip(axes, ("train_20004", "train_21010", "train_21036")):
        with np.load(ROOT / "outputs/v2c30_best_sim_20261013" / ("best_" + name) / "trajectory.npz") as raw:
            live = raw["action_applied"][:, 7] * 1000
        with np.load(ROOT / "data/drive_v2c30_20261012/merged_all_v1" / name / "episode.npz") as raw:
            expert = raw["action_applied"][:len(live), 7] * 1000
        ax.plot(expert, color="#16806a", label="Expert applied command")
        ax.plot(live, color="#cf662a", label="VLA applied in closed loop")
        probes = [p for p in diagnosis["predictions"] if p["episode"] == name and p["step"] <= 321]
        ax.scatter([p["step"] for p in probes], [p["prediction_finger_mm"] for p in probes], color="#2463ce", s=22,
                   zorder=4, label="VLA on recorded observations")
        ax.axhline(11, color="gray", linestyle="--", linewidth=1, label="Training minimum (11 mm)")
        ax.set_title(name); ax.set_xlabel("Command index"); ax.grid(alpha=.2)
        rows.append({"episode": name, "expert_finger_mm_at_failure": float(expert[-1]),
                     "policy_finger_mm_at_failure": float(live[-1])})
    axes[0].set_ylabel("Per-finger travel target [mm] (smaller = more closed)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .065), ncol=2, frameon=False, fontsize=11)
    fig.text(.06, .035, "Diagnostic subset: 3 held-out recordings, 3 noise seeds per sampled expert frame. Matched expert replay succeeds; qpos/qvel errors = 0.", fontsize=10)
    fig.savefig(result / "gripper_failure_comparison.png", dpi=150)
    fig.savefig(result / "gripper_failure_comparison.pdf")
    plt.close(fig)
    summary = {"normalization_roundtrip_max_abs": diagnosis["action_roundtrip_max_abs"],
               "manual_decoder_max_abs_m": diagnosis["manual_decoder_max_abs_m"],
               "early_recorded_observation_gripper_mae_mm": diagnosis["early_gripper_mae_mm"],
               "early_recorded_predictions_below_train_min": diagnosis["early_prediction_below_train_min_count"],
               "early_recorded_prediction_count": diagnosis["early_prediction_count"],
               "expert_replay_task_success": replay["task_success"], "expert_replay_errors": replay["replay_errors"],
               "closed_loop": rows,
               "inference": "Results support accumulated closed-loop error. Relative contribution of state drift, image changes and action execution needs an intervention test."}
    (result / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
