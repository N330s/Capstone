"""Plot controlled gripper input swaps and record the execution-prefix control."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    directory = ROOT / "results/v2c30_input_ablation_20261013"
    directory.mkdir(exist_ok=False)
    source = ROOT / "outputs/v2c30_failure_diagnosis_20261013"
    report = json.loads((source / "input_swaps_v2/report.json").read_text())
    conditions = [("expert_state_expert_images", "Expert state + expert images"),
                  ("policy_state_expert_images", "Policy state + expert images"),
                  ("expert_state_policy_images", "Expert state + policy images"),
                  ("policy_state_policy_images", "Policy state + policy images"),
                  ("policy_joints_only", "Change only joint positions"),
                  ("policy_gripper_only", "Change only gripper position"),
                  ("policy_with_expert_gripper", "Policy inputs; restore expert gripper position")]
    values = [[r["finger_mm"] for r in report["predictions"] if r["frame"] == 320 and r["inputs"] == key]
              for key, _ in conditions]
    means, stds = [np.mean(v) for v in values], [np.std(v) for v in values]
    fig = plt.figure(figsize=(16, 9))
    fig.text(.06, .93, "Controlled test | Gripper-state feedback drives premature closure", fontsize=22, weight="bold")
    fig.text(.06, .87, "Same checkpoint, matched noise, three failed resets at command 320 | 3 sampling seeds per condition", fontsize=13)
    ax = fig.add_axes([.34, .23, .60, .54])
    colors = ["#16806a", "#cf662a", "#16806a", "#cf662a", "#2463ce", "#cf662a", "#2463ce"]
    ax.barh(np.arange(7), means, xerr=stds, color=colors, capsize=5)
    ax.set_yticks(np.arange(7), [label for _, label in conditions], fontsize=12)
    ax.invert_yaxis(); ax.set_xlim(0, 40)
    ax.set_xlabel("Predicted first gripper command [mm]; smaller = more closed", fontsize=12)
    ax.axvline(11, color="gray", linestyle="--", label="Demonstrated minimum: 11 mm")
    ax.grid(axis="x", alpha=.15)
    for index, value in enumerate(means):
        ax.text(value + stds[index] + .4, index, f"{value:.1f}", va="center", fontsize=12)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.text(.06, .13, "Changing only the gripper-state input recreates closure; restoring that input reverses it.", fontsize=16, weight="bold")
    fig.text(.06, .055, "Bars: mean across 3 resets × 3 seeds; error bars: descriptive standard deviation. Hybrid inputs are diagnostic interventions, not rollout successes.", fontsize=10)
    fig.savefig(directory / "gripper_state_ablation.png", dpi=150)
    fig.savefig(directory / "gripper_state_ablation.pdf")
    plt.close(fig)
    prefix_report = source / "prefix25/report.json"
    prefix = json.loads(prefix_report.read_text()) if prefix_report.exists() else None
    summary = {"conditions": [{"name": name, "mean_mm": float(mean), "std_mm": float(std)}
                               for (_, name), mean, std in zip(conditions, means, stds)],
               "per_reset_effects": [r for r in report["effects"] if r["frame"] == 320],
               "prefix25_control": prefix, "source": str(source / "input_swaps_v2/report.json")}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output": str(directory), "means_mm": means, "prefix25_complete": prefix is not None}, indent=2))


if __name__ == "__main__":
    main()
