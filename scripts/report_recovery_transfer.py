"""Summarize paired old-scene rollouts and recorded-observation diagnostics."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


def rollout_metrics(row, collection, common_prefix=320):
    directory = Path(row["directory"])
    report = row["report"]
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    with np.load(directory / "trajectory.npz", allow_pickle=False) as arrays:
        states, applied = arrays["observation_state"], arrays["action_applied"]
    with np.load(collection / row["episode"] / "episode.npz", allow_pickle=False) as arrays:
        expert_actions, expert_states, phases = arrays["action_applied"], arrays["observation_state"], arrays["phase"]
    close = int(np.flatnonzero(phases == "close")[0])
    early = min(common_prefix, len(applied), len(expert_actions), close)
    empty_close = [i for i, action in enumerate(applied) if action[7] < .011
                   and not diagnostics[i + 1]["grasp_established"]]
    return {"model": row["model"], "episode": row["episode"], "outcome": report["outcome"],
            "steps": report["steps"], "task_success": report["task_success"],
            "grasp_established": report["grasp_established"], "safety_abort": report["safety_abort"],
            "max_plug_finger_contacts": max(i["plug_finger_contacts"] for i in diagnostics),
            "observed_gripper_min_mm": float(states[:, 15].min() * 1000),
            "applied_gripper_min_mm": float(applied[:, 7].min() * 1000),
            "first_empty_command_below_11mm": empty_close[0] if empty_close else None,
            "below_11mm_note": "Training-range diagnostic, not a mechanical safety bound",
            "early_compared_commands": early,
            "early_applied_gripper_mae_mm": float(np.abs(applied[:early, 7] - expert_actions[:early, 7]).mean() * 1000),
            "early_observed_joint_mae_rad": float(np.abs(states[:early, 8:15] - expert_states[:early, 8:15]).mean()),
            "expert_first_close_command": close,
            "expert_reference_phase_at_stop": str(phases[min(len(applied) - 1, len(phases) - 1)]),
            "observed_gripper_at_320_mm": float(states[320, 15] * 1000) if len(states) > 320 else None,
            "peak_robot_unwanted_contact_n": report["peak_robot_unwanted_contact_n"],
            "video": str(directory / "rollout.mp4")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--collection", type=Path, default=Path("data/drive_v2c30_20261012/merged_all_v1"))
    parser.add_argument("--baseline-probes", type=Path)
    parser.add_argument("--candidate-probes", type=Path)
    parser.add_argument("--recovery-view", type=Path, default=Path("data/recovery_pipeline_20261019/qualification_view"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads((args.evaluation / "report.json").read_text())
    if not evaluation.get("completed"):
        raise ValueError("Completed paired evaluation required")
    args.output.mkdir(parents=True, exist_ok=False)
    prefixes = {name: min(320, *(r["report"]["steps"] for r in evaluation["rollouts"] if r["episode"] == name))
                for name in evaluation["episodes"]}
    rows = [rollout_metrics(row, args.collection, prefixes[row["episode"]]) for row in evaluation["rollouts"]]
    candidate = Path(evaluation["checkpoints"]["recovery_pilot"])
    updates = json.loads((candidate.parent / "complete.json").read_text())["step"]
    probes = {}
    for name, path in (("baseline", args.baseline_probes), ("recovery_pilot", args.candidate_probes)):
        if path:
            probes[name] = json.loads((path / "report.json").read_text())
            if probes[name]["weights_sha256"] != evaluation["weights_sha256"][name]:
                raise ValueError("Probe/model binding mismatch")
    coverage = []
    view_manifest = json.loads((args.recovery_view / "manifest.json").read_text())
    for record in view_manifest["episodes"]:
        with np.load(args.recovery_view / record["episode"] / "episode.npz", allow_pickle=False) as raw:
            recovery = raw["step_label"] == "recovery"
            states = raw["observation_state"][:-1]
            coverage.append({"episode": record["episode"], "split": record["split"],
                             "recovery_phase_counts": dict(Counter(raw["phase"][recovery].tolist())),
                             "recovery_gripper_range_mm": [float(states[recovery, 15].min() * 1000),
                                                            float(states[recovery, 15].max() * 1000)]})
    result = {"rollouts": rows, "recorded_probe_summaries": {k: v["summary"] for k, v in probes.items()},
              "recorded_probe_state_sensitivity": {k: v["state_sensitivity"] for k, v in probes.items()},
              "training_sampling_audit": json.loads((args.evaluation / "sampling_audit.json").read_text()),
              "downloaded_recovery_coverage": coverage,
              "scope": evaluation["scope"], "candidate_updates": updates,
              "limits": f"Selected old resets, one policy seed; {updates} recovery updates. Offline sensitivity probes are not physically consistent recovery rollouts."}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "rollouts.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    episodes = list(evaluation["episodes"])
    colors = {"baseline": "#cf662a", "recovery_pilot": "#2463ce"}
    fig, axes = plt.subplots(1, len(episodes), figsize=(16, 6), squeeze=False)
    for axis, episode in zip(axes[0], episodes):
        with np.load(args.collection / episode / "episode.npz", allow_pickle=False) as source:
            expert = source["observation_state"][:, 15]
        axis.plot(np.arange(min(601, len(expert))), expert[:601] * 1000, color="#16806a", label="Recorded expert")
        for row in evaluation["rollouts"]:
            if row["episode"] != episode:
                continue
            with np.load(Path(row["directory"]) / "trajectory.npz", allow_pickle=False) as trajectory:
                finger = trajectory["observation_state"][:, 15]
            axis.plot(np.arange(min(601, len(finger))), finger[:601] * 1000,
                      color=colors[row["model"]], label=row["model"].replace("_", " "))
        axis.set_title(episode)
        axis.set_xlabel("Executed command boundary (first 600)")
        axis.set_ylim(0, 47)
        axis.grid(alpha=.2)
    axes[0, 0].set_ylabel("Observed per-finger travel [mm]")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"Paired comparison in the original source-matched scene | baseline vs {updates}-update recovery pilot", fontsize=15)
    fig.text(.04, .015, "Identical resets, sampling seed 1000, execution prefix 5, unchanged guards. Not evaluation in the new recovery scene.", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .93))
    fig.savefig(args.output / "gripper_rollout_comparison.png", dpi=150)
    fig.savefig(args.output / "gripper_rollout_comparison.pdf")
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
