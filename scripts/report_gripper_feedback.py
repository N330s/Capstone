"""Independent update/overlay/video audit and plots for paired feedback training."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.gripper_feedback import open_hold_anchors
from learning.vla.smolvla_data import sha256


def audit_updates(rows, eligible, count, augmented):
    expected = list(range(1, 801))
    if [r["step"] for r in rows] != expected:
        raise ValueError("Expected exactly 800 consecutive optimizer updates")
    eligible = set(eligible)
    for row in rows:
        if (len(row["indices"]) != 8 or any(type(i) is not int or not 0 <= i < count for i in row["indices"])
                or not set(row["indices"][:4]) <= eligible
                or not np.isfinite([row["loss"], row["grad_norm"], *row["gripper_sensor_offsets_m"]]).all()
                or len(row["gripper_sensor_offsets_m"]) != 4
                or max(abs(x) for x in row["gripper_sensor_offsets_m"]) > .00200001
                or (not augmented and any(row["gripper_sensor_offsets_m"]))):
            raise ValueError("Faulty sampling/noise/finite-gradient audit")
    return {"updates": len(rows), "anchors": len(rows) * 8,
        "open_hold_quota_anchors": len(rows) * 4,
        "perturbed_anchors": sum(x != 0 for r in rows for x in r["gripper_sensor_offsets_m"]),
        "other_parent_anchors": 0, "all_updates_finite": True}


def feedback_metrics(probe, actions):
    result = dict(probe["metrics"])
    rows = probe["predictions"]
    open_nominal = [r for r in rows if r["frame"] in (100, 200, 300, 400) and r["offset_m"] == 0]
    closing = [r for r in rows if r["frame"] not in (100, 200, 300, 400)]
    perturbed = [r for r in rows if r["offset_m"] != 0]
    result["nominal_open_hold_mae_mm"] = float(np.mean([r["gripper_mae_mm"] for r in open_nominal]))
    result["close_and_hold_mae_mm"] = float(np.mean([r["gripper_mae_mm"] for r in closing]))
    result["restoring_direction_fraction"] = float(np.mean([
        (r["predicted_gripper_mm"] - r["input_gripper_mm"]) *
        (float(actions[r["frame"], 7] * 1000) - r["input_gripper_mm"]) > 0 for r in perturbed]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-only", action="store_true")
    group.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    paired = args.experiment / "paired_feedback"
    experiment = json.loads((paired / "report.json").read_text())
    if not experiment["completed"]:
        raise ValueError("Completed paired experiment required")
    for name, expected in experiment["plan"]["files"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("Experiment implementation changed")
    episode = ROOT / "data/drive_v2c30_20261012/merged_all_v1/train_20002"
    metadata = json.loads((episode / "metadata.json").read_text())
    if sha256(episode / "episode.npz") != metadata["data_sha256"]:
        raise ValueError("Source changed")
    with np.load(episode / "episode.npz", allow_pickle=False) as raw:
        reference = raw["observation_state"].copy()
        actions = raw["action_applied"].copy()
        times = raw["timestamp_s"].copy()
    times -= times[0]
    eligible = open_hold_anchors(reference, actions)
    models, updates, exposure = {}, {}, {}
    for mode in ("nominal", "augmented"):
        models[mode] = json.loads((paired / mode / "report.json").read_text())
        verification = json.loads((paired / mode / "verification.json").read_text())
        if (not verification["passed"] or verification["prediction_max_abs_difference"] != 0
                or models[mode]["binding"]["eligible_indices"] != eligible
                or models[mode]["binding"]["episode"]["data_sha256"] != metadata["data_sha256"]):
            raise ValueError("Overlay reload/data binding differs")
        updates[mode] = [json.loads(line) for line in (paired / mode / "updates.jsonl").read_text().splitlines()]
        exposure[mode] = audit_updates(updates[mode], eligible, len(actions), mode == "augmented")
    if ([r["indices"] for r in updates["nominal"]] != [r["indices"] for r in updates["augmented"]]
            or models["nominal"]["initial"] != models["augmented"]["initial"]):
        raise ValueError("Unmatched initial conditions/sample schedules")
    conditions = {
        "original": ROOT / "outputs/single_episode_overfit_20261019/baseline",
        "quantized_1mm": args.experiment / "quantized_1mm/rollout",
        "nominal": paired / "nominal_rollout",
        "augmented": paired / "augmented_rollout"}
    traces, rows = {}, []
    if args.verify_only:
        import av
    for mode, folder in conditions.items():
        run = json.loads((folder / "report.json").read_text())
        if (not run["pipeline_passed"] or run["mode"] != "policy" or run["recorded_input_assistance"]
                or run["simulator_state_forced_to_recording"] or run["reset_state_max_error"] != 0
                or run["source_data_sha256"] != metadata["data_sha256"] or run["policy_seed"] != 1000
                or run["execution_prefix"] != 5 or run["checkpoint_weights_sha256"] != experiment["plan"]["base_sha256"]):
            raise ValueError("Unmatched live physical comparison")
        if mode in models:
            if (run["expert_overlay_weights_sha256"] != models[mode]["overlay"]["weights_sha256"]
                    or run["actuation_filter"]["gripper_resolution_mm"] != 0):
                raise ValueError("Wrong overlay or filtered training comparison")
        with np.load(folder / "trajectory.npz", allow_pickle=False) as trajectory:
            trace = {key: trajectory[key].copy() for key in ("observation_state", "action_requested", "executed_dt_s")}
            if mode in models and not np.array_equal(trajectory["policy_action_raw"], trace["action_requested"]):
                raise ValueError("Commands changed after inference")
        if (trace["observation_state"].shape != (run["steps"] + 1, 16)
                or trace["action_requested"].shape != (run["steps"], 8)
                or not all(np.isfinite(v).all() for v in trace.values())):
            raise ValueError("Invalid physical trace")
        count = None
        if args.verify_only:
            with av.open(str(folder / "rollout.mp4")) as container:
                count = sum(1 for _ in container.decode(video=0))
            if count != run["steps"] + 1:
                raise ValueError("Video/trajectory mismatch")
        diagnostics = json.loads((folder / "diagnostics.json").read_text())
        rows.append({"condition": mode, "steps": run["steps"], "outcome": run["outcome"],
            "grasp": run["grasp_established"], "insertion": run["task_success"],
            "minimum_gripper_mm": float(trace["observation_state"][:, 15].min() * 1000),
            "max_plug_finger_contacts": max(r["plug_finger_contacts"] for r in diagnostics),
            "decoded_video_frames": count, "video": str(folder / "rollout.mp4")})
        traces[mode] = trace
    if args.verify_only:
        args.output.mkdir(parents=True, exist_ok=False)
        summary = {"exposure": exposure, "conditions": rows,
            "metrics": {mode: {"initial": feedback_metrics(model["initial"], actions), "final": feedback_metrics(model["final"], actions)}
                        for mode, model in models.items()},
            "identical_sample_schedules": True, "identical_initial_predictions": True,
            "unchanged_base_normalization": True,
            "limits": "One training reset and rollout seed; counterfactual sensor probes are not physical recovery trials. No task-success claim unless explicitly reported by live rollout."}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        with (args.output / "conditions.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(summary), flush=True)
        return
    if not (args.output / "summary.json").is_file():
        raise ValueError("Verify before rendering")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.subplots_adjust(left=.075, right=.98, bottom=.09, top=.87, hspace=.35, wspace=.28)
    colors = {"nominal": "#d9822b", "augmented": "#2471a3"}
    for mode, model in models.items():
        checks = [model["initial"], *[c for c in model["evaluations"] if c["step"] > 0]]
        checks = [c for c in checks if c["step"] < model["updates"]] + [model["final"]]
        x = [c["step"] for c in checks]
        label = "Low LR only" if mode == "nominal" else "Low LR + open-hold sensor noise"
        axes[0, 0].plot(x, [c["metrics"]["perturbed_open_hold"]["gripper_mae_mm"] for c in checks],
                        "o-", color=colors[mode], label=label)
        axes[0, 1].plot(x, [c["metrics"]["gripper_sensor_gain"] for c in checks],
                        "o-", color=colors[mode], label=label)
    axes[0, 0].set(title="Offline two-mm gripper-state perturbations", xlabel="Optimizer updates", ylabel="Prefix-five gripper MAE (mm)")
    axes[0, 1].axhline(1., color="gray", linestyle="--", alpha=.5)
    axes[0, 1].set(title="Open-hold dependence on measured gripper", xlabel="Optimizer updates", ylabel="Output change / sensor change")
    for axis in axes[1]:
        axis.plot(times, reference[:, 15] * 1000, "k--", label="Recorded expert")
    for mode in ("original", "nominal", "augmented"):
        trace = traces[mode]
        t = np.r_[0., np.cumsum(trace["executed_dt_s"])]
        color = colors.get(mode, "#68737d")
        axes[1, 0].plot(t, trace["observation_state"][:, 15] * 1000, color=color, label=mode)
        axes[1, 0].plot(t[-1], trace["observation_state"][-1, 15] * 1000, "x", color=color)
        axes[1, 1].plot(t[:-1], trace["action_requested"][:, 7] * 1000, color=color, label=mode)
    axes[1, 0].set(title="Unfiltered live-input physical rollouts", xlabel="Simulated time (s)", ylabel="Measured finger travel (mm)")
    axes[1, 1].set(title="Original continuous target interface", xlabel="Simulated time (s)", ylabel="Gripper command (mm)")
    for axis in axes.flat:
        axis.legend(fontsize=7)
        axis.grid(alpha=.2)
    fig.suptitle("Paired feedback-supervision test: identical batches and completed schedules\nOnly training-time open-hold sensor perturbations differ; not a generalization test", fontsize=12)
    fig.savefig(args.output / "feedback_comparison.png", dpi=160)
    fig.savefig(args.output / "feedback_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
