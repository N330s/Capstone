"""Audit single-parent update exposure and visualize observed training/rollouts."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.smolvla_data import sha256


def verify_export_targets(plan, raw):
    """Read back actual LeRobot chunks, including the masked terminal tail."""
    os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(ROOT / "data/hf_cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data/hf_cache/datasets"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    export = Path(plan["export"])
    receipt = json.loads((export / "openarm_export.json").read_text())
    if sha256(export / "openarm_export.json") != plan["export_sha256"]:
        raise ValueError("Export changed")
    dataset = LeRobotDataset(receipt["datasets"]["train"]["repo_id"], root=export / "train",
        delta_timestamps={"action": [i / 50 for i in range(50)]}, video_backend="pyav")
    start, end = plan["train_index_span"]
    local = sorted({0, 1, (end - start) // 3, (end - start) // 2, end - start - 50, end - start - 1})
    for index in local:
        frame = dataset[start + index]
        expected = raw["action_applied"][np.minimum(np.arange(index, index + 50), end - start - 1)].astype(np.float32)
        if (not np.array_equal(frame["action"].numpy(), expected)
                or not np.array_equal(frame["observation.state"].numpy(), raw["observation_state"][index].astype(np.float32))
                or not np.array_equal(frame["action_is_pad"].numpy(), np.arange(index, index + 50) >= end - start)):
            raise ValueError("State/action timing, units or terminal mask differs from raw source")
    return {"passed": True, "local_indices": local, "raw_action_state_and_terminal_mask_exact": True}


def audit_updates(events, start, end, steps, batch_size):
    updates = [row for row in events if row["type"] == "update"]
    if len(updates) != steps or [r["step"] for r in updates] != list(range(1, steps + 1)):
        raise ValueError("Missing/duplicate optimizer updates")
    indices = []
    for row in updates:
        if (len(row["indices"]) != batch_size or not np.isfinite([row["loss"], row["grad_norm"]]).all()
                or any(type(i) is not int or not start <= i < end for i in row["indices"])):
            raise ValueError("Nonfinite update or other-parent exposure")
        indices.extend(row["indices"])
    return {"updates": steps, "sampled_anchors": len(indices), "distinct_anchors": len(set(indices)),
            "other_parent_anchors": 0, "all_updates_finite": True,
            "mean_anchor_visits": len(indices) / (end - start)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-only", action="store_true")
    group.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    report = json.loads((args.evaluation / "report.json").read_text())
    plan, training = report["plan"], report["training"]
    events = [json.loads(line) for line in (args.evaluation / "events.jsonl").read_text().splitlines()]
    exposure = audit_updates(events, *plan["train_index_span"], training["updates"], plan["batch_size"])
    source = Path(plan["episode"]) / "episode.npz"
    if not report["completed"] or not report["verification"]["passed"] or sha256(source) != plan["source_sha256"]:
        raise ValueError("Incomplete or changed experiment")
    for file, key in (("scripts/overfit_smolvla_episode.py", "trainer_sha256"),
                      ("scripts/finetune_smolvla.py", "original_trainer_sha256")):
        from learning.vla.source_archive import verified_source
        verified_source(file, plan[key])
    with np.load(source, allow_pickle=False) as raw:
        reference = raw["observation_state"].copy()
        targets = raw["action_applied"].copy()
        ref_time = raw["timestamp_s"].copy()
        target_check = verify_export_targets(plan, raw) if args.verify_only else None
    ref_time -= ref_time[0]
    traces, rows = {}, []
    if args.verify_only:
        import av
    for name, run in report["conditions"].items():
        folder = args.evaluation / name
        if (run["source_data_sha256"] != plan["source_sha256"] or run["recorded_input_assistance"]
                or run["simulator_state_forced_to_recording"]
                or run["mode"] != ("replay" if name == "expert_replay" else "policy")
                or run["execution_prefix"] != 5 or run["policy_seed"] != plan["seed"]
                or run["reset_state_max_error"] > 1e-5
                or run["input_state_source"] != "live_simulation" or run["input_image_source"] != "live_simulation"):
            raise ValueError("Assisted or changed evaluation")
        expected = (report["verification"]["weights_sha256"] if name == "overfit" else plan["base_weights_sha256"])
        if run["checkpoint_weights_sha256"] != expected:
            raise ValueError("Wrong checkpoint")
        with np.load(folder / "trajectory.npz", allow_pickle=False) as arrays:
            trace = {k: arrays[k].copy() for k in ("observation_state", "action_requested", "executed_dt_s")}
        if (trace["observation_state"].shape != (run["steps"] + 1, 16)
                or trace["action_requested"].shape != (run["steps"], 8)
                or trace["executed_dt_s"].shape != (run["steps"],)
                or not all(np.isfinite(v).all() for v in trace.values())):
            raise ValueError("Invalid trajectory")
        decoded = None
        if args.verify_only:
            with av.open(str(folder / "rollout.mp4")) as container:
                decoded = sum(1 for _ in container.decode(video=0))
            if decoded != run["steps"] + 1:
                raise ValueError("Incomplete video")
        traces[name] = trace
        rows.append({"condition": name, "steps": run["steps"], "grasp": run["grasp_established"],
            "insertion": run["task_success"], "outcome": run["outcome"],
            "recorded_input_assistance": False,
            "minimum_gripper_mm": float(trace["observation_state"][:, 15].min() * 1000),
            "decoded_video_frames": decoded, "video": str(folder / "rollout.mp4")})
    if args.verify_only:
        args.output.mkdir(parents=True, exist_ok=False)
        summary = {"episode": Path(plan["episode"]).name, "exposure": exposure, "conditions": rows,
            "raw_export_readback": target_check,
            "initial_probes": training["initial"]["scores"], "final_probes": training["final"]["scores"],
            "training_elapsed_s": training["elapsed_s"], "total_elapsed_s": report["elapsed_s"],
            "normalization_unchanged": report["verification"]["normalization_unchanged"],
            "limits": "One selected training reset and noise seed; memorization was attempted, not established. No held-out, real-time or sim2real success claim."}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        with (args.output / "conditions.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(summary), flush=True)
        return
    if not (args.output / "summary.json").is_file():
        raise ValueError("Verify before plotting")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    checks = [training["initial"], *[c for c in training["evaluations"] if c["step"] > 0]]
    if checks[-1]["step"] != training["final"]["step"]:
        checks.append(training["final"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.subplots_adjust(left=.075, right=.975, bottom=.09, top=.87, hspace=.36, wspace=.27)
    steps = [c["step"] for c in checks]
    for label, title in (("train_episode", "Selected training episode"), ("validation_monitor", "8 held-out monitor probes")):
        axes[0, 0].plot(steps, [c["scores"][label]["loss"] for c in checks], "o-", label=title)
        axes[0, 1].plot(steps, [c["scores"][label]["prefix5_gripper_mae_mm"] for c in checks], "o-", label=title)
    axes[0, 0].set(title="Fixed-noise flow-loss probes", xlabel="Optimizer updates", ylabel="Mean flow loss")
    axes[0, 0].set_yscale("log")
    axes[0, 1].set(title="Predicted first-five-command gripper error", xlabel="Optimizer updates", ylabel="MAE (mm)")
    axes[1, 0].plot(ref_time, reference[:, 15] * 1000, "k--", label="Recorded expert")
    axes[1, 1].plot(ref_time[:-1], targets[:, 7] * 1000, "k--", label="Recorded expert")
    for name, color in (("baseline", "#68737d"), ("overfit", "#2471a3")):
        trace = traces[name]
        times = np.r_[0., np.cumsum(trace["executed_dt_s"])]
        label = "Original VLA" if name == "baseline" else "Single-episode VLA"
        axes[1, 0].plot(times, trace["observation_state"][:, 15] * 1000, color=color, label=label)
        axes[1, 0].plot(times[-1], trace["observation_state"][-1, 15] * 1000, "x", color=color)
        axes[1, 1].plot(times[:-1], trace["action_requested"][:, 7] * 1000, color=color, label=label)
    axes[1, 0].set(title="Live-camera / live-state physical rollout", xlabel="Simulated time (s)", ylabel="Measured gripper (mm)")
    axes[1, 1].set(title="Gripper target sent to simulator", xlabel="Simulated time (s)", ylabel="Requested target (mm)")
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.suptitle("One-demonstration memorization test: " + Path(plan["episode"]).name +
                 "\nSame training reset, unchanged safety guards; not a generalization test", fontsize=13)
    fig.savefig(args.output / "overfit_diagnostic.png", dpi=160)
    fig.savefig(args.output / "overfit_diagnostic.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
