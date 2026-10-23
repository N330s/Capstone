"""Verify and summarize physically executed recorded-input control diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.smolvla_data import sha256


def verify_input_trace(chunks, reference, steps):
    cursor = 0
    for chunk in chunks:
        index = chunk["recorded_observation_index"]
        if (index != cursor or chunk["start_step"] != index or not 0 <= index < len(reference) - 1
                or not np.array_equal(np.asarray(chunk["policy_input_state"]), reference[index])
                or not 1 <= chunk["executed_steps"] <= 50):
            raise ValueError("Recorded input index/value/command alignment mismatch")
        cursor += chunk["executed_steps"]
    if cursor != steps:
        raise ValueError("Recorded input trace does not cover actual commands")
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    rendering = parser.add_mutually_exclusive_group()
    rendering.add_argument("--verify-only", action="store_true", help="Write verified JSON/CSV without requiring matplotlib")
    rendering.add_argument("--plot-only", action="store_true", help="Render already verified results without requiring PyAV")
    args = parser.parse_args()
    if args.output.exists() and not args.plot_only:
        raise FileExistsError(args.output)
    verified = json.loads((args.output / "summary.json").read_text()) if args.plot_only else None
    report = json.loads((args.evaluation / "report.json").read_text())
    if not report["completed"] or sha256(args.episode / "episode.npz") != report["source_sha256"]:
        raise ValueError("Complete, source-bound diagnostic required")
    with np.load(args.episode / "episode.npz", allow_pickle=False) as arrays:
        reference = arrays["observation_state"].copy()
        targets = arrays["action_applied"].copy()
        reference_time = arrays["timestamp_s"].copy()
    reference_time -= reference_time[0]
    cases = [{"mode": "policy", "directory": report["closed_loop_baseline"]["path"],
              "report": report["closed_loop_baseline"]["report"]}, *report["conditions"]]
    traces, rows = {}, []
    if not args.plot_only:
        import av
    elif (verified["source_sha256"] != report["source_sha256"]
          or verified["checkpoint_weights_sha256"] != report["checkpoint_weights_sha256"]):
        raise ValueError("Plot input differs from verified result")
    for case in cases:
        mode, folder, run = case["mode"], Path(case["directory"]), case["report"]
        if run["checkpoint_weights_sha256"] != report["checkpoint_weights_sha256"]:
            raise ValueError("Condition uses a different checkpoint")
        with np.load(folder / "trajectory.npz", allow_pickle=False) as arrays:
            trace = {k: arrays[k].copy() for k in ("observation_state", "action_requested", "executed_dt_s")}
        actual, steps = trace["observation_state"], run["steps"]
        if actual.shape != (steps + 1, 16) or trace["action_requested"].shape != (steps, 8):
            raise ValueError("Trajectory step/observation shapes differ")
        if not all(np.isfinite(v).all() for v in trace.values()):
            raise ValueError("Nonfinite physical trajectory")
        chunks = json.loads((folder / "chunks.json").read_text())
        verified_chunks = verify_input_trace(chunks, reference, steps) if mode.startswith("recorded-") else 0
        if args.plot_only:
            frames = next(row["decoded_video_frames"] for row in verified["conditions"] if row["mode"] == mode)
        else:
            with av.open(str(folder / "rollout.mp4")) as container:
                frames = sum(1 for _ in container.decode(video=0))
        if frames != steps + 1:
            raise ValueError("Video does not match actual trajectory")
        traces[mode] = trace
        difference = actual - reference[:steps + 1]
        rows.append({"mode": mode, "input_state": "recorded" if mode.startswith("recorded-") else "live" if mode == "policy" else "not_used",
            "input_images": "recorded" if mode == "recorded-observations" else "live" if mode != "replay" else "not_used",
            "steps": steps, "outcome": run["outcome"], "grasp": run["grasp_established"],
            "simulated_task_success": run["task_success"], "autonomous_success": mode == "policy" and run["task_success"],
            "minimum_actual_gripper_mm": float(actual[:, 15].min() * 1000),
            "right_joint_mae_vs_recording_rad": float(np.abs(difference[:, 8:15]).mean()),
            "gripper_mae_vs_recording_mm": float(np.abs(difference[:, 15]).mean() * 1000),
            "verified_recorded_input_chunks": verified_chunks, "decoded_video_frames": frames,
            "video": str(folder / "rollout.mp4")})
    if not args.plot_only:
        args.output.mkdir(parents=True)
    result = {"episode": report["episode"], "checkpoint_weights_sha256": report["checkpoint_weights_sha256"],
              "source_sha256": report["source_sha256"], "conditions": rows,
              "interpretation": "Actual simulator motion under predicted actions. Recorded observations assist the actor only; never force simulator state. Assisted task success is not autonomous success.",
              "limits": "One held-out reset, one noise seed, original source-matched simulator; not new-scene or sim2real validation."}
    if not args.plot_only:
        (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        with (args.output / "conditions.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.verify_only:
        print(json.dumps(result, indent=2), flush=True)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.subplots_adjust(left=.085, right=.98, bottom=.1, top=.84, hspace=.24)
    axes[0].plot(reference_time, reference[:, 15] * 1000, "k--", linewidth=1.5, label="Recorded expert state")
    axes[1].plot(reference_time[:-1], targets[:, 7] * 1000, "k--", linewidth=1.5, label="Recorded expert applied target")
    names = {"policy": "VLA: live inputs", "recorded-state": "ASSISTED VLA: recorded state / live RGB",
             "recorded-observations": "ASSISTED VLA: recorded state + RGB"}
    colors = {"policy": "#68737d", "recorded-state": "#d9822b", "recorded-observations": "#2471a3"}
    for mode, trace in traces.items():
        if mode == "replay":
            continue  # Numerically checked replay is listed in CSV, not duplicated over reference.
        times = np.r_[0., np.cumsum(trace["executed_dt_s"])]
        axes[0].plot(times, trace["observation_state"][:, 15] * 1000, color=colors[mode], label=names[mode])
        axes[1].plot(times[:-1], trace["action_requested"][:, 7] * 1000, color=colors[mode], label=names[mode])
        axes[0].plot(times[-1], trace["observation_state"][-1, 15] * 1000, "x", color=colors[mode])
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    axes[0].set(ylabel="Actual measured opening (mm)")
    axes[1].set(ylabel="Commanded gripper target (mm)", xlabel="Simulated time (s)")
    fig.suptitle("Recorded-input intervention: physical rollout, not state playback\nPrivileged input modes are diagnostics, not autonomous policies", fontsize=13)
    fig.savefig(args.output / "gripper_intervention.png", dpi=160)
    fig.savefig(args.output / "gripper_intervention.pdf")
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
