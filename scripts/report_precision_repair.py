"""Audit the confirmed precision defect and actual repaired-policy outcomes."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.smolvla_data import sha256
from learning.vla.gripper_feedback import open_hold_anchors
from scripts.report_gripper_feedback import audit_updates, feedback_metrics


def audit_rollout(report, *, label, base_sha, source_sha, overlay_sha, grasp_metrics):
    """Reject mismatched conditions, not just an apparently successful video."""
    repaired = label.startswith("repaired_prefix")
    expected_prefix = (int(label.removeprefix("repaired_prefix")) if repaired else
                       int(label.removeprefix("original_prefix")) if label.startswith("original_prefix") else 5)
    if (report["recorded_input_assistance"] or report["simulator_state_forced_to_recording"]
            or not report["pipeline_passed"] or report["reset_state_max_error"] != 0
            or report["mode"] != ("replay" if label == "expert" else "policy")
            or report["policy_seed"] != 1000 or report["execution_prefix"] != expected_prefix
            or report["checkpoint_weights_sha256"] != base_sha
            or report["source_data_sha256"] != source_sha
            or report["grasp_metrics"] != grasp_metrics
            or grasp_metrics["revision"] != "physical-contact-settle-v1"
            or grasp_metrics["physics_or_guard_thresholds_changed"]
            or report["actuation_filter"] != {"gripper_resolution_mm": 0.,
                "reference_or_phase_inputs": False, "arm_targets_changed": False}
            or report["expert_overlay_weights_sha256"] != (overlay_sha if repaired else None)):
        raise ValueError("Rollout source/reset/model/assistance/physics conditions differ")
    if label == "expert" and (not report["task_success"] or not report["grasp_established"]
                               or max(report["replay_errors"].values()) != 0):
        raise ValueError("Physical expert replay did not match")


def motion_landmarks(trace, reference, reference_actions):
    """Post-hoc joint-space alignment only, never an input to policy control."""
    state, requested = trace["observation_state"], trace["action_requested"]
    result = {}
    for label, values in (("first_measured_below_30mm", state[:, 15]),
                          ("first_requested_below_30mm", requested[:, 7])):
        hits = np.flatnonzero(values < .030)
        if not len(hits):
            result[label] = None
            continue
        i = int(hits[0])
        distances = np.sqrt(np.mean((reference[:, 8:15] - state[i, 8:15]) ** 2, axis=1))
        nearest = int(np.argmin(distances))
        result[label] = {"trace_index": i, "nearest_expert_right_joint_index": nearest,
            "nearest_expert_joint_rmse_rad": float(distances[nearest]),
            "nearest_expert_measured_gripper_mm": float(reference[nearest, 15] * 1000),
            "nearest_expert_target_gripper_mm": float(reference_actions[min(nearest, len(reference_actions) - 1), 7] * 1000),
            "actual_measured_gripper_mm": float(state[i, 15] * 1000)}
    result["interpretation"] = "Nearest recorded right-joint configuration, not time synchronization or a unique task-phase assignment; post-hoc diagnostic only."
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--matched-control", type=Path, help="Optional completed same-prefix original-weight control")
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--verify-only", action="store_true")
    modes.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    folder = args.experiment / "precision_repair"
    run = json.loads((folder / "report.json").read_text())
    training = json.loads((folder / "training/report.json").read_text())
    reload = json.loads((folder / "training/verification.json").read_text())
    defect = json.loads((args.experiment / "precision_issue.json").read_text())
    if not run["completed"] or not reload["passed"] or reload["prediction_max_abs_difference"] != 0:
        raise ValueError("Completed/reloaded repair required")
    for name, expected in run["plan"]["implementation"].items():
        from learning.vla.source_archive import verified_source
        verified_source(name, expected)
    episode = ROOT / "data/drive_v2c30_20261012/merged_all_v1/train_20002"
    metadata = json.loads((episode / "metadata.json").read_text())
    if sha256(episode / "episode.npz") != metadata["data_sha256"]:
        raise ValueError("Source changed")
    with np.load(episode / "episode.npz", allow_pickle=False) as raw:
        reference, actions, times = (raw[k].copy() for k in ("observation_state", "action_applied", "timestamp_s"))
    times -= times[0]
    updates = [json.loads(line) for line in (folder / "training/updates.jsonl").read_text().splitlines()]
    exposure = audit_updates(updates, open_hold_anchors(reference, actions), len(actions), True)
    if (training["overlay"]["trainable_storage"] != "float32" or training["overlay"]["parameter_count"] != 99880992
            or training["binding"]["precision"]["after"] != {"torch.float32": 99880992}
            or training["binding"]["episode"]["data_sha256"] != metadata["data_sha256"]
            or training["binding"]["episode"]["split"] != "train"
            or training["binding"]["eligible_indices"] != open_hold_anchors(reference, actions)
            or training["binding"]["mode"] != "augmented"
            or training["updates"] != 800 or training["stop_reason"] != "step_budget"
            or sha256(folder / "training/expert/expert.safetensors") != training["overlay"]["weights_sha256"]):
        raise ValueError("Wrong trained parameter storage/source/schedule/weights")
    if args.verify_only:
        from safetensors import safe_open
        with safe_open(str(folder / "training/expert/expert.safetensors"), framework="pt", device="cpu") as weights:
            if any(weights.get_slice(key).get_dtype() != "F32" for key in weights.keys()):
                raise ValueError("Saved trainable expert is not FP32")
        import av
    conditions = {"expert": ("expert_physical_metrics", run["expert"]),
                  "original_vla": ("baseline_physical_metrics", run["baseline"]),
                  **{"repaired_prefix" + key: ("repaired_prefix" + key, value) for key, value in run["rollouts"].items()}}
    if args.matched_control:
        control = json.loads((args.matched_control / "report.json").read_text())
        if (not control["completed"] or control["repair_report_sha256"] != sha256(folder / "report.json")
                or control["repaired"] != run["rollouts"][str(control["prefix"]) ]):
            raise ValueError("Matched control is bound to a different repair")
        conditions["original_prefix" + str(control["prefix"])] = (
            str(args.matched_control.resolve() / "rollout"), control["baseline"])
    traces, rows, landmarks = {}, [], {}
    for label, (path, report) in conditions.items():
        audit_rollout(report, label=label, base_sha=run["plan"]["base_sha256"],
            source_sha=metadata["data_sha256"], overlay_sha=training["overlay"]["weights_sha256"],
            grasp_metrics=run["expert"]["grasp_metrics"])
        with np.load(folder / path / "trajectory.npz", allow_pickle=False) as raw:
            trace = {key: raw[key].copy() for key in ("observation_state", "action_requested", "executed_dt_s")}
            if not np.array_equal(raw["policy_action_raw"], trace["action_requested"]):
                raise ValueError("Commands were post-filtered")
        if trace["observation_state"].shape != (report["steps"] + 1, 16) or not all(np.isfinite(v).all() for v in trace.values()):
            raise ValueError("Invalid physical trace")
        frames = None
        if args.verify_only:
            with av.open(str(folder / path / "rollout.mp4")) as video:
                frames = sum(1 for _ in video.decode(video=0))
            if frames != report["steps"] + 1:
                raise ValueError("Incomplete rollout video")
        diagnostics = json.loads((folder / path / "diagnostics.json").read_text())
        first_grasp = next((i for i, d in enumerate(diagnostics) if d["grasp_established"]), None)
        rows.append({"condition": label, "prefix": report["execution_prefix"], "steps": report["steps"],
            "outcome": report["outcome"], "grasp": report["grasp_established"], "insertion": report["task_success"],
            "first_grasp_command": first_grasp, "drops": diagnostics[-1]["drops"],
            "final_y_offset_mm": diagnostics[-1]["offset_y_m"] * 1000,
            "final_z_offset_mm": diagnostics[-1]["offset_z_m"] * 1000,
            "final_orientation_error_deg": diagnostics[-1]["orientation_error_deg"],
            "minimum_gripper_mm": float(trace["observation_state"][:, 15].min() * 1000),
            "max_plug_finger_contacts": max(r["plug_finger_contacts"] for r in diagnostics),
            "decoded_video_frames": frames, "video": str(folder / path / "rollout.mp4")})
        traces[label] = trace
        landmarks[label] = motion_landmarks(trace, reference, actions)
    if args.verify_only:
        args.output.mkdir(parents=True, exist_ok=False)
        summary = {"precision_issue": defect, "conditions": rows, "motion_landmarks": landmarks, "exposure": exposure,
            "training_initial": feedback_metrics(training["initial"], actions),
            "training_final": feedback_metrics(training["final"], actions),
            "training_precision_audit": training["precision_update_audit"],
            "reload_prediction_max_abs_error": reload["prediction_max_abs_difference"],
            "training_elapsed_s": training["elapsed_s"], "repair_elapsed_s": run["elapsed_s"],
            "limits": "Combined FP32 optimizer-storage repair and training-time augmentation; not an isolated augmentation comparison. Explicit physical-contact grasp bookkeeping extension; unchanged underlying physics/guards. One training reset/seed and declared chunk tests, not generalization."}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        with (args.output / "conditions.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps({k: v for k, v in summary.items() if k != "precision_issue"}), flush=True)
        return
    if not (args.output / "summary.json").is_file():
        raise ValueError("Verify before rendering")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.subplots_adjust(left=.08, right=.98, bottom=.09, top=.87, hspace=.37, wspace=.28)
    evidence = defect["real_gradient_gpu_audit"]
    bars = axes[0, 0].bar(["Old BF16 storage", "Correct FP32 storage"],
        [evidence["same_gradient_bf16_changed_fraction"] * 100, evidence["actual_fp32_changed_fraction"] * 100],
        color=["#68737d", "#2471a3"])
    axes[0, 0].bar_label(bars, fmt="%.2f%%", padding=3)
    axes[0, 0].set(title="Same real gradient: effective parameter updates", ylabel="Values changed in one expert matrix (%)", ylim=(0, 112))
    checks = [training["initial"], *[c for c in training["evaluations"] if c["step"] > 0 and c["step"] < 800], training["final"]]
    axes[0, 1].plot([c["step"] for c in checks], [c["metrics"]["perturbed_open_hold"]["gripper_mae_mm"] for c in checks], "o-", label="FP32 storage + sensor augmentation")
    axes[0, 1].set(title="Offline two-mm sensor-perturbation probes", xlabel="Optimizer updates", ylabel="Prefix-five gripper MAE (mm)")
    axes[1, 0].plot(times, reference[:, 15] * 1000, "k--", label="Recorded expert")
    axes[1, 1].plot(times[:-1], actions[:, 7] * 1000, "k--", label="Recorded expert")
    for label, trace in traces.items():
        if label == "expert":
            continue
        t = np.r_[0., np.cumsum(trace["executed_dt_s"])]
        axes[1, 0].plot(t, trace["observation_state"][:, 15] * 1000, label=label)
        axes[1, 1].plot(t[:-1], trace["action_requested"][:, 7] * 1000, label=label)
    axes[1, 0].set(title="Live-input physical rollouts", xlabel="Simulated time (s)", ylabel="Measured finger travel (mm)")
    axes[1, 1].set(title="Unfiltered continuous gripper commands", xlabel="Simulated time (s)", ylabel="Requested target (mm)")
    for axis in axes.flat:
        axis.grid(alpha=.2)
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.legend(fontsize=7)
    fig.suptitle("Training precision repair and source-physics-matched simulation checks\nGrasp bookkeeping extension is explicit; this is not held-out evaluation", fontsize=12)
    fig.savefig(args.output / "precision_repair.png", dpi=160)
    fig.savefig(args.output / "precision_repair.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
