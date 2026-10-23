"""Recover terminal artifacts from a complete saved VLA trace; never rerun policy."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.runtime_io import write_json
from learning.vla.smolvla_data import sha256
from learning.vla.pickup_evaluation import PickupGate, audit_pickup_trace
from learning.vla.source_archive import archive_source
from scripts.rollout_smolvla import verify_snapshot
from scripts.train_gripper_feedback import BASE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--case", required=True)
    args = p.parse_args()
    started = time.monotonic()
    import numpy as np
    import mujoco
    import av
    evaluation = args.evaluation.resolve()
    folder = evaluation / (args.case + "_policy")
    if (folder / "report.json").exists() or (folder / "rollout_recovered.mp4").exists():
        raise ValueError("Do not overwrite existing terminal/recovery artifacts")
    plan = json.loads((evaluation / "plan.json").read_text())
    record = next(r for r in plan["parents"] if r["episode"] == args.case)
    source = ROOT / "data/drive_v2c30_20261012/merged_all_v1" / args.case
    meta = json.loads((source / "metadata.json").read_text())
    snapshot = ROOT / "third_party/capstone_automateData"
    workspace = verify_snapshot(snapshot, meta)
    if (sha256(source / "episode.npz") != record["data_sha256"]
            or sha256(BASE / "model.safetensors") != plan["base_weights_sha256"]
            or sha256(ROOT / "outputs/gripper_control_fix_20261019/precision_repair/training/expert/expert.safetensors") != plan["overlay_weights_sha256"]):
        raise ValueError("Frozen source/model changed")
    original_files = ("trajectory.npz", "chunks.json", "diagnostics.json", "manifest.json", "rollout.gif", "rollout.mp4")
    original_hashes = {name: sha256(folder / name) for name in original_files}
    with np.load(folder / "trajectory.npz", allow_pickle=False) as raw:
        trace = {k: raw[k].copy() for k in raw.files}
    diagnostics = json.loads((folder / "diagnostics.json").read_text())
    chunks = json.loads((folder / "chunks.json").read_text())
    cursor, predictions = 0, []
    for chunk in chunks:
        predicted = np.asarray(chunk["prediction"])
        if (chunk["start_step"] != cursor or predicted.shape != (50, 8)
                or chunk["executed_steps"] != 25 or "recorded_observation_index" in chunk
                or not np.isfinite(predicted).all()):
            raise ValueError("Not a complete live-input prefix-25 chunk sequence")
        cursor += chunk["executed_steps"]
        predictions.extend(predicted[:chunk["executed_steps"]])
    if (cursor != 1200 or len(diagnostics) != 1201 or trace["qpos"].shape[0] != 1201
            or not np.array_equal(predictions, trace["policy_action_raw"])
            or diagnostics[-1]["outcome"] != "running"):
        raise ValueError("Physical execution did not finish its fixed command budget")
    sys.path.insert(0, str(snapshot))
    from envs.openarm_insert import OpenArmInsertEnv
    env = OpenArmInsertEnv(images=True, workspace=workspace)
    hard_deadline = threading.Timer(240, lambda: os._exit(124))
    hard_deadline.daemon = True
    hard_deadline.start()
    scene, wrist = [], []
    try:
        initial, _ = env.reset(seed=meta["seed"], options=meta["reset_options"])
        if not np.array_equal(initial["state"], trace["observation_state"][0]):
            raise ValueError("Source reset changed")
        gate = PickupGate(float(trace["qpos"][0, env.plug_qadr + 2]))
        for i, info in enumerate(diagnostics[1:], 1):
            gate.update(plug_z_m=float(trace["qpos"][i, env.plug_qadr+2]), grasp=info["grasp_established"],
                contacts=info["plug_finger_contacts"], drops=info["drops"], safe=info["outcome"] in ("running", "success"),
                dt=float(trace["executed_dt_s"][i-1]))
        if gate.passed:
            raise ValueError("Unexpected successful pickup; inspect before recovering a failure endpoint")
        base_time = float(env.data.time)
        times = np.r_[0., np.cumsum(trace["executed_dt_s"])]
        # Rendering saved states is artifact reconstruction only. No policy,
        # env.step or physics integration runs here; original trace is immutable.
        for i in range(1201):
            if time.monotonic() - started > 180:
                raise TimeoutError("Bounded artifact reconstruction exceeded 180 seconds")
            env.data.qpos[:] = trace["qpos"][i]
            env.data.qvel[:] = trace["qvel"][i]
            env.data.time = base_time + times[i]
            mujoco.mj_forward(env.model, env.data)
            obs = env.observe()
            if not np.array_equal(obs["state"], trace["observation_state"][i]):
                raise ValueError("Rendered state differs from saved physical state")
            scene.append(obs["images"]["scene_rgb"])
            wrist.append(obs["images"]["wrist_rgb"])
        from learning.vla.rollout_video import write_video
        write_video(folder / "rollout_recovered.mp4", scene, wrist, mode="policy", episode=args.case,
            checkpoint="POST-HOC: saved VLA trajectory", outcome="step_budget_exhausted")
        with av.open(str(folder / "rollout_recovered.mp4")) as video:
            if sum(1 for _ in video.decode(video=0)) != 1201:
                raise ValueError("Recovered video frame count mismatch")
        manifest = json.loads((folder / "manifest.json").read_text())
        recovered = {"pipeline_passed": True, "mode": "policy", "safety_abort": False,
            "outcome": "step_budget_exhausted", "steps": 1200, "task_success": False,
            "autonomous_task_success": False, "autonomous_pickup_success": False,
            "grasp_established": any(v["grasp_established"] for v in diagnostics),
            "recorded_input_assistance": False, "simulator_state_forced_to_recording": False,
            "reset_state_max_error": 0., "execution_prefix": 25, "policy_seed": 1000,
            "checkpoint_weights_sha256": plan["base_weights_sha256"],
            "expert_overlay_weights_sha256": plan["overlay_weights_sha256"],
            "source_episode": args.case, "source_data_sha256": record["data_sha256"],
            "grasp_metrics": manifest["evaluation_grasp_metrics"],
            "actuation_filter": {"gripper_resolution_mm": 0., "reference_or_phase_inputs": False, "arm_targets_changed": False},
            "pickup_evaluation": {**gate.definition(), "passed": False,
                "maximum_consecutive_carry_s": gate.maximum_s, "maximum_lift_m": gate.maximum_lift_m,
                "plug_qpos_address": int(env.plug_qadr), "helper_sha256": plan["implementation"]["learning/vla/pickup_evaluation.py"]},
            "video": "rollout_recovered.mp4", "video_fps": 50, "simulated_s": float(times[-1]),
            "artifact_recovery": {"no_policy_rerun": True, "original_execution_state_assistance": False,
                "video_rendered_from_saved_qpos": True, "terminal_report_reconstructed": True,
                "original_file_sha256": original_hashes, "recovery_script_sha256": sha256(Path(__file__))},
            "limits": "Original VLA completed 1200 physical commands; encoder was killed before final report. This video is explicitly post-hoc rendering of saved original states, not a new closed-loop policy trial. Missing original wall-time/final-contact metadata is not fabricated."}
        audit_pickup_trace(recovered, trace, diagnostics)
        for name, expected in original_hashes.items():
            if sha256(folder / name) != expected:
                raise ValueError("Original terminal artifact changed")
        prior = max(json.loads(f.read_text())["elapsed_s"] for f in
            [evaluation / "failure.json", *evaluation.glob("failure_continuation_*.json")])
        elapsed = time.monotonic() - started
        recovery = {"case": args.case, "original_plan_sha256": sha256(evaluation / "plan.json"),
            "original_file_sha256": original_hashes, "recovery_script_sha256": sha256(Path(__file__)),
            "previous_elapsed_s": prior, "recovery_elapsed_s": elapsed, "cumulative_elapsed_s": prior+elapsed,
            "no_policy_rerun": True, "video_rendered_from_saved_qpos": True}
        archive_source("scripts/recover_pickup_recording.py")
        write_json(folder / "report.json", recovered)
        write_json(evaluation / "artifact_recovery_001.json", recovery)
        print(json.dumps({"recovered": args.case, "physical_steps": 1200, "pickup_success": False,
            "no_policy_rerun": True, "recovery_elapsed_s": elapsed}), flush=True)
    finally:
        hard_deadline.cancel()
        env.close()


if __name__ == "__main__":
    main()
