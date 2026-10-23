"""Guarded closed-loop simulation using a source-hash-matched collector checkout."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def serial(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def policy_observation(live, recording, index, mode):
    """Recorded inputs assist the actor only; never change simulated qpos/qvel.

    A recording has T+1 observations, but only anchors [0,T) have actions.
    Recorded-state deliberately combines oracle proprioception with live RGB.
    Recorded-observations is a teacher-forced, externally guided action sequence.
    """
    import numpy as np
    if mode == "policy":
        return live
    if mode not in ("recorded-state", "recorded-observations"):
        raise ValueError("Unknown policy observation mode")
    states = recording["observation_state"]
    if type(index) is not int or not 0 <= index < len(states) - 1:
        raise IndexError("No recorded action anchor at this observation")
    result = {"state": np.array(states[index], copy=True), "instruction": live["instruction"]}
    result["images"] = (live["images"] if mode == "recorded-state" else
                        {camera: np.array(recording["image_" + camera][index], copy=True)
                         for camera in ("scene_rgb", "wrist_rgb")})
    return result


def verify_snapshot(snapshot, metadata):
    from learning.vla.smolvla_data import sha256
    snapshot = Path(snapshot).resolve()
    for rel, expected in metadata["source_hashes"].items():
        path = (snapshot / rel).resolve()
        if not path.is_relative_to(snapshot) or sha256(path) != expected:
            raise ValueError(f"Collector source differs from recording: {rel}")
    workspace = (snapshot / metadata["workspace"]["path"]).resolve()
    if not workspace.is_relative_to(snapshot) or sha256(workspace) != metadata["workspace"]["sha256"]:
        raise ValueError("Workspace differs from recording")
    if sha256(snapshot / "third_party/openarm_mujoco/v1/openarm_bimanual.xml") != metadata["robot"]["robot_xml_sha256"]:
        raise ValueError("Robot assets differ from recording")
    return workspace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--episode", type=Path, required=True, help="Reset/provenance; replay mode also reads applied commands")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--backbone", type=Path, required=True)
    p.add_argument("--base-export", type=Path, help="Explicit train-only task adaptation for an unmodified base model; never use with an adapted checkpoint")
    p.add_argument("--expert-overlay", type=Path, help="Optional hash-bound trainable-expert replacement; same base normalization")
    p.add_argument("--gripper-resolution-mm", type=float, choices=(0., .25, .5, 1.), default=0.,
                   help="Explicit actuator-resolution diagnostic; zero keeps original continuous targets")
    p.add_argument("--physical-grasp-metrics", action="store_true",
                   help="Explicit contact-based debounce extension; qualify expert replay separately")
    p.add_argument("--pickup-only", action="store_true",
                   help="Stop after physical grasp + 20mm lift + 1s continuous carry, without commanding a hold")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--prefix", type=int, default=5)
    p.add_argument("--mode", choices=("policy", "replay", "recorded-state", "recorded-observations"), default="policy")
    p.add_argument("--seed", type=int, default=1000, help="Policy sampling seed")
    p.add_argument('--noise-schedule',choices=('sequential','step-keyed'),default='sequential',
                   help='Opt-in cadence diagnostic: seed initial flow noise by executed-command index')
    p.add_argument('--numerics', choices=('default','controlled'), default='default',
                   help='Opt-in explicit deterministic BF16 evaluation settings; default behavior unchanged')
    p.add_argument('--verify-first-prediction', action='store_true',
                   help='Offline repeat of the first live observation/noise, before any robot command')
    args = p.parse_args()
    if not 1 <= args.max_steps <= 2000 or not 1 <= args.prefix <= 50:
        raise ValueError("Invalid step budget or execution prefix")
    if args.noise_schedule=='step-keyed' and (args.mode!='policy' or args.seed<0):
        raise ValueError('Step-keyed noise requires live policy mode and a nonnegative seed')
    if args.verify_first_prediction and (args.mode!='policy' or args.noise_schedule!='step-keyed'):
        raise ValueError('First-prediction repeat audit requires live policy and explicit step-keyed noise')
    if args.numerics=='controlled':
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import torch
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
    if args.pickup_only and (not args.physical_grasp_metrics or args.mode not in ("policy", "replay")):
        raise ValueError("Pickup gate requires declared physical grasp metrics and live policy or replay")
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    from learning.vla.smolvla_data import sha256
    from learning.vla.smolvla_policy import SmolVLAController
    from learning.vla.inference_inputs import inference_input_fingerprint
    metadata = json.loads((args.episode / "metadata.json").read_text())
    if sha256(args.episode / "episode.npz") != metadata["data_sha256"]:
        raise ValueError("Recorded episode checksum mismatch")
    workspace = verify_snapshot(args.snapshot, metadata)
    # The policy module is already bound to this repo; environment imports bind
    # to the independently verified checkout. Never mix old/new envs in a process.
    if "envs" in sys.modules or "connector" in sys.modules:
        raise RuntimeError("Environment packages were already imported; use a fresh process")
    sys.path.insert(0, str(args.snapshot.resolve()))
    from envs.openarm_insert import OpenArmInsertEnv
    if not Path(sys.modules["envs.openarm_insert"].__file__).resolve().is_relative_to(args.snapshot.resolve()):
        raise AssertionError("Wrong environment imported")
    recorded_inputs = args.mode in ("recorded-state", "recorded-observations")
    if args.expert_overlay and args.mode != "policy":
        raise ValueError("Expert overlay evaluation must use live policy inputs")
    if args.gripper_resolution_mm and args.mode != "policy":
        raise ValueError("Actuator-resolution diagnostic requires live policy inputs")
    policy = SmolVLAController(args.checkpoint, args.backbone, args.seed, args.base_export,
                              expert_overlay=args.expert_overlay) if args.mode != "replay" else None
    env = OpenArmInsertEnv(images=True, workspace=workspace)
    grasp_metrics = {"revision": "collector-original", "physics_or_guard_thresholds_changed": False}
    if args.physical_grasp_metrics:
        if args.mode not in ("policy", "replay"):
            raise ValueError("Physical grasp metric qualification requires live policy or replay")
        from learning.vla.physical_grasp import install_physical_grasp_metrics
        grasp_metrics = install_physical_grasp_metrics(env)
    try:
        obs, info = env.reset(seed=metadata["seed"], options=metadata["reset_options"])
        from learning.vla.pickup_evaluation import PickupGate
        pickup_gate = PickupGate(float(env.data.qpos[env.plug_qadr + 2])) if args.pickup_only else None
        pickup_done = False
        manifest = env.manifest()
        for key in ("schema", "scene_sha256", "robot_config", "contact_config", "action_names", "state_order"):
            if manifest[key] != metadata[key]:
                raise ValueError(f"Runtime contract mismatch: {key}")
        if manifest["reset_mode"] != "tabletop":
            raise ValueError("Expected table-start task")
        manifest["evaluation_grasp_metrics"] = grasp_metrics
        with np.load(args.episode / "episode.npz", allow_pickle=False) as raw:
            initial_error = float(np.max(np.abs(obs["state"] - raw["observation_state"][0])))
            reference_actions = raw["action_applied"].copy()
            reference_qpos, reference_qvel = raw["qpos"].copy(), raw["qvel"].copy()
            recording = {"observation_state": raw["observation_state"].copy()} if recorded_inputs else None
            if args.mode == "recorded-observations":
                recording.update({key: raw[key].copy() for key in ("image_scene_rgb", "image_wrist_rgb")})
        if initial_error > 1e-5:
            raise ValueError(f"Reset state differs from source: {initial_error}")
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=serial))
        qpos, qvel, states = [env.data.qpos.copy()], [env.data.qvel.copy()], [obs["state"].copy()]
        frames = [obs["images"]["scene_rgb"].copy()]
        wrist_frames = [obs["images"]["wrist_rgb"].copy()]
        requested, policy_raw, applied, durations, diagnostics, chunks = [], [], [], [], [info], []
        outcome, phase = "step_budget_exhausted", "table_start"
        started = time.perf_counter()
        repeat_audit = None
        while len(applied) < args.max_steps:
            if recorded_inputs and len(applied) >= len(reference_actions):
                outcome = "reference_observations_exhausted"
                break
            if policy is not None:
                actor = policy_observation(obs, recording, len(applied), args.mode)
                input_fingerprint = inference_input_fingerprint(actor)
                if not applied:
                    # Evidence only: these arrays never replace live observations.
                    np.savez_compressed(args.output / 'first_actor_input.npz',
                        state=actor['state'], instruction=np.asarray(actor['instruction']),
                        **{key: actor['images'][key] for key in ('scene_rgb', 'wrist_rgb')})
                if args.noise_schedule=='step-keyed':
                    from learning.vla.inference_cadence import anchor_noise_seed
                    commands,latency=policy.predict(actor,noise_seed=anchor_noise_seed(args.seed,len(applied)))
                else:
                    commands, latency = policy.predict(actor)
                if args.verify_first_prediction and not applied:
                    repeated, repeated_latency = policy.predict(actor, noise_seed=args.seed)
                    error = float(np.max(np.abs(commands-repeated)))
                    repeat_audit = {'max_abs_difference': error, 'additional_inference_calls': 1,
                                    'latency_s': repeated_latency, 'before_any_command': True,
                                    'same_live_observation': True, 'noise_seed': args.seed}
                    if error > 1e-6:
                        raise ValueError('First live observation/noise prediction is not repeatable')
            else:
                commands = reference_actions[len(applied):len(applied) + args.prefix]
                latency = 0.
                if not len(commands):
                    outcome = "reference_actions_exhausted"
                    break
            if policy is not None and commands.shape != (50, 8):
                raise ValueError("Expected 50x8 decoded chunk")
            chunk = {"start_step": len(applied), "latency_s": latency,
                     "prediction": commands.tolist(), "executed_steps": 0}
            if policy is not None:
                chunk['input_fingerprint'] = input_fingerprint
            if args.noise_schedule=='step-keyed':
                chunk['noise_seed']=anchor_noise_seed(args.seed,len(applied))
            if recorded_inputs:
                chunk.update(input_mode=args.mode, recorded_observation_index=len(applied),
                    policy_input_state=actor["state"].tolist(),
                    input_vs_actual_state_max_abs=float(np.max(np.abs(actor["state"] - obs["state"]))),
                    image_source="recorded" if args.mode == "recorded-observations" else "live_simulation")
            chunks.append(chunk)
            available = min(args.prefix, args.max_steps - len(applied))
            if recorded_inputs:
                available = min(available, len(reference_actions) - len(applied))
            for command in commands[:available]:
                policy_raw.append(command.copy())
                if args.gripper_resolution_mm:
                    from learning.vla.gripper_feedback import quantize_finger_command
                    command = quantize_finger_command(command, args.gripper_resolution_mm / 1000)
                obs, reward, term, trunc, info = env.step(command)
                requested.append(command.copy())
                applied.append(info["applied_action"].copy())
                durations.append(info["executed_dt_s"])
                diagnostics.append(info)
                states.append(obs["state"].copy())
                qpos.append(env.data.qpos.copy())
                qvel.append(env.data.qvel.copy())
                frames.append(obs["images"]["scene_rgb"].copy())
                wrist_frames.append(obs["images"]["wrist_rgb"].copy())
                chunk["executed_steps"] += 1
                # Privileged phase labels are diagnostics only; no controller switches use them.
                if info["grasp_established"]:
                    phase = "carrying_or_inserting"
                elif info["plug_finger_contacts"]:
                    phase = "grasp_contact"
                else:
                    phase = "pregrasp"
                if pickup_gate:
                    pickup_done = pickup_gate.update(plug_z_m=float(env.data.qpos[env.plug_qadr + 2]),
                        grasp=info["grasp_established"], contacts=info["plug_finger_contacts"], drops=info["drops"],
                        safe=info["outcome"] in ("running", "success"), dt=info["executed_dt_s"])
                if term or trunc:
                    outcome = info["outcome"]
                    break
                if pickup_done:
                    outcome = "pickup_success"
                    break
            if term or trunc or pickup_done:
                break
            print(json.dumps({"steps": len(applied), "phase_diagnostic": phase,
                              "latency_s": latency, "outcome": info["outcome"]}), flush=True)
        np.savez_compressed(args.output / "trajectory.npz", observation_state=np.asarray(states),
                            qpos=np.asarray(qpos), qvel=np.asarray(qvel),
                            action_requested=np.asarray(requested), action_applied=np.asarray(applied),
                            policy_action_raw=np.asarray(policy_raw),
                            executed_dt_s=np.asarray(durations))
        (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, default=serial))
        (args.output / "chunks.json").write_text(json.dumps(chunks, indent=2))
        from PIL import Image
        every = max(1, len(frames) // 100)
        animation = [Image.fromarray(frame) for frame in frames[::every]]
        animation[0].save(args.output / "rollout.gif", save_all=True, append_images=animation[1:],
                          duration=every * 20, loop=0)
        from learning.vla.rollout_video import write_video
        write_video(args.output / "rollout.mp4", frames, wrist_frames, mode=args.mode,
                    episode=args.episode.name,
                    checkpoint="FP32 adapted VLA" if args.expert_overlay else args.checkpoint.parent.name, outcome=outcome)
        latencies = [r["latency_s"] for r in chunks] if policy is not None else []
        import mujoco
        contacts, robot_contacts = [], []
        for n, contact in enumerate(env.data.contact):
            wrench = np.zeros(6)
            mujoco.mj_contactForce(env.model, env.data, n, wrench)
            force = float(np.linalg.norm(wrench[:3]))
            if force > .1:
                bodies = [int(env.model.geom_bodyid[int(g)]) for g in contact.geom]
                record = {"geoms": [env.model.geom(int(g)).name for g in contact.geom],
                          "bodies": [env.model.body(b).name for b in bodies],
                          "force_n": force, "distance_m": float(contact.dist)}
                contacts.append(record)
                if any(b in env.robot_bodies for b in bodies):
                    robot_contacts.append(record)
        replay_errors = None
        if args.mode == "replay":
            replay_errors = {"qpos_max": float(np.max(np.abs(np.asarray(qpos) - reference_qpos[:len(qpos)]))),
                             "qvel_max": float(np.max(np.abs(np.asarray(qvel) - reference_qvel[:len(qvel)])))}
        report = {"pipeline_passed": len(applied) > 0 and outcome != "numerical_failure",
                  "pipeline_passed_definition": "Commands executed through the interface; not task success or safety qualification",
                  "safety_abort": outcome in ("numerical_failure", "robot_collision", "table_collision",
                                               "penetration_abort", "force_limit_abort", "leaf_force_abort",
                                               "grasp_slip", "grasp_rotation"),
                  "mode": args.mode, "final_contacts": sorted(contacts, key=lambda c: -c["force_n"])[:10],
                  "final_robot_contacts": sorted(robot_contacts, key=lambda c: -c["force_n"])[:10],
                  "replay_errors": replay_errors,
                  "task_success": outcome == "success", "outcome": outcome,
                  "autonomous_task_success": args.mode == "policy" and outcome == "success",
                  "autonomous_pickup_success": args.mode == "policy" and outcome == "pickup_success",
                  "pickup_evaluation": ({**pickup_gate.definition(), "passed": pickup_done,
                      "maximum_consecutive_carry_s": pickup_gate.maximum_s,
                      "maximum_lift_m": pickup_gate.maximum_lift_m,
                      "plug_qpos_address": int(env.plug_qadr),
                      "helper_sha256": sha256(ROOT / "learning/vla/pickup_evaluation.py")}
                      if pickup_gate else None),
                  "recorded_input_assistance": recorded_inputs,
                  "source_data_sha256": metadata["data_sha256"],
                  "input_state_source": "recorded" if recorded_inputs else "live_simulation",
                  "input_image_source": "recorded" if args.mode == "recorded-observations" else "live_simulation",
                  "simulator_state_forced_to_recording": False,
                  "failure_phase_diagnostic": phase, "steps": len(applied),
                  "simulated_s": sum(durations), "wall_s": time.perf_counter() - started,
                  "execution_prefix": args.prefix, "inference_calls": len(latencies),
                  "inference_noise_schedule":args.noise_schedule,
                  "inference_numerics":args.numerics,
                  "inference_repeat_audit":repeat_audit,
                  "torch_inference_flags": ({
                      "deterministic_algorithms": policy.torch.are_deterministic_algorithms_enabled(),
                      "cudnn_benchmark": policy.torch.backends.cudnn.benchmark,
                      "matmul_allow_tf32": policy.torch.backends.cuda.matmul.allow_tf32,
                      "cublas_workspace_config": os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                      "forward_dtype": "torch.bfloat16"} if policy else None),
                  "inference_input_audit": ({"format": "openarm-raw-actor-input-v1",
                      "first_input_file": "first_actor_input.npz",
                      "first_input_sha256": sha256(args.output / 'first_actor_input.npz'),
                      "helper_sha256": sha256(ROOT / 'learning/vla/inference_inputs.py'),
                      "per_chunk_fingerprints": True, "observations_modified": False} if policy else None),
                  "latency_mean_s": float(np.mean(latencies)) if latencies else None,
                  "latency_warm_mean_s": float(np.mean(latencies[1:])) if len(latencies) > 1 else None,
                  "latency_max_s": max(latencies) if latencies else None,
                  "latency_scope": "Model plus action postprocessing, excluding observation preprocessing and rendering",
                  "prefix_duration_s": args.prefix * env.dt,
                  "latency_within_prefix_fraction": float(np.mean(np.asarray(latencies) <= args.prefix * env.dt)) if latencies else None,
                  "commands_clipped": sum(bool(i["action_clipped"]) for i in diagnostics[1:]),
                  "grasp_established": any(i["grasp_established"] for i in diagnostics),
                  "peak_socket_force_n": float(env.peak_force),
                  "peak_penetration_m": float(env.peak_penetration),
                  "peak_robot_unwanted_contact_n": float(env.peak_robot_contact),
                  "peak_robot_support_contact_n": float(env.peak_support_contact),
                  "left_park_error_max_rad": max(i["left_park_error_rad"] for i in diagnostics),
                  "reset_state_max_error": initial_error,
                  "checkpoint_weights_sha256": sha256(args.checkpoint / "model.safetensors"),
                  "expert_overlay_weights_sha256": policy.binding.get("expert_overlay_weights_sha256") if policy else None,
                  "actuation_filter": {"gripper_resolution_mm": args.gripper_resolution_mm,
                      "reference_or_phase_inputs": False, "arm_targets_changed": False},
                  "grasp_metrics": grasp_metrics,
                  "inference_binding": policy.binding if policy is not None else None,
                  "source_episode": args.episode.name, "policy_seed": args.seed,
                  "video": "rollout.mp4", "video_fps": 50,
                  "video_timebase": "1x simulated time; synchronous inference pauses omitted; final partial interval rounded to one video frame",
                  "limits": ("Privileged recorded-input diagnostic, not autonomous success. Actual simulator evolves only through env.step; no state teleportation. "
                             if recorded_inputs else "One source-recorded reset; split membership must be checked by the experiment. ") +
                            "Synchronous paused-time inference; no real-time, generalization or sim2real claim."}
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
