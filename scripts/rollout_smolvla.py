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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--prefix", type=int, default=5)
    p.add_argument("--mode", choices=("policy", "replay"), default="policy")
    p.add_argument("--seed", type=int, default=1000, help="Policy sampling seed")
    args = p.parse_args()
    if not 1 <= args.max_steps <= 2000 or not 1 <= args.prefix <= 50:
        raise ValueError("Invalid step budget or execution prefix")
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    from learning.vla.smolvla_data import sha256
    from learning.vla.smolvla_policy import SmolVLAController
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
    policy = SmolVLAController(args.checkpoint, args.backbone, args.seed) if args.mode == "policy" else None
    env = OpenArmInsertEnv(images=True, workspace=workspace)
    try:
        obs, info = env.reset(seed=metadata["seed"], options=metadata["reset_options"])
        manifest = env.manifest()
        for key in ("schema", "scene_sha256", "robot_config", "contact_config", "action_names", "state_order"):
            if manifest[key] != metadata[key]:
                raise ValueError(f"Runtime contract mismatch: {key}")
        if manifest["reset_mode"] != "tabletop":
            raise ValueError("Expected table-start task")
        with np.load(args.episode / "episode.npz", allow_pickle=False) as raw:
            initial_error = float(np.max(np.abs(obs["state"] - raw["observation_state"][0])))
            reference_actions = raw["action_applied"].copy()
            reference_qpos, reference_qvel = raw["qpos"].copy(), raw["qvel"].copy()
        if initial_error > 1e-5:
            raise ValueError(f"Reset state differs from source: {initial_error}")
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=serial))
        qpos, qvel, states = [env.data.qpos.copy()], [env.data.qvel.copy()], [obs["state"].copy()]
        frames = [obs["images"]["scene_rgb"].copy()]
        requested, applied, durations, diagnostics, chunks = [], [], [], [info], []
        outcome, phase = "step_budget_exhausted", "table_start"
        started = time.perf_counter()
        while len(applied) < args.max_steps:
            if args.mode == "policy":
                commands, latency = policy.predict(obs)
            else:
                commands = reference_actions[len(applied):len(applied) + args.prefix]
                latency = 0.
                if not len(commands):
                    outcome = "reference_actions_exhausted"
                    break
            if args.mode == "policy" and commands.shape != (50, 8):
                raise ValueError("Expected 50x8 decoded chunk")
            chunk = {"start_step": len(applied), "latency_s": latency,
                     "prediction": commands.tolist(), "executed_steps": 0}
            chunks.append(chunk)
            for command in commands[:min(args.prefix, args.max_steps - len(applied))]:
                obs, reward, term, trunc, info = env.step(command)
                requested.append(command.copy())
                applied.append(info["applied_action"].copy())
                durations.append(info["executed_dt_s"])
                diagnostics.append(info)
                states.append(obs["state"].copy())
                qpos.append(env.data.qpos.copy())
                qvel.append(env.data.qvel.copy())
                frames.append(obs["images"]["scene_rgb"].copy())
                chunk["executed_steps"] += 1
                # Privileged phase labels are diagnostics only; no controller switches use them.
                if info["grasp_established"]:
                    phase = "carrying_or_inserting"
                elif info["plug_finger_contacts"]:
                    phase = "grasp_contact"
                else:
                    phase = "pregrasp"
                if term or trunc:
                    outcome = info["outcome"]
                    break
            if term or trunc:
                break
            print(json.dumps({"steps": len(applied), "phase_diagnostic": phase,
                              "latency_s": latency, "outcome": info["outcome"]}), flush=True)
        np.savez_compressed(args.output / "trajectory.npz", observation_state=np.asarray(states),
                            qpos=np.asarray(qpos), qvel=np.asarray(qvel),
                            action_requested=np.asarray(requested), action_applied=np.asarray(applied),
                            executed_dt_s=np.asarray(durations))
        (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, default=serial))
        (args.output / "chunks.json").write_text(json.dumps(chunks, indent=2))
        from PIL import Image
        every = max(1, len(frames) // 100)
        animation = [Image.fromarray(frame) for frame in frames[::every]]
        animation[0].save(args.output / "rollout.gif", save_all=True, append_images=animation[1:],
                          duration=every * 20, loop=0)
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
                  "failure_phase_diagnostic": phase, "steps": len(applied),
                  "simulated_s": sum(durations), "wall_s": time.perf_counter() - started,
                  "execution_prefix": args.prefix, "inference_calls": len(latencies),
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
                  "source_episode": args.episode.name, "policy_seed": args.seed,
                  "limits": "One smoke rollout on a seen validation reset; synchronous paused-time inference. No real-time or task generalization claim."}
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
