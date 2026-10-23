"""Bounded offline expert-input inference and physical command-playback diagnostics.

No optimization, new controller, source-scene changes, or robot hardware. The
teacher-input condition is explicitly assisted and cannot establish autonomy.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", str(ROOT / "data/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data/hf_cache/datasets"))
from learning.vla.runtime_io import write_json
from learning.vla.smolvla_data import sha256, observation
from learning.vla.source_archive import archive_source
from learning.vla.pickup_diagnosis import unfold_prefixes, command_errors

CAMPAIGN = ROOT / "outputs/fp32_multiepisode_20261020"
COLLECTION = ROOT / "data/drive_v2c30_20261012/merged_all_v1"
EXPORT = ROOT / "data/drive_v2c30_20261012/export_all_v1"
SNAPSHOT = ROOT / "third_party/capstone_automateData"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=900)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.parent != ROOT / "outputs" or not 120 <= args.max_seconds <= 1200:
        raise ValueError("New direct outputs directory and a <=20-minute diagnostic required")
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()

    def budget():
        if time.monotonic()-started >= args.max_seconds-20:
            raise TimeoutError("Bounded diagnostic time allowance exhausted")

    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from safetensors.torch import load_file
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from learning.vla.smolvla_policy import SmolVLAController
    from learning.vla.training_state import capture_rng, restore_rng
    from learning.vla.physical_grasp import install_physical_grasp_metrics
    from learning.vla.pickup_evaluation import PickupGate
    from scripts.rollout_smolvla import verify_snapshot
    trained = json.loads((CAMPAIGN / "training/report.json").read_text())
    bundle = Path(trained["final_checkpoint"]) / "pretrained_model"
    original = json.loads((CAMPAIGN / "pickup_evaluation/plan.json").read_text())
    if sha256(bundle / "model.safetensors") != original["base_weights_sha256"]:
        raise ValueError("Original final checkpoint changed")
    source_files = ("scripts/diagnose_fp32_pickup.py", "learning/vla/pickup_diagnosis.py",
        "learning/vla/smolvla_policy.py", "learning/vla/physical_grasp.py")
    sources = {name: archive_source(name) for name in source_files}
    instruction = json.loads((COLLECTION / "manifest.json").read_text())["instruction"]
    receipt = json.loads((EXPORT / "openarm_export.json").read_text())
    pre_stats = load_file(str(bundle / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post_stats = load_file(str(bundle / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    export_stats = json.loads((EXPORT / "train/meta/stats.json").read_text())
    normalization = {}
    for feature in ("observation.state", "action"):
        normalization[feature] = {stat: float(np.max(np.abs(pre_stats[feature+"."+stat].numpy()
            -np.asarray(export_stats[feature][stat])))) for stat in ("mean", "std")}
    normalization["action_pre_post_max_abs"] = max(float((pre_stats["action."+stat]-post_stats["action."+stat]).abs().max())
        for stat in ("mean", "std"))
    write_json(args.output / "plan.json", {"checkpoint": str(bundle), "weights_sha256": original["base_weights_sha256"],
        "episodes": ["train_20002", "train_20004"], "groups": ["training_control", "validation"],
        "prefix": 25, "seed": 1000, "max_commands": 1200, "max_seconds": args.max_seconds,
        "sources": sources, "training": False, "normalization": normalization,
        "limits": "Two predeclared resets, assisted recorded-input diagnostic, not autonomous success or generalization."})
    controller = SmolVLAController(bundle, ROOT / "data/models/smolvlm_backbone", 1000)
    initial_rng = capture_rng()
    sys.path.insert(0, str(SNAPSHOT))
    from envs.openarm_insert import OpenArmInsertEnv
    import mujoco
    if not Path(sys.modules["envs.openarm_insert"].__file__).resolve().is_relative_to(SNAPSHOT):
        raise ValueError("Wrong simulator imported")
    rows = []
    for episode, split in (("train_20002", "train"), ("train_20004", "validation")):
        budget()
        folder = args.output / episode
        folder.mkdir()
        meta = json.loads((COLLECTION / episode / "metadata.json").read_text())
        workspace = verify_snapshot(SNAPSHOT, meta)
        if sha256(COLLECTION / episode / "episode.npz") != meta["data_sha256"]:
            raise ValueError("Source recording changed")
        live_dir = CAMPAIGN / "pickup_evaluation" / (episode+"_policy")
        live_chunks = json.loads((live_dir / "chunks.json").read_text())
        live_report = json.loads((live_dir / "report.json").read_text())
        if live_report["checkpoint_weights_sha256"] != original["base_weights_sha256"] or live_report["execution_prefix"] != 25:
            raise ValueError("Live comparator differs")
        with np.load(live_dir / "trajectory.npz", allow_pickle=False) as data:
            live = {key: data[key].copy() for key in data.files}
        with np.load(COLLECTION / episode / "episode.npz", allow_pickle=False) as data:
            raw = {key: data[key].copy() for key in ("observation_state", "action_applied", "qpos", "phase", "image_scene_rgb", "image_wrist_rgb")}
        total = min(1200, len(raw["action_applied"]))
        anchors, predictions = np.arange(0, total, 25), []
        restore_rng(initial_rng)
        controller.model.reset()
        for index in anchors:
            budget()
            obs = {"state": raw["observation_state"][index], "instruction": instruction,
                "images": {cam: raw["image_"+cam][index] for cam in ("scene_rgb", "wrist_rgb")}}
            prediction, latency = controller.predict(obs)
            predictions.append(prediction)
            event = {"episode": episode, "recorded_anchor": int(index), "inference_s": latency,
                "first_finger_target_mm": 1000*float(prediction[0, 7])}
            with (args.output / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event)+"\n")
            if index % 100 == 0:
                print(json.dumps(event), flush=True)
        predictions = np.asarray(predictions)
        teacher_commands = unfold_prefixes(anchors, predictions, total)
        np.savez_compressed(folder / "recorded_predictions.npz", anchors=anchors, predictions=predictions,
            executed_prefix_commands=teacher_commands)
        live_commands = live["action_requested"][:total]
        first_chunk_error = float(np.max(np.abs(predictions[0]-np.asarray(live_chunks[0]["prediction"]))))
        roundtrip_errors, readbacks = [], []
        ds = LeRobotDataset(receipt["datasets"][split]["repo_id"], root=EXPORT / split,
            delta_timestamps={"action": [i/50 for i in range(50)]}, video_backend="pyav")
        offset = 0
        for record in receipt["datasets"][split]["episodes"]:
            if record["episode"] == episode:
                break
            offset += record["steps"]
        close = int(np.flatnonzero(raw["phase"] == "close")[0])
        for index in (0, 100, close-25, close, close+25):
            target = raw["action_applied"][index:index+50].astype(np.float32)
            obs = {"state": raw["observation_state"][index], "instruction": instruction,
                "images": {cam: raw["image_"+cam][index] for cam in ("scene_rgb", "wrist_rgb")}}
            batch = default_collate([observation(obs)])
            batch["action"] = torch.from_numpy(target).unsqueeze(0)
            processed = controller.pre(batch)
            recovered = controller.post(processed["action"]).detach().cpu().numpy()[0]
            roundtrip_errors.append(float(np.max(np.abs(recovered-target))))
            exported = controller.pre(default_collate([ds[offset+index]]))
            readbacks.append({"index": index,
                "state_max_abs": float((processed["observation.state"]-exported["observation.state"]).abs().max()),
                "action_max_abs": float((processed["action"]-exported["action"]).abs().max()),
                "tokens_equal": bool(torch.equal(processed["observation.language.tokens"], exported["observation.language.tokens"])),
                "image_mean_abs": {cam: float((processed[cam]-exported[cam]).abs().mean())
                    for cam in ("observation.images.camera1", "observation.images.camera2")}})
        del ds
        env = OpenArmInsertEnv(images=False, workspace=workspace)
        extension = install_physical_grasp_metrics(env)
        if extension["extension_sha256"] != original["implementation"]["learning/vla/physical_grasp.py"]:
            raise ValueError("Physical grasp metric changed")

        def playback(commands, name, exact_expert=False):
            obs, info = env.reset(seed=meta["seed"], options=meta["reset_options"])
            if float(np.max(np.abs(obs["state"]-raw["observation_state"][0]))) != 0:
                raise ValueError("Physical reset mismatch")
            gate = PickupGate(float(env.data.qpos[env.plug_qadr+2]))
            qpos, states, diagnostics, applied = [env.data.qpos.copy()], [obs["state"].copy()], [info], []
            outcome = "step_budget_exhausted"
            for action in commands:
                budget()
                obs, reward, term, trunc, info = env.step(action)
                applied.append(info["applied_action"].copy())
                qpos.append(env.data.qpos.copy()); states.append(obs["state"].copy()); diagnostics.append(info)
                passed = gate.update(plug_z_m=float(env.data.qpos[env.plug_qadr+2]), grasp=info["grasp_established"],
                    contacts=info["plug_finger_contacts"], drops=info["drops"], safe=info["outcome"] in ("running", "success"), dt=info["executed_dt_s"])
                if term or trunc or passed:
                    outcome = "pickup_success" if passed else info["outcome"]
                    break
            exact_error = float(np.max(np.abs(np.asarray(qpos)-raw["qpos"][:len(qpos)]))) if exact_expert else None
            if exact_expert and (not gate.passed or exact_error != 0):
                raise ValueError("Fresh headless expert replay must physically qualify the diagnostic")
            np.savez_compressed(folder / (name+"_trajectory.npz"), qpos=np.asarray(qpos), observation_state=np.asarray(states),
                action_requested=np.asarray(commands[:len(applied)]), action_applied=np.asarray(applied))
            write_json(folder / (name+"_diagnostics.json"), json.loads(json.dumps(diagnostics,
                default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())))
            return {"name": name, "steps": len(applied), "outcome": outcome, "physical_pickup": gate.passed,
                "autonomous_success": False, "actual_simulator_state_forced": False,
                "recorded_observation_guidance": not exact_expert, "first_grasp": next((i for i,x in enumerate(diagnostics) if x["grasp_established"]), None),
                "maximum_lift_mm": 1000*gate.maximum_lift_m, "maximum_carry_s": gate.maximum_s,
                "maximum_finger_contacts": max(x["plug_finger_contacts"] for x in diagnostics), "exact_expert_qpos_max_abs": exact_error}

        expert_result = playback(raw["action_applied"][:total], "expert", True)
        assisted_result = playback(teacher_commands, "recorded_observations")
        # Read-only forward kinematics on a separate MjData, never the executing env.
        kin = mujoco.MjData(env.model)
        fk = {}
        for label, trajectory in (("expert", raw["qpos"][:total+1]), ("live", live["qpos"][:total+1])):
            tcp, plug = [], []
            for q in trajectory:
                kin.qpos[:] = q
                mujoco.mj_forward(env.model, kin)
                tcp.append(kin.site_xpos[env.grasp_site].copy()); plug.append(kin.xpos[env.plug].copy())
            fk[label+"_tcp"] = np.asarray(tcp); fk[label+"_plug"] = np.asarray(plug)
        np.savez_compressed(folder / "kinematics.npz", **fk)
        env.close()
        row = {"episode": episode, "split": split, "source_data_sha256": meta["data_sha256"],
            "live_report_sha256": sha256(live_dir / "report.json"), "source_first_close_command": close,
            "first_chunk_teacher_vs_live_max_abs": first_chunk_error,
            "normalization_roundtrip_max_abs": max(roundtrip_errors), "raw_export_readback": readbacks,
            "teacher": command_errors(teacher_commands, raw["action_applied"][:total], raw["phase"][:total]),
            "live": command_errors(live_commands, raw["action_applied"][:total], raw["phase"][:total]),
            "fresh_expert_replay": expert_result, "assisted_playback": assisted_result,
            "tcp_error_at_200_mm": float(np.linalg.norm(fk["live_tcp"][200]-fk["expert_tcp"][200])*1000),
            "tcp_error_at_expert_close_mm": float(np.linalg.norm(fk["live_tcp"][close]-fk["expert_tcp"][close])*1000)}
        write_json(folder / "report.json", row)
        rows.append(row)
        write_json(args.output / "progress.json", {"rows": rows})
        print(json.dumps({"episode": episode, "result": row}, default=lambda x: x.item()), flush=True)
        del raw, live, predictions, teacher_commands
    write_json(args.output / "report.json", {"completed": True, "checkpoint_weights_sha256": original["base_weights_sha256"],
        "normalization": normalization, "rows": rows, "elapsed_s": time.monotonic()-started,
        "limits": "Two resets and one fixed noise stream; recorded-observation action playback is externally guided, not autonomous. No training or controller/scene modifications."})


if __name__ == "__main__":
    main()
