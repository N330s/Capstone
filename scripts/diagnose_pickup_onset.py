"""Bounded onset audit: reset images, ambiguous targets, native loss and early prefix tests.

Diagnostic interventions only. No training, production-controller changes,
guard relaxation, task-success selection or physical robot actions.
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
from learning.vla.smolvla_data import observation, sha256
from learning.vla.source_archive import archive_source
from learning.vla.pickup_onset import filtered_targets, loss_components


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.output.parent != ROOT / "outputs":
        raise ValueError("New direct outputs directory required")
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()

    def budget():
        if time.monotonic()-started > 850:
            raise TimeoutError("15-minute onset diagnostic allowance exhausted")

    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from PIL import Image
    from learning.vla.smolvla_policy import SmolVLAController
    from learning.vla.smolvla_compat import masked_forward
    from learning.vla.physical_grasp import install_physical_grasp_metrics
    from scripts.rollout_smolvla import verify_snapshot
    campaign = ROOT / "outputs/fp32_multiepisode_20261020"
    collection = ROOT / "data/drive_v2c30_20261012/merged_all_v1"
    snapshot = ROOT / "third_party/capstone_automateData"
    trained = json.loads((campaign / "training/report.json").read_text())
    bundle = Path(trained["final_checkpoint"]) / "pretrained_model"
    frozen = json.loads((campaign / "pickup_evaluation/plan.json").read_text())
    if sha256(bundle / "model.safetensors") != frozen["base_weights_sha256"]:
        raise ValueError("Checkpoint changed")
    instruction = json.loads((collection / "manifest.json").read_text())["instruction"]
    implementation = {f: archive_source(f) for f in ("scripts/diagnose_pickup_onset.py", "learning/vla/pickup_onset.py",
        "learning/vla/smolvla_policy.py", "learning/vla/smolvla_compat.py")}
    write_json(args.output / "plan.json", {"checkpoint": str(bundle), "weights_sha256": frozen["base_weights_sha256"],
        "implementation": implementation, "max_seconds": 900, "early_prefixes": [1, 5, 25], "early_commands": 200,
        "prefix_test_reset": "train_20002", "training": False,
        "limits": "Diagnostic onsets, not a prefix selection sweep or a pickup/generalization result."})
    controller = SmolVLAController(bundle, ROOT / "data/models/smolvlm_backbone", 1000)
    if controller.model.config.n_obs_steps != 1:
        raise ValueError("Single-observation onset protocol required")
    sys.path.insert(0, str(snapshot))
    from envs.openarm_insert import OpenArmInsertEnv
    import mujoco

    def predict(obs, noise):
        actor = controller.pre(default_collate([observation(obs)]))
        controller.model.reset()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = controller.post(controller.model.predict_action_chunk(actor, noise=noise.clone()))
        return prediction[0].float().cpu().numpy()

    rows = []
    for name in ("train_20002", "train_20004"):
        budget()
        folder = args.output / name
        folder.mkdir()
        meta = json.loads((collection / name / "metadata.json").read_text())
        if sha256(collection / name / "episode.npz") != meta["data_sha256"]:
            raise ValueError("Source recording changed")
        with np.load(collection / name / "episode.npz", allow_pickle=False) as raw:
            data = {k: raw[k].copy() for k in ("observation_state", "action_applied", "action_requested", "phase", "qpos",
                "image_scene_rgb", "image_wrist_rgb", "executed_dt_s")}
        env = OpenArmInsertEnv(images=True, workspace=verify_snapshot(snapshot, meta))
        extension = install_physical_grasp_metrics(env)
        if extension["extension_sha256"] != frozen["implementation"]["learning/vla/physical_grasp.py"]:
            raise ValueError("Physical contact metric changed")
        reset, info = env.reset(seed=meta["seed"], options=meta["reset_options"])
        initial_target = env.target.copy()
        reset_comparison = {"state_max_abs": float(np.abs(reset["state"]-data["observation_state"][0]).max()),
            "instruction_equal": reset["instruction"] == instruction,
            "images": {}}
        for camera in ("scene_rgb", "wrist_rgb"):
            source, actual = data["image_"+camera][0], reset["images"][camera]
            delta = np.abs(source.astype(float)-actual.astype(float))
            reset_comparison["images"][camera] = {"equal": bool(np.array_equal(source, actual)),
                "mean_abs_8bit": float(delta.mean()), "max_abs_8bit": float(delta.max())}
            Image.fromarray(source).save(folder / (camera+"_recorded_reset.png"))
            Image.fromarray(actual).save(folder / (camera+"_rendered_reset.png"))
        live = np.load(campaign / "pickup_evaluation" / (name+"_policy") / "trajectory.npz", allow_pickle=False)
        filter_errors = {}
        for label, requested, applied in (("source", data["action_requested"], data["action_applied"]),
                ("historical_live", live["action_requested"], live["action_applied"])):
            computed = filtered_targets(requested, initial_target, env.lower, env.upper, env.dt,
                env.config["joint_target_rate_rad_s"], env.config["finger_target_rate_m_s"])
            filter_errors[label] = float(np.abs(computed-applied).max())
        survey = int(np.flatnonzero(data["phase"] != "survey")[0])
        idle = {"commands": survey, "state_max_change": float(np.abs(data["observation_state"][:survey]-data["observation_state"][0]).max()),
            "image_0_vs25_mae_8bit": {camera: float(np.abs(data["image_"+camera][0].astype(float)-data["image_"+camera][25]).mean())
                for camera in ("scene_rgb", "wrist_rgb")},
            "same_survey_observation_future_finger_at_offset24_mm": [float(data["action_applied"][i+24, 7])*1000 for i in (0, 25, 43)]}
        row = {"episode": name, "reset_comparison": reset_comparison, "target_filter_max_abs": filter_errors,
               "survey_aliasing": idle, "profiles": [], "loss_audit": []}
        noise = torch.randn((1, 50, controller.model.config.max_action_dim), device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(1000))
        source_obs = {"state": data["observation_state"][0], "instruction": instruction,
            "images": {cam: data["image_"+cam][0] for cam in ("scene_rgb", "wrist_rgb")}}
        reset_predictions = {"recorded": predict(source_obs, noise), "actual_render": predict(reset, noise)}
        row["matched_noise_reset_prediction_max_abs"] = float(np.abs(reset_predictions["recorded"]-reset_predictions["actual_render"]).max())
        for index in (0, 10, 25, 43, 50, 75, 100):
            budget()
            obs = {"state": data["observation_state"][index], "instruction": instruction,
                "images": {cam: data["image_"+cam][index] for cam in ("scene_rgb", "wrist_rgb")}}
            predicted = predict(obs, noise)
            target = data["action_applied"][index:index+50]
            row["profiles"].append({"anchor": index, "finger_prediction_mm": (predicted[:, 7]*1000).tolist(),
                "finger_target_mm": (target[:, 7]*1000).tolist(),
                "prefix_errors": {str(k): {"gripper_mae_mm": float(np.abs(predicted[:k, 7]-target[:k, 7]).mean()*1000),
                    "joint_mae_rad": float(np.abs(predicted[:k, :7]-target[:k, :7]).mean())} for k in (1, 5, 25, 50)}})
        for index in (0, 200, int(np.flatnonzero(data["phase"] == "close")[0])-25):
            obs = {"state": data["observation_state"][index], "instruction": instruction,
                "images": {cam: data["image_"+cam][index] for cam in ("scene_rgb", "wrist_rgb")}}
            physical = default_collate([observation(obs)])
            physical["action"] = torch.from_numpy(data["action_applied"][index:index+50].astype(np.float32)).unsqueeze(0)
            physical["action_is_pad"] = torch.zeros((1, 50), dtype=torch.bool)
            batch = controller.pre(physical)
            original_forward = controller.model.model.forward
            captured = []
            def capture(*a, **kw):
                losses = original_forward(*a, **kw)
                captured.append(losses.detach().float().cpu().numpy())
                return losses
            controller.model.model.forward = capture
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    native, _ = masked_forward(controller.model, batch, noise=noise.clone(), time=torch.full((1,), .5, device="cuda"))
            finally:
                controller.model.model.forward = original_forward
            audit = loss_components(captured[0], np.zeros((1, 50), bool))
            audit.update(anchor=index, native_returned=float(native), native_reconstruction_error=abs(float(native)-audit["native_mean"]))
            row["loss_audit"].append(audit)
        if name == "train_20002":
            early = []
            for prefix in (1, 5, 25):
                budget()
                obs, info = env.reset(seed=meta["seed"], options=meta["reset_options"])
                qpos, states, requested, applied, tcp, chunks = [env.data.qpos.copy()], [obs["state"].copy()], [], [], [env.grasp_pose()[0]], []
                outcome = "diagnostic_200_commands"
                while len(requested) < 200:
                    budget()
                    anchor = len(requested)
                    common_noise = torch.randn((1, 50, controller.model.config.max_action_dim), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(1000+anchor))
                    predicted = predict(obs, common_noise)
                    chunks.append({"anchor": anchor, "prediction": predicted.tolist()})
                    for action in predicted[:min(prefix, 200-anchor)]:
                        obs, reward, term, trunc, info = env.step(action)
                        requested.append(action.copy()); applied.append(info["applied_action"].copy())
                        states.append(obs["state"].copy()); qpos.append(env.data.qpos.copy()); tcp.append(env.grasp_pose()[0])
                        if term or trunc:
                            outcome = info["outcome"]
                            break
                    if term or trunc:
                        break
                np.savez_compressed(folder / f"prefix_{prefix}_early.npz", qpos=np.asarray(qpos), state=np.asarray(states),
                    action_requested=np.asarray(requested), action_applied=np.asarray(applied), tcp=np.asarray(tcp))
                write_json(folder / f"prefix_{prefix}_chunks.json", chunks)
                entry = {"prefix": prefix, "steps": len(requested), "outcome": outcome,
                    "finger_min_mm": float(np.min(np.asarray(states)[:, 15])*1000), "finger_final_mm": float(states[-1][15])*1000,
                    "safety_abort": outcome != "diagnostic_200_commands", "not_pickup_evaluation": True}
                early.append(entry)
                print(json.dumps({"early_test": entry}), flush=True)
            row["early_prefix_tests"] = early
        env.close(); live.close()
        rows.append(row)
        write_json(args.output / (name+"_report.json"), row)
        print(json.dumps({"finished": name, "filter": filter_errors, "reset": reset_comparison,
            "loss": row["loss_audit"]}), flush=True)
        del data
    write_json(args.output / "report.json", {"completed": True, "rows": rows, "elapsed_s": time.monotonic()-started,
        "weights_sha256": frozen["base_weights_sha256"],
        "limits": "Prefix interventions use time-indexed common noise, one training reset and 200 commands only. No completed pickup/generalization comparison or production change. Loss hooks only inspect a no-grad model in this process."})


if __name__ == "__main__":
    main()
