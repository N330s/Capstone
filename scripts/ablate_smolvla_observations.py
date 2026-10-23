"""Diagnostic state/image swaps at saved expert and failed-policy configurations."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", str(ROOT / "data/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data/hf_cache/datasets"))
from scripts.rollout_smolvla import verify_snapshot
from learning.vla.smolvla_data import observation, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import mujoco
    from torch.utils.data import default_collate
    from learning.vla.smolvla_policy import SmolVLAController
    snapshot = ROOT / "third_party/capstone_automateData"
    collection = ROOT / "data/drive_v2c30_20261012/merged_all_v1"
    bundle = ROOT / "outputs/v2c30_sft_overnight_v1/checkpoints/step_0009426_best/pretrained_model"
    controller = SmolVLAController(bundle, ROOT / "data/models/smolvlm_backbone")
    if "envs" in sys.modules:
        raise RuntimeError("Use a fresh environment process")
    sys.path.insert(0, str(snapshot))
    from envs.openarm_insert import OpenArmInsertEnv
    if not Path(sys.modules["envs.openarm_insert"].__file__).resolve().is_relative_to(snapshot):
        raise RuntimeError("Wrong environment loaded")
    rows, render_checks, dynamics = [], [], []
    names = ("expert_state_expert_images", "policy_state_expert_images",
             "expert_state_policy_images", "policy_state_policy_images",
             "policy_joints_only", "policy_gripper_only", "policy_with_expert_gripper")
    for episode in ("train_20004", "train_21010", "train_21036"):
        source = collection / episode
        meta = json.loads((source / "metadata.json").read_text())
        if sha256(source / "episode.npz") != meta["data_sha256"]:
            raise ValueError("Source checksum mismatch")
        workspace = verify_snapshot(snapshot, meta)
        raw = np.load(source / "episode.npz", allow_pickle=False)
        live = np.load(ROOT / "outputs/v2c30_best_sim_20261013" / ("best_" + episode) / "trajectory.npz")
        expert_qpos, expert_qvel, expert_state, expert_actions = [raw[key] for key in
                      ("qpos", "qvel", "observation_state", "action_applied")]
        live_qpos, live_qvel, live_state, requested, applied = [live[key] for key in
                      ("qpos", "qvel", "observation_state", "action_requested", "action_applied")]
        env = OpenArmInsertEnv(images=True, workspace=workspace)
        try:
            initial, _ = env.reset(seed=meta["seed"], options=meta["reset_options"])
            if float(np.max(np.abs(initial["state"] - expert_state[0]))) > 1e-6:
                raise ValueError("Initial state mismatch")
            def render(qpos, qvel, expected):
                # Diagnostic rerender only; no rollout controls use saved qpos playback.
                env.data.qpos[:] = qpos
                env.data.qvel[:] = qvel
                mujoco.mj_forward(env.model, env.data)
                obs = env.observe()
                if float(np.max(np.abs(obs["state"] - expected))) > 1e-6:
                    raise ValueError("Rerender state mismatch")
                return obs
            for frame in (0, 100, 200, 300, 320):
                e = render(expert_qpos[frame], expert_qvel[frame], expert_state[frame])
                p = render(live_qpos[frame], live_qvel[frame], live_state[frame])
                if frame == 0:
                    render_checks.append({"episode": episode, "expert_vs_policy_initial_pixels_max":
                        max(int(np.abs(e["images"][k].astype(int) - p["images"][k].astype(int)).max()) for k in e["images"]),
                        "recorded_vs_rerendered_mean_abs": {camera: float(np.abs(
                            raw["image_" + camera][0].astype(float) - e["images"][camera].astype(float)).mean() / 255)
                            for camera in e["images"]}})
                joint_state = e["state"].copy()
                joint_state[8:15] = p["state"][8:15]
                finger_state = e["state"].copy()
                finger_state[15] = p["state"][15]
                corrected_finger = p["state"].copy()
                corrected_finger[15] = e["state"][15]
                variants = [e, {**e, "state": p["state"]}, {**p, "state": e["state"]}, p,
                            {**e, "state": joint_state}, {**e, "state": finger_state},
                            {**p, "state": corrected_finger}]
                batch = controller.pre(default_collate([observation(v) for v in variants]))
                for seed in (1000, 1001, 1002):
                    controller.model.reset()
                    base_noise = torch.randn((1, 50, controller.model.config.max_action_dim), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(seed))
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = controller.post(controller.model.predict_action_chunk(batch, noise=base_noise.repeat(len(names), 1, 1)))
                    physical = prediction.float().cpu().numpy()
                    if physical.shape != (len(names), 50, 8) or not np.isfinite(physical).all():
                        raise ValueError("Invalid swapped-input predictions")
                    for index, name in enumerate(names):
                        rows.append({"episode": episode, "frame": frame, "seed": seed, "inputs": name,
                            "finger_mm": float(physical[index, 0, 7] * 1000),
                            "first_joint_mae_vs_expert_rad": float(np.abs(physical[index, 0, :7] - expert_actions[frame, :7]).mean()),
                            "expert_finger_mm": float(expert_actions[frame, 7] * 1000),
                            "live_applied_finger_mm": float(applied[frame, 7] * 1000)})
                print(json.dumps({"episode": episode, "frame": frame}), flush=True)
            # Quantify repeated rate limiting and whether joint motion leads/lags expert.
            dynamics.append({"episode": episode,
                "clipped_command_count": int(np.any(np.abs(requested - applied) > 1e-7, axis=1).sum()),
                "max_request_apply_joint_delta_rad": float(np.abs(requested[:, :7] - applied[:, :7]).max()),
                "max_request_apply_finger_delta_mm": float(np.abs(requested[:, 7] - applied[:, 7]).max() * 1000),
                "state_joint_error_rad_at_320": (live_state[320, 8:15] - expert_state[320, 8:15]).tolist(),
                "config_joint_rate_rad_s": env.config["joint_target_rate_rad_s"],
                "config_finger_rate_m_s": env.config["finger_target_rate_m_s"], "dt": env.dt})
        finally:
            env.close()
            raw.close(); live.close()
    effects = []
    for episode in ("train_20004", "train_21010", "train_21036"):
        for frame in (0, 100, 200, 300, 320):
            means = {name: float(np.mean([r["finger_mm"] for r in rows if r["episode"] == episode
                                          and r["frame"] == frame and r["inputs"] == name])) for name in names}
            ee, pe, ep, pp = [means[name] for name in names[:4]]
            effects.append({"episode": episode, "frame": frame, "means_mm": means,
                           "state_effect_mm": pe - ee, "image_effect_mm": ep - ee,
                           "interaction_mm": pp - pe - ep + ee})
    result = {"predictions": rows, "effects": effects, "rerender_checks": render_checks,
              "execution_diagnostics": dynamics, "checkpoint_sha256": sha256(bundle / "model.safetensors"),
              "limits": "Saved-state rerender and counterfactual input swaps are diagnostics, not autonomous rollouts. Three noise seeds per combination; hybrid state/image observations may be physically inconsistent."}
    (args.output / "report.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
