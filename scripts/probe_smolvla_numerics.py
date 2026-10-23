"""Compare sampling precision/steps on identical observations and Gaussian noise.

Recorded-gripper perturbations are offline sensor counterfactuals, not rollouts.
No training, robot control, or checkpoint writes occur.
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from learning.vla.smolvla_policy import SmolVLAController
    from learning.vla.smolvla_data import observation, sha256
    checkpoint = ROOT / "outputs/v2c30_sft_overnight_v1/checkpoints/step_0009426_best/pretrained_model"
    episode = ROOT / "data/drive_v2c30_20261012/merged_all_v1/train_20002"
    meta = json.loads((episode / "metadata.json").read_text())
    if sha256(episode / "episode.npz") != meta["data_sha256"]:
        raise ValueError("Source changed")
    actor = SmolVLAController(checkpoint, ROOT / "data/models/smolvlm_backbone")
    with np.load(episode / "episode.npz", allow_pickle=False) as raw:
        states, targets = raw["observation_state"].copy(), raw["action_applied"].copy()
        observations, cases = [], []
        for frame in (100, 300, 520, 650):
            for offset in ((0., -.002) if frame in (100, 300) else (0.,)):
                state = states[frame].copy()
                state[15] += offset
                observations.append(observation({"state": state, "instruction": meta["instruction"],
                    "images": {camera: raw["image_" + camera][frame] for camera in ("scene_rgb", "wrist_rgb")}}))
                cases.append({"frame": frame, "sensor_offset_m": offset, "input_gripper_mm": float(state[15] * 1000)})
    rows, begun = [], time.monotonic()
    for precision, steps in (("bf16", 10), ("bf16", 50), ("fp32", 10), ("fp32", 50)):
        actor.model.config.num_steps = steps
        for seed in (1000, 1001, 1002):
            noise = torch.randn((1, 50, actor.model.config.max_action_dim), device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(seed))
            for start in range(0, len(cases), 2):
                batch = actor.pre(default_collate(observations[start:start + 2]))
                actor.model.reset()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                    pred = actor.post(actor.model.predict_action_chunk(batch, noise=noise.repeat(min(2, len(cases) - start), 1, 1)))
                values = pred.float().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite inference")
                for n, case in enumerate(cases[start:start + 2]):
                    goal = targets[case["frame"]:case["frame"] + 5]
                    rows.append({**case, "precision": precision, "steps": steps, "seed": seed,
                        "prediction_gripper_mm": float(values[n, :5, 7].mean() * 1000),
                        "reference_gripper_mm": float(goal[:, 7].mean() * 1000),
                        "prefix5_gripper_mae_mm": float(np.abs(values[n, :5, 7] - goal[:, 7]).mean() * 1000),
                        "prefix5_joint_mae_rad": float(np.abs(values[n, :5, :7] - goal[:, :7]).mean())})
        print(json.dumps({"finished": [precision, steps], "elapsed_s": time.monotonic() - begun}), flush=True)
    summary = []
    for precision, steps in (("bf16", 10), ("bf16", 50), ("fp32", 10), ("fp32", 50)):
        nominal = [r for r in rows if r["precision"] == precision and r["steps"] == steps and r["sensor_offset_m"] == 0]
        gains = []
        for frame in (100, 300):
            for seed in (1000, 1001, 1002):
                pair = [r for r in rows if r["precision"] == precision and r["steps"] == steps and r["frame"] == frame and r["seed"] == seed]
                gains.append((pair[1]["prediction_gripper_mm"] - pair[0]["prediction_gripper_mm"]) / -2.)
        summary.append({"precision": precision, "steps": steps,
            "nominal_gripper_mae_mm": float(np.mean([r["prefix5_gripper_mae_mm"] for r in nominal])),
            "nominal_joint_mae_rad": float(np.mean([r["prefix5_joint_mae_rad"] for r in nominal])),
            "gripper_state_gain": float(np.mean(gains))})
    result = {"source_sha256": meta["data_sha256"], "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
        "rows": rows, "summary": summary, "elapsed_s": time.monotonic() - begun,
        "limits": "One training episode, four anchors and three fixed Gaussian noise seeds. Offline counterfactual input tests, not autonomous success."}
    (args.output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
