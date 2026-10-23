"""Same explicit latent noise: repeat identical observations and tiny render changes."""
import argparse
import hashlib
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
from learning.vla.smolvla_data import observation, sha256
from learning.vla.runtime_io import write_json
from learning.vla.source_archive import archive_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import default_collate
    from learning.vla.smolvla_policy import SmolVLAController
    onset = ROOT / "outputs/fp32_pickup_onset_20261020"
    plan = json.loads((onset / "plan.json").read_text())
    bundle = Path(plan["checkpoint"])
    if sha256(bundle / "model.safetensors") != plan["weights_sha256"]:
        raise ValueError("Checkpoint changed")
    controller = SmolVLAController(bundle, ROOT / "data/models/smolvlm_backbone", 1000)
    outputs = {}
    rows = []
    for name in ("train_20002", "train_20004"):
        source = ROOT / "data/drive_v2c30_20261012/merged_all_v1" / name
        meta = json.loads((source / "metadata.json").read_text())
        with np.load(source / "episode.npz", allow_pickle=False) as raw:
            state = raw["observation_state"][0].copy()
        for precision in ("bf16", "fp16"):
            for seed in (1000, 1001):
                noise = torch.randn((1, 50, controller.model.config.max_action_dim), device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(seed))
                saved, hashes = {}, {}
                for label, image_source in (("recorded_a", "recorded"), ("rendered_a", "rendered"),
                        ("recorded_b", "recorded"), ("rendered_b", "rendered")):
                    if time.monotonic()-started > 280:
                        raise TimeoutError("Five-minute repeatability bound exhausted")
                    obs = {"state": state.copy(), "instruction": meta["instruction"], "images": {
                        camera: np.array(Image.open(onset / name / (camera+"_"+image_source+"_reset.png")))
                        for camera in ("scene_rgb", "wrist_rgb")}}
                    batch = controller.pre(default_collate([observation(obs)]))
                    hashes[label] = {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                        for k,v in batch.items() if isinstance(v, torch.Tensor)}
                    controller.model.reset()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16):
                        out = controller.post(controller.model.predict_action_chunk(batch, noise=noise.clone()))
                    predicted = out[0].float().cpu().numpy()
                    if predicted.shape != (50, 8) or not np.isfinite(predicted).all():
                        raise ValueError("Nonfinite repeatability probe")
                    saved[label] = predicted
                    outputs[f"{name}_{precision}_{seed}_{label}"] = predicted
                row = {"episode": name, "precision": precision, "seed": seed,
                    "identical_recorded_input_hashes": hashes["recorded_a"] == hashes["recorded_b"],
                    "identical_rendered_input_hashes": hashes["rendered_a"] == hashes["rendered_b"],
                    "recorded_repeat_max_abs": float(np.abs(saved["recorded_a"]-saved["recorded_b"]).max()),
                    "rendered_repeat_max_abs": float(np.abs(saved["rendered_a"]-saved["rendered_b"]).max()),
                    "recorded_vs_rendered_joint_max_abs_rad": float(np.abs(saved["recorded_a"][:,:7]-saved["rendered_a"][:,:7]).max()),
                    "recorded_vs_rendered_gripper_max_abs_mm": float(np.abs(saved["recorded_a"][:,7]-saved["rendered_a"][:,7]).max()*1000),
                    "first25_gripper_mae_mm": float(np.abs(saved["recorded_a"][:25,7]-.044).mean()*1000)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    np.savez_compressed(args.output / "predictions.npz", **outputs)
    write_json(args.output / "report.json", {"completed": True, "rows": rows,
        "weights_sha256": plan["weights_sha256"], "implementation": archive_source("scripts/probe_pickup_repeatability.py"),
        "elapsed_s": time.monotonic()-started,
        "limits": "Two resets, two explicit latent seeds; FP16 is an offline precision intervention, not an approved policy/controller change or success evaluation."})


if __name__ == "__main__":
    main()
