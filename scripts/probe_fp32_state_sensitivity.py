"""Same RGB and identical latent noise: intervene on numeric state channels only.

Counterfactual offline sensitivity, not a controller change or physical success.
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
from learning.vla.smolvla_data import sha256, observation
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
    from torch.utils.data import default_collate
    from learning.vla.smolvla_policy import SmolVLAController
    campaign = ROOT / "outputs/fp32_multiepisode_20261020"
    collection = ROOT / "data/drive_v2c30_20261012/merged_all_v1"
    trained = json.loads((campaign / "training/report.json").read_text())
    bundle = Path(trained["final_checkpoint"]) / "pretrained_model"
    frozen = json.loads((campaign / "pickup_evaluation/plan.json").read_text())
    if sha256(bundle / "model.safetensors") != frozen["base_weights_sha256"]:
        raise ValueError("Checkpoint changed")
    archive = archive_source("scripts/probe_fp32_state_sensitivity.py")
    instruction = json.loads((collection / "manifest.json").read_text())["instruction"]
    controller = SmolVLAController(bundle, ROOT / "data/models/smolvlm_backbone", 1000)
    rows = []
    for name in ("train_20002", "train_20004"):
        meta = json.loads((collection / name / "metadata.json").read_text())
        if sha256(collection / name / "episode.npz") != meta["data_sha256"]:
            raise ValueError("Source changed")
        with np.load(collection / name / "episode.npz", allow_pickle=False) as raw:
            data = {k: raw[k].copy() for k in ("observation_state", "action_applied", "phase", "image_scene_rgb", "image_wrist_rgb")}
        with np.load(campaign / "pickup_evaluation" / (name+"_policy") / "trajectory.npz", allow_pickle=False) as raw:
            live = raw["observation_state"].copy()
        close = int(np.flatnonzero(data["phase"] == "close")[0])
        anchors = sorted(set([0, 25, 100, 200, 250, 300, 350, 400, 450, close-25, close]))
        for index in anchors:
            for seed in (1000, 1001):
                if time.monotonic()-started > 280:
                    raise TimeoutError("Five-minute state-sensitivity allowance exhausted")
                noise = torch.randn((1, 50, controller.model.config.max_action_dim), device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(seed+index))
                prediction = {}
                for variant in ("recorded", "only_live_right_finger", "all_live_state"):
                    state = data["observation_state"][index].copy()
                    if variant == "only_live_right_finger":
                        state[15] = live[index, 15]
                    elif variant == "all_live_state":
                        state[:] = live[index]
                    actor = {"state": state, "instruction": instruction,
                        "images": {cam: data["image_"+cam][index] for cam in ("scene_rgb", "wrist_rgb")}}
                    batch = controller.pre(default_collate([observation(actor)]))
                    controller.model.reset()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        result = controller.post(controller.model.predict_action_chunk(batch, noise=noise.clone()))
                    chunk = result[0].float().cpu().numpy()
                    if chunk.shape != (50, 8) or not np.isfinite(chunk).all():
                        raise ValueError("Invalid sensitivity prediction")
                    prediction[variant] = {"first_finger_mm": float(chunk[0, 7])*1000,
                        "prefix_finger_mean_mm": float(chunk[:25, 7].mean())*1000,
                        "prefix_finger_mae_mm": float(np.abs(chunk[:25, 7]-data["action_applied"][index:index+25, 7]).mean())*1000,
                        "first_joint_mae_rad": float(np.abs(chunk[0, :7]-data["action_applied"][index, :7]).mean())}
                row = {"episode": name, "index": index, "seed": seed, "phase": str(data["phase"][index]),
                    "expert_finger_target_mm": float(data["action_applied"][index, 7])*1000,
                    "recorded_finger_state_mm": float(data["observation_state"][index, 15])*1000,
                    "live_finger_state_mm": float(live[index, 15])*1000, "variants": prediction}
                rows.append(row)
                with (args.output / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row)+"\n")
            print(json.dumps({"episode": name, "anchor": index}), flush=True)
        del data, live
    write_json(args.output / "report.json", {"completed": True, "rows": rows, "elapsed_s": time.monotonic()-started,
        "checkpoint_sha256": frozen["base_weights_sha256"], "implementation": archive,
        "controls": "Identical recorded RGB, instruction and cloned explicit latent noise within every three-way comparison; only numeric state differs.",
        "limits": "Counterfactual state/RGB combinations; sensitivity evidence, not autonomous or physical task-success evidence. No optimization."})


if __name__ == "__main__":
    main()
