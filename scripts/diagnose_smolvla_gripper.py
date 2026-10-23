"""Compare recorded-observation predictions, normalization and failed rollouts."""
import argparse
import csv
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
from learning.vla.smolvla_data import observation, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import default_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from learning.vla.smolvla_policy import SmolVLAController
    collection = ROOT / "data/drive_v2c30_20261012/merged_all_v1"
    export = ROOT / "data/drive_v2c30_20261012/export_all_v1"
    bundle = ROOT / "outputs/v2c30_sft_overnight_v1/checkpoints/step_0009426_best/pretrained_model"
    stats = json.loads((export / "train/meta/stats.json").read_text())["action"]
    pre_stats = load_file(str(bundle / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post_stats = load_file(str(bundle / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    differences = {key: {"pre_vs_post_max_abs": float((pre_stats['action.' + key] - post_stats['action.' + key]).abs().max()),
                        "saved_vs_export_max_abs": float(np.max(np.abs(pre_stats['action.' + key].numpy() - stats[key])))}
                   for key in ("mean", "std")}
    controller = SmolVLAController(bundle, ROOT / "data/models/smolvlm_backbone")
    manifest = json.loads((collection / "manifest.json").read_text())
    receipt = json.loads((export / "openarm_export.json").read_text())
    exported = LeRobotDataset(receipt["datasets"]["validation"]["repo_id"], root=export / "validation",
                             delta_timestamps={"action": [i / 50 for i in range(50)]}, video_backend="pyav")
    offsets, cursor = {}, 0
    for record in receipt["datasets"]["validation"]["episodes"]:
        offsets[record["episode"]] = cursor
        cursor += record["steps"]
    predictions, comparisons, roundtrips = [], [], []
    for name in ("train_20004", "train_21010", "train_21036"):
        source = collection / name
        meta = json.loads((source / "metadata.json").read_text())
        if sha256(source / "episode.npz") != meta["data_sha256"]:
            raise ValueError("Source recording checksum mismatch")
        with np.load(source / "episode.npz", allow_pickle=False) as raw:
            states, actions, phases = raw["observation_state"], raw["action_applied"], raw["phase"]
            images = {camera: raw[camera] for camera in ("image_scene_rgb", "image_wrist_rgb")}
            first_close = int(np.flatnonzero(phases == "close")[0])
            first_lift = int(np.flatnonzero(phases == "lift")[0])
            frames = sorted({0, 100, 200, 300, 321, 400, 500, first_close, first_close + 25, first_lift})
            for index in frames:
                raw_obs = {"state": states[index], "instruction": manifest["instruction"],
                           "images": {"scene_rgb": images["image_scene_rgb"][index],
                                      "wrist_rgb": images["image_wrist_rgb"][index]}}
                physical = default_collate([observation(raw_obs)])
                indices = np.minimum(np.arange(index, index + 50), len(actions) - 1)
                physical["action"] = torch.from_numpy(actions[indices].astype(np.float32)).unsqueeze(0)
                batch = controller.pre(physical)
                recovered = controller.post(batch["action"]).detach().cpu().numpy()[0]
                roundtrips.append(float(np.max(np.abs(recovered - actions[indices]))))
                if index == 0:
                    entry = exported[offsets[name]]
                    exp_batch = controller.pre(default_collate([entry]))
                    comparisons.append({"episode": name,
                        "state_max_abs": float((batch["observation.state"] - exp_batch["observation.state"]).abs().max()),
                        "action_max_abs": float((batch["action"] - exp_batch["action"]).abs().max()),
                        "tokens_equal": bool(torch.equal(batch["observation.language.tokens"], exp_batch["observation.language.tokens"])),
                        "camera_raw_vs_export_mean_abs": {k: float((batch[k] - exp_batch[k]).abs().mean())
                            for k in ("observation.images.camera1", "observation.images.camera2")}})
                actor = {k: v for k, v in batch.items() if k not in ("action", "action_is_pad", "actions_id_pad")}
                for seed in (1000, 1001, 1002):
                    controller.model.reset()
                    noise = torch.randn((1, 50, controller.model.config.max_action_dim), device="cuda",
                                        generator=torch.Generator(device="cuda").manual_seed(seed))
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        normalized = controller.model.predict_action_chunk(actor, noise=noise)
                        predicted = controller.post(normalized).float().cpu().numpy()[0]
                    if predicted.shape != (50, 8) or not np.isfinite(predicted).all():
                        raise ValueError("Invalid predicted actions")
                    predictions.append({"episode": name, "step": index, "phase": str(phases[index]), "seed": seed,
                        "expert_finger_mm": float(actions[index, 7] * 1000),
                        "prediction_finger_mm": float(predicted[0, 7] * 1000),
                        "prediction_prefix_min_mm": float(predicted[:5, 7].min() * 1000),
                        "first_joint_mae_rad": float(np.abs(predicted[0, :7] - actions[index, :7]).mean()),
                        "prefix_gripper_mae_mm": float(np.abs(predicted[:5, 7] - actions[indices[:5], 7]).mean() * 1000),
                        "normalized_finger": float(normalized[0, 0, 7]),
                        "manual_decode_error_m": float(abs(predicted[0, 7] -
                            (float(normalized[0, 0, 7]) * post_stats["action.std"][7].item() + post_stats["action.mean"][7].item())))})
                print(json.dumps({"episode": name, "frame": index, "phase": str(phases[index])}), flush=True)
    fields = list(predictions[0])
    with (args.output / "recorded_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(predictions)
    early = [r for r in predictions if r["step"] <= 321]
    report = {"checkpoint_weights_sha256": sha256(bundle / "model.safetensors"),
              "saved_stats_differences": differences, "action_roundtrip_max_abs": max(roundtrips),
              "raw_vs_export_preprocessing": comparisons, "predictions": predictions,
              "early_prediction_below_train_min_count": sum(r["prediction_finger_mm"] < stats["min"][7] * 1000 for r in early),
              "early_prediction_count": len(early),
              "early_gripper_mae_mm": float(np.mean([abs(r["prediction_finger_mm"] - r["expert_finger_mm"]) for r in early])),
              "manual_decoder_max_abs_m": max(r["manual_decode_error_m"] for r in predictions),
              "limits": "30 recorded expert observations, 3 fixed noise seeds each. Diagnostic subset, not full validation or a closed-loop intervention."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("action_roundtrip_max_abs", "early_gripper_mae_mm",
                       "early_prediction_below_train_min_count", "early_prediction_count", "manual_decoder_max_abs_m")}), flush=True)


if __name__ == "__main__":
    main()
