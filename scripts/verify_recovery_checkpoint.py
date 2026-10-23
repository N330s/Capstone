"""Fresh-process offline reload check for a completed recovery training pilot."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[key] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HOME", str(ROOT / "data/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data/hf_cache/datasets"))
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("training", "export", "warmstart", "backbone", "report"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = json.loads((args.training / "report.json").read_text())
    if not report.get("completed"):
        raise ValueError("Completed training required")
    bundle = Path(report["final_checkpoint"]) / "pretrained_model"
    import torch
    from torch.utils.data import default_collate
    from safetensors.torch import load_file
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from learning.vla.smolvla_compat import masked_forward
    from learning.vla.smolvla_data import decoded_actions, sha256
    from learning.vla.recovery_data import RECOVERY_CONTRACT

    receipt_path = args.export / "openarm_export.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["contract"] != RECOVERY_CONTRACT or sha256(receipt_path) != report["binding"]["export_receipt_sha256"]:
        raise ValueError("Recovery export binding mismatch")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    config = PreTrainedConfig.from_pretrained(str(bundle))
    config.vlm_model_name, config.load_vlm_weights, config.device = str(args.backbone.resolve()), False, "cuda"
    policy = SmolVLAPolicy.from_pretrained(str(bundle), config=config)
    from learning.vla.training_precision import restore_checkpoint_trainable_dtypes
    restore_checkpoint_trainable_dtypes(policy, bundle)
    pre, post = make_pre_post_processors(config, pretrained_path=str(bundle),
        preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": str(args.backbone.resolve())},
                                "device_processor": {"device": "cuda"}})
    stats_equal = {}
    for name in ("policy_preprocessor_step_5_normalizer_processor.safetensors",
                 "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
        original, saved = load_file(str(args.warmstart / name)), load_file(str(bundle / name))
        equal = set(original) == set(saved) and all(torch.equal(original[k], saved[k]) for k in original)
        if not equal:
            raise ValueError("Warm-start normalization changed")
        stats_equal[name] = equal
    ds = LeRobotDataset(receipt["datasets"]["validation"]["repo_id"], root=args.export / "validation",
                        delta_timestamps={"action": [i / 50 for i in range(50)]}, video_backend="pyav")
    probe = report["binding"]["probes"]["validation"][0]
    batch = pre(default_collate([ds[probe["index"]]]))
    generator = torch.Generator(device="cuda").manual_seed(report["binding"]["seed"] + 2000)
    noise = torch.randn((1, 50, config.max_action_dim), device="cuda", generator=generator)
    policy.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = masked_forward(policy, batch, noise=noise, time=torch.full((1,), .5, device="cuda"))
        actor = {k: v for k, v in batch.items() if k not in ("action", "action_is_pad", "actions_id_pad")}
        policy.reset()
        actions = post(policy.predict_action_chunk(actor, noise=noise))
    decoded_actions(actions[0].float().cpu().numpy())
    expected = report["final_evaluation"]["scores"]["validation"]["samples"][0]
    difference = abs(float(loss) - expected)
    if not torch.isfinite(loss) or difference > 1e-8:
        raise ValueError(f"Reloaded flow loss differs: {float(loss)} vs {expected}")
    result = {"passed": True, "checkpoint": str(bundle), "probe": probe,
              "expected_loss": expected, "reloaded_loss": float(loss), "loss_difference": difference,
              "normalization_tensors_unchanged": stats_equal, "decoded_action_shape": list(actions.shape),
              "weights_sha256": sha256(bundle / "model.safetensors"), "robot_execution": False,
              "limits": "One matched reload probe, not exact optimizer resume or autonomous task success"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
