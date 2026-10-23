"""Bounded real pretrained SmolVLA training/save/reload test; no robot execution."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 20:
        raise ValueError("Smoke test is limited to 1-20 updates")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from torch.utils.data import default_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from learning.vla.smolvla_compat import masked_forward
    from learning.vla.smolvla_data import sha256

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this smoke test")
    torch.manual_seed(1000)
    started = time.time()
    receipt = json.loads((args.export / "openarm_export.json").read_text())
    ds = LeRobotDataset(receipt["datasets"]["train"]["repo_id"], root=args.export / "train",
                        delta_timestamps={"action": [i / 50 for i in range(50)]}, video_backend="pyav")
    config = PreTrainedConfig.from_pretrained(str(args.checkpoint))
    if not isinstance(config, SmolVLAConfig):
        raise ValueError("Expected SmolVLA checkpoint")
    config.input_features = {}
    config.pretrained_path = args.checkpoint
    config.vlm_model_name = str(args.backbone.resolve())
    # The full SmolVLA checkpoint supplies ALL weights; do not fetch a second backbone.
    config.load_vlm_weights = False
    config.device = "cuda"
    config.push_to_hub = False
    policy = make_policy(config, ds_meta=ds.meta)
    from learning.vla.training_precision import restore_checkpoint_trainable_dtypes
    restore_checkpoint_trainable_dtypes(policy, args.checkpoint)
    pre, post = make_pre_post_processors(policy.config, dataset_stats=ds.meta.stats)
    # Mid-episode sample and terminal sample exercise real video decoding and padding.
    index = receipt["datasets"]["train"]["episodes"][0]["steps"] // 2
    batch = pre(default_collate([ds[index]]))
    terminal = pre(default_collate([ds[receipt["datasets"]["train"]["episodes"][0]["steps"] - 1]]))
    from learning.vla.training_precision import promote_trainable_fp32, assert_fp32_optimizer
    promote_trainable_fp32(policy)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)
    assert_fp32_optimizer(policy, optimizer)
    probe = trainable[0].detach().clone()
    losses = []
    # Fix noise/time for a plumbing fit comparison; not a generalization experiment.
    noise = torch.randn((1, 50, 32), device="cuda")
    timestep = torch.full((1,), .5, device="cuda")
    def loss_for(b):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return masked_forward(policy, b, noise=noise, time=timestep)
    policy.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, info = loss_for(batch)
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite training loss")
        loss.backward()
        if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
            raise AssertionError("Frozen parameters received gradients")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 10, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        print(json.dumps({"step": step + 1, "loss": losses[-1], "grad_norm": float(norm)}), flush=True)
    changed = not torch.equal(probe, trainable[0].detach())
    if not changed:
        raise AssertionError("Trainable parameters did not update")
    policy.eval()
    with torch.no_grad():
        fitted_loss = float(loss_for(batch)[0])
        terminal_loss, terminal_info = loss_for(terminal)
        if "losses_after_in_ep_bound" not in terminal_info or not torch.isfinite(terminal_loss):
            raise AssertionError("Terminal mask was not consumed")
    bundle = args.output / "pretrained_model"
    policy.save_pretrained(bundle)
    pre.save_pretrained(bundle)
    post.save_pretrained(bundle)
    # Store optimizer state, but do not claim exact interrupted-job resume.
    torch.save(optimizer.state_dict(), args.output / "optimizer.pt")
    expected = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    del optimizer, trainable, probe, policy
    gc.collect()
    torch.cuda.empty_cache()
    restored = SmolVLAPolicy.from_pretrained(str(bundle))
    restore_checkpoint_trainable_dtypes(restored, bundle)
    for key, value in restored.state_dict().items():
        if not torch.equal(value.detach().cpu(), expected[key]):
            raise AssertionError(f"Reload mismatch: {key}")
    restored.eval()
    restored_pre, restored_post = make_pre_post_processors(restored.config, pretrained_path=str(bundle))
    restored_batch = restored_pre(default_collate([ds[index]]))
    for key in ("observation.state", "action"):
        if not torch.equal(restored_batch[key], batch[key]):
            raise AssertionError(f"Processor reload mismatch: {key}")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference = {key: value for key, value in restored_batch.items() if key not in ("action", "action_is_pad", "actions_id_pad")}
        actions = restored_post(restored.predict_action_chunk(inference))
    if tuple(actions.shape) != (1, 50, 8) or not torch.isfinite(actions).all():
        raise AssertionError("Invalid postprocessed actions")
    report = {"passed": True, "steps": args.steps, "losses": losses, "fixed_sample_loss_after": fitted_loss,
              "parameters_updated": changed, "reload_weights_exact": True,
              "processor_reload_exact": True,
              "trainable_parameters": trainable_count, "frozen_gradients_absent": True,
              "terminal_mask_consumed": True, "inference_shape": list(actions.shape),
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
              "seconds": time.time() - started, "source_manifest_sha256": receipt["source_manifest_sha256"],
              "base_weights_sha256": sha256(args.checkpoint / "model.safetensors"),
              "base_config_sha256": sha256(args.checkpoint / "config.json"),
              "limits": "Few updates on one training sample; no task success, validation loss or exact resume claim."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
