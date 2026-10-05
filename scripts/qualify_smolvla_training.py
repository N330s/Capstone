"""Train across episodes, evaluate a held-out episode, and verify exact resume."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sample_plan(records):
    """Four spaced command windows per episode; terminals checked separately."""
    result, offset = [], 0
    for record in records:
        for fraction in (.1, .35, .6, .85):
            step = min(int(record["steps"] * fraction), record["steps"] - 50)
            if step < 0:
                raise ValueError("Qualification requires episodes with at least 50 commands")
            result.append({"index": offset + step, "episode": record["episode"], "step": step})
        offset += record["steps"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()
    if not 12 <= args.steps <= 100:
        raise ValueError("Qualification allows 12-100 updates")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from torch.utils.data import default_collate
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from learning.vla.smolvla_compat import masked_forward
    from learning.vla.smolvla_data import LEROBOT_VERSION, sha256
    from learning.vla.training_state import capture_rng, restore_rng, assert_state_equal, cpu_state
    from importlib.metadata import version

    if version("lerobot") != LEROBOT_VERSION or not torch.cuda.is_available():
        raise RuntimeError("Pinned LeRobot and CUDA are required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1000)
    random.seed(1000)
    np_seed = 1000
    import numpy as np
    np.random.seed(np_seed)
    started = time.time()
    receipt = json.loads((args.export / "openarm_export.json").read_text())
    plans = {split: sample_plan(receipt["datasets"][split]["episodes"])
             for split in ("train", "validation")}
    if set(r["episode"] for r in plans["train"]) & set(r["episode"] for r in plans["validation"]):
        raise ValueError("Episode split leakage")
    order = list(range(len(plans["train"])))
    random.Random(1000).shuffle(order)
    datasets = {split: LeRobotDataset(receipt["datasets"][split]["repo_id"], root=args.export / split,
                                     delta_timestamps={"action": [i / 50 for i in range(50)]}, video_backend="pyav")
                for split in plans}
    config = PreTrainedConfig.from_pretrained(str(args.checkpoint))
    config.input_features = {}
    config.pretrained_path = args.checkpoint
    config.vlm_model_name = str(args.backbone.resolve())
    config.load_vlm_weights = False
    config.device, config.push_to_hub = "cuda", False
    policy = make_policy(config, ds_meta=datasets["train"].meta)
    pre, post = make_pre_post_processors(policy.config, dataset_stats=datasets["train"].meta.stats)

    def optimizer_for(model):
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=.95)
        return optimizer, scheduler

    def evaluate(model, processor):
        model.eval()
        scores = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for split, records in plans.items():
                values = []
                for n, record in enumerate(records):
                    batch = processor(default_collate([datasets[split][record["index"]]]))
                    # Dedicated generator makes comparisons repeatable without advancing training RNG.
                    generator = torch.Generator(device="cuda").manual_seed(2000 + n)
                    noise = torch.randn((1, 50, 32), device="cuda", generator=generator)
                    loss, _ = masked_forward(model, batch, noise=noise, time=torch.full((1,), .5, device="cuda"))
                    if not torch.isfinite(loss):
                        raise AssertionError("Nonfinite evaluation loss")
                    values.append(float(loss))
                scores[split] = {"mean": sum(values) / len(values), "samples": values}
        return scores

    def update(model, processor, optimizer, scheduler, step, sample_order):
        record = plans["train"][sample_order[step % len(sample_order)]]
        batch = processor(default_collate([datasets["train"][record["index"]]]))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = masked_forward(model, batch)  # stochastic flow matching
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite training loss")
        loss.backward()
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise AssertionError("Frozen model received gradients")
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           10, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        event = {"step": step + 1, "episode": record["episode"], "frame": record["step"],
                 "loss": float(loss.detach()), "grad_norm": float(norm), "lr": scheduler.get_last_lr()[0]}
        with (args.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
        return event

    before = evaluate(policy, pre)
    print(json.dumps({"evaluation_before": before}), flush=True)
    optimizer, scheduler = optimizer_for(policy)
    split_step = args.steps // 2
    events = [update(policy, pre, optimizer, scheduler, step, order) for step in range(split_step)]
    resume_dir = args.output / "resume_checkpoint"
    bundle = resume_dir / "pretrained_model"
    policy.save_pretrained(bundle)
    pre.save_pretrained(bundle)
    post.save_pretrained(bundle)
    state = {"step": split_step, "sample_order": order, "plans": plans,
             "export_receipt_sha256": sha256(args.export / "openarm_export.json"),
             "optimizer": cpu_state(optimizer.state_dict()), "scheduler": scheduler.state_dict(),
             "rng": capture_rng()}
    torch.save(state, resume_dir / "training_state.pt")
    del state
    expected_events = [update(policy, pre, optimizer, scheduler, step, order)
                       for step in range(split_step, split_step + 3)]
    expected_weights = {name: p.detach().cpu().clone() for name, p in policy.named_parameters() if p.requires_grad}
    expected_optimizer = cpu_state(optimizer.state_dict())
    expected_scheduler, expected_rng = scheduler.state_dict(), capture_rng()
    del policy, optimizer, scheduler, pre, post
    gc.collect()
    torch.cuda.empty_cache()
    policy = SmolVLAPolicy.from_pretrained(str(bundle))
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(bundle))
    optimizer, scheduler = optimizer_for(policy)
    # This is our own local checkpoint; it includes Python/NumPy RNG objects.
    state = torch.load(resume_dir / "training_state.pt", map_location="cpu", weights_only=False)
    if state["plans"] != plans or state["export_receipt_sha256"] != sha256(args.export / "openarm_export.json"):
        raise ValueError("Resume input/config mismatch")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    order = state["sample_order"]
    cursor = state["step"]
    del state
    resumed_events = [update(policy, pre, optimizer, scheduler, step, order)
                      for step in range(cursor, cursor + 3)]
    assert_state_equal(resumed_events, expected_events, "resume.events")
    actual_weights = {name: p for name, p in policy.named_parameters() if p.requires_grad}
    assert_state_equal(actual_weights, expected_weights, "resume.weights")
    assert_state_equal(optimizer.state_dict(), expected_optimizer, "resume.optimizer")
    assert_state_equal(scheduler.state_dict(), expected_scheduler, "resume.scheduler")
    assert_state_equal(capture_rng(), expected_rng, "resume.rng")
    print(json.dumps({"resume_three_updates_exact": True}), flush=True)
    del expected_weights, expected_optimizer, expected_rng
    events.extend(resumed_events)
    events.extend(update(policy, pre, optimizer, scheduler, step, order)
                  for step in range(cursor + 3, args.steps))
    after = evaluate(policy, pre)
    final_dir = args.output / "final_checkpoint"
    policy.save_pretrained(final_dir / "pretrained_model")
    pre.save_pretrained(final_dir / "pretrained_model")
    post.save_pretrained(final_dir / "pretrained_model")
    torch.save({"step": args.steps, "sample_order": order, "plans": plans,
                "export_receipt_sha256": sha256(args.export / "openarm_export.json"),
                "optimizer": cpu_state(optimizer.state_dict()), "scheduler": scheduler.state_dict(),
                "rng": capture_rng()}, final_dir / "training_state.pt")
    report = {"passed": True, "unique_training_updates": args.steps,
              "resume_comparison_extra_updates": 3, "sample_plan": plans,
              "evaluation_before": before, "evaluation_after": after,
              "exact_resume": {"compared_updates": 3, "loss_grad_lr_data": True, "trainable_weights": True,
                               "optimizer": True, "scheduler": True, "rng": True},
              "training_events": events, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9, "seconds": time.time() - started,
              "export_receipt_sha256": sha256(args.export / "openarm_export.json"),
              "limits": "12 training windows and 4 validation windows; one held-out episode. Offline flow loss, no task-success claim. Resume exactness verified on this hardware/runtime only."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
