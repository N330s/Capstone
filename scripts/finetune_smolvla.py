"""Full-window local SmolVLA SFT with held-out probes and resumable checkpoints.

Uses the pinned upstream policy, processors, optimizer and scheduler. No network,
API token, simulator, robot command or Hub upload is used by this process.
"""
import argparse
import gc
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data/hf_cache/datasets"))
sys.path.insert(0, str(ROOT))
from learning.vla.window_training import CommandWindows, validate_budget
from learning.vla.runtime_io import write_json, write_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export", "checkpoint", "backbone", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--max-seconds", type=float, default=14000)
    parser.add_argument("--runtime-ceiling-seconds", type=int, choices=(14400, 28800), default=14400,
                        help="Eight-hour ceiling requires explicit user authorization")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--probes", type=int, default=64)
    parser.add_argument("--onset-index", type=Path, help="Source-bound initial-wait ablation; never an actor input")
    parser.add_argument("--onset-collection", type=Path, help="Original collection for independent index verification")
    parser.add_argument("--onset-sampling", choices=("uniform", "skip-survey"), default="uniform")
    parser.add_argument("--final-checkpoint-only", action="store_true",
                        help="Bounded comparison: retain only final, no best/periodic/recovery snapshots")
    parser.add_argument("--qualification-only", action="store_true",
                        help="At most 12 stop-after updates with final-only; keep logs/metrics but no model copy")
    parser.add_argument("--resume", type=Path, help="Trusted local checkpoint directory")
    parser.add_argument("--stop-after", type=int, help="Additional update cap for resume qualification")
    parser.add_argument("--checkpoint-seconds", type=float, default=0, help="Additional wall-clock recovery cadence; zero disables")
    parser.add_argument("--initial-checkpoint-step", type=int, default=0)
    parser.add_argument("--keep-recent-checkpoints", type=int, default=0,
                        help="Opt-in prune superseded completed checkpoints, preserving initial/best/final; zero retains all")
    args = parser.parse_args()
    validate_budget(args.max_seconds, args.steps, args.batch_size, args.learning_rate,
                    ceiling=args.runtime_ceiling_seconds)
    if (not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds < 0
            or args.initial_checkpoint_step < 0 or args.keep_recent_checkpoints < 0):
        raise ValueError("Nonnegative finite recovery settings required")
    if min(args.eval_every, args.save_every, args.probes) < 1:
        raise ValueError("Positive evaluation/checkpoint/probe intervals required")
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("Positive stop-after required")
    if args.seed < 0:
        raise ValueError("Nonnegative seed required")
    if bool(args.onset_index) != bool(args.onset_collection) or (args.onset_sampling != "uniform" and not args.onset_index):
        raise ValueError("Onset sampling requires an independently verified index and collection")
    if args.final_checkpoint_only and (args.resume or args.initial_checkpoint_step or args.checkpoint_seconds or args.keep_recent_checkpoints):
        raise ValueError("Final-only comparisons cannot also request resume/recovery/retention")
    if args.qualification_only and (not args.final_checkpoint_only or not args.stop_after or args.stop_after > 12):
        raise ValueError("No-checkpoint qualification is limited to at most 12 explicit stop-after updates")
    for name in ("export", "checkpoint", "backbone", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if any(args.output.is_relative_to(p) or p.is_relative_to(args.output)
           for p in (args.export, args.checkpoint, args.backbone)):
        raise ValueError("Output must be disjoint from model and dataset")
    if args.resume is None:
        args.output.mkdir(parents=True, exist_ok=False)
    elif not args.output.is_dir():
        raise ValueError("Resume requires the original output directory")
    started = time.monotonic()
    deadline = started + args.max_seconds

    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from learning.vla.smolvla_compat import masked_forward
    from learning.vla.smolvla_data import CONTRACT, LEROBOT_VERSION, sha256
    from learning.vla.training_state import capture_rng, restore_rng, cpu_state
    from learning.vla.training_precision import promote_trainable_fp32, restore_checkpoint_trainable_dtypes, assert_fp32_optimizer
    from learning.vla.training_safety import checkpoint_budget_bytes, require_checkpoint_space, training_space_available

    if version("lerobot") != LEROBOT_VERSION or not torch.cuda.is_available():
        raise RuntimeError("Pinned LeRobot and CUDA required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support required for this qualified configuration")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    receipt_path = args.export / "openarm_export.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["contract"] != CONTRACT or receipt["lerobot_version"] != LEROBOT_VERSION:
        raise ValueError("Unreviewed export contract")
    records = {s: receipt["datasets"][s]["episodes"] for s in ("train", "validation")}
    if set(r["episode"] for r in records["train"]) & set(r["episode"] for r in records["validation"]):
        raise ValueError("Episode split leakage")
    windows = {s: CommandWindows(r) for s, r in records.items()}
    plans = {s: w.probes(args.probes) for s, w in windows.items()}
    onset_index, eligible = None, None
    if args.onset_index:
        from learning.vla.onset_sampling import verify_index, sample_indices, audited_forward, chunk_errors
        onset_index = json.loads(args.onset_index.read_text())
        verify_index(onset_index, args.onset_collection, args.export)
        if args.onset_sampling == "skip-survey":
            eligible = onset_index["eligible"]
        plans["train_phase"] = onset_index["phase_probes"]
    binding = {"export_receipt_sha256": sha256(receipt_path),
               "base_weights_sha256": sha256(args.checkpoint / "model.safetensors"),
               "base_config_sha256": sha256(args.checkpoint / "config.json"),
               "steps": args.steps, "batch_size": args.batch_size,
               "learning_rate": args.learning_rate, "seed": args.seed,
               "probes": plans, "precision": "bf16", "lerobot": LEROBOT_VERSION,
               "trainer_sha256": sha256(Path(__file__)),
               "trainable_storage": "float32",
               "precision_helper_sha256": sha256(ROOT / "learning/vla/training_precision.py"),
               "sampling": "uniform commands with replacement; masked terminal chunks"}
    if onset_index:
        binding.update(onset_sampling=args.onset_sampling,
            onset_index_sha256=sha256(args.onset_index),
            onset_helper_sha256=sha256(ROOT / "learning/vla/onset_sampling.py"),
            loss_audit_helper_sha256=sha256(ROOT / "learning/vla/pickup_onset.py"),
            final_checkpoint_only=args.final_checkpoint_only,
            qualification_only=args.qualification_only,
            eligible_train_commands=len(eligible) if eligible is not None else len(windows["train"]),
            excluded_train_commands=len(onset_index["excluded"]) if eligible is not None else 0)
    datasets = {s: LeRobotDataset(receipt["datasets"][s]["repo_id"], root=args.export / s,
                                 delta_timestamps={"action": [i / 50 for i in range(50)]},
                                 video_backend="pyav") for s in records}
    if any(len(datasets[s]) != len(windows[s]) for s in records):
        raise ValueError("Export length differs from receipt")
    if args.resume:
        args.resume = args.resume.resolve()
        if not args.resume.is_relative_to(args.output):
            raise ValueError("Resume must use this run's checkpoint")
        if not (args.resume / "complete.json").is_file():
            raise ValueError("Incomplete checkpoint; do not resume partial writes")
        state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=False)
        if state["binding"] != binding:
            raise ValueError("Resume dataset/config mismatch")
        policy = SmolVLAPolicy.from_pretrained(str(args.resume / "pretrained_model"))
        restore_checkpoint_trainable_dtypes(policy, args.resume / "pretrained_model")
        pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.resume / "pretrained_model"))
    else:
        config = PreTrainedConfig.from_pretrained(str(args.checkpoint))
        config.input_features, config.pretrained_path = {}, args.checkpoint
        config.vlm_model_name, config.load_vlm_weights = str(args.backbone), False
        config.device, config.push_to_hub = "cuda", False
        config.freeze_vision_encoder = config.train_expert_only = config.train_state_proj = True
        policy = make_policy(config, ds_meta=datasets["train"].meta)
        restore_checkpoint_trainable_dtypes(policy, args.checkpoint)
        pre, post = make_pre_post_processors(policy.config, dataset_stats=datasets["train"].meta.stats)
    precision_audit = promote_trainable_fp32(policy)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    checkpoint_bytes = checkpoint_budget_bytes(
        sum(p.numel() * p.element_size() for p in policy.parameters())
        + sum(p.numel() * p.element_size() for p in policy.buffers()),
        sum(p.numel() * p.element_size() for p in trainable))
    require_checkpoint_space(shutil.disk_usage(args.output).free, (1 if args.final_checkpoint_only else 2) * checkpoint_bytes)
    optim_cfg = policy.config.get_optimizer_preset()
    optim_cfg.lr = args.learning_rate
    optimizer = optim_cfg.build(trainable)
    assert_fp32_optimizer(policy, optimizer)
    schedule_cfg = policy.config.get_scheduler_preset()
    schedule_cfg.peak_lr, schedule_cfg.decay_lr = args.learning_rate, args.learning_rate * .025
    schedule_cfg.num_decay_steps = args.steps
    schedule_cfg.num_warmup_steps = min(schedule_cfg.num_warmup_steps, max(1, args.steps // 30))
    scheduler = schedule_cfg.build(optimizer, args.steps)
    sampler = random.Random(args.seed + 1)
    step, best, initial, elapsed_prior, checks = 0, float("inf"), None, 0., []
    if args.resume:
        optimizer.load_state_dict(state["optimizer"])
        assert_fp32_optimizer(policy, optimizer)
        scheduler.load_state_dict(state["scheduler"])
        sampler.setstate(state["sampler"])
        step, best, initial = state["step"], state["best_validation"], state["initial_evaluation"]
        elapsed_prior, checks = state["elapsed_s"], state["evaluations"]
        restore_rng(state["rng"])
        del state
        deadline = started + max(0., args.max_seconds - elapsed_prior)
    write_json(args.output / "run_config.json", {**binding, "max_seconds": args.max_seconds,
               "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
               "checkpoint_seconds": args.checkpoint_seconds,
               "initial_checkpoint_step": args.initial_checkpoint_step,
               "keep_recent_checkpoints": args.keep_recent_checkpoints,
               "runtime_ceiling_seconds": args.runtime_ceiling_seconds,
               "trainable_parameters": sum(p.numel() for p in trainable),
               "precision_audit": precision_audit,
               "disk_reserve_gib": 25, "checkpoint_budget_bytes": checkpoint_bytes,
               "source_splits": receipt["splits"], "robot_execution": False})

    def emit(event):
        event = {"step": step, "elapsed_s": elapsed_prior + time.monotonic() - started, **event}
        with (args.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        if not write_status(args.output / "status.json", event):
            print(json.dumps({"type": "telemetry_warning", "step": step,
                              "reason": "Status snapshot locked; optimizer continues, event retained in JSONL/stdout"}), flush=True)
        print(json.dumps(event), flush=True)
        return event

    def evaluate():
        saved_rng = capture_rng()
        policy.eval()
        scores = {}
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for split, samples in plans.items():
                    values, joint_errors, gripper_errors, physical_values, phase_rows = [], [], [], [], []
                    for n, sample in enumerate(samples):
                        dataset_split = "train" if split == "train_phase" else split
                        raw_batch = default_collate([datasets[dataset_split][sample["index"]]])
                        batch = pre(raw_batch)
                        generator = torch.Generator(device="cuda").manual_seed(args.seed + 2000 + n)
                        noise = torch.randn((1, 50, policy.config.max_action_dim), device="cuda", generator=generator)
                        forward_kwargs = {"noise": noise, "time": torch.full((1,), .5, device="cuda")}
                        if onset_index:
                            loss, _, audit = audited_forward(policy, batch, masked_forward, **forward_kwargs)
                            physical_values.append(audit["actual_dimensions_mean"])
                        else:
                            loss, _ = masked_forward(policy, batch, **forward_kwargs)
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite held-out loss")
                        values.append(float(loss))
                        action_indices = {int(k * len(samples) / min(8, len(samples)))
                                          for k in range(min(8, len(samples)))}
                        if n in action_indices or split == "train_phase":
                            actor = {k: v for k, v in batch.items() if k not in
                                     ("action", "action_is_pad", "actions_id_pad")}
                            policy.reset()
                            prediction = post(policy.predict_action_chunk(actor, noise=noise))
                            target = raw_batch["action"].to(prediction.device)
                            valid = (~raw_batch["action_is_pad"]).to(prediction.device)
                            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                                raise RuntimeError("Invalid decoded action chunk")
                            error = (prediction - target).abs()[valid]
                            joint_errors.append(error[:, :7].mean(dim=0).float().cpu().tolist())
                            gripper_errors.append(float(error[:, 7].mean()) * 1000)
                            if split == "train_phase":
                                pred = prediction[0].float().cpu().numpy()
                                truth = target[0].float().cpu().numpy()
                                phase_rows.append({**sample, "loss_audit": audit,
                                    "prefix_errors": chunk_errors(pred, truth, valid[0].cpu().numpy()),
                                    "predicted_finger_mm": (pred[:, 7]*1000).tolist(),
                                    "target_finger_mm": (truth[:, 7]*1000).tolist()})
                    scores[split] = {"mean": sum(values) / len(values), "samples": values,
                                     "action_probe_count": len(joint_errors),
                                     "joint_mae_rad": np.mean(joint_errors, axis=0).tolist(),
                                     "gripper_mae_mm": sum(gripper_errors) / len(gripper_errors)}
                    if onset_index:
                        scores[split].update(actual_8_channel_mean=float(np.mean(physical_values)),
                                             physical_loss_samples=physical_values)
                    if phase_rows:
                        scores[split]["phase_rows"] = phase_rows
        finally:
            restore_rng(saved_rng)
        return emit({"type": "evaluation", "scores": scores})

    def save(label):
        parent = args.output / "checkpoints"
        parent.mkdir(exist_ok=True)
        target = parent / f"step_{step:07d}_{label}"
        if target.exists():
            if (target / "complete.json").is_file():
                return target
            raise RuntimeError("Refusing to reuse an incomplete checkpoint")
        require_checkpoint_space(shutil.disk_usage(args.output).free, checkpoint_bytes)
        target.mkdir()
        emit({"type": "checkpoint_started", "path": str(target)})
        bundle = target / "pretrained_model"
        policy.save_pretrained(bundle)
        pre.save_pretrained(bundle)
        post.save_pretrained(bundle)
        saved_state = {"binding": binding, "step": step, "best_validation": best,
                    "initial_evaluation": initial, "evaluations": checks,
                    "elapsed_s": elapsed_prior + time.monotonic() - started,
                    "optimizer": cpu_state(optimizer.state_dict()), "scheduler": scheduler.state_dict(),
                    "rng": capture_rng(), "sampler": sampler.getstate()}
        torch.save(saved_state, target / "training_state.pt")
        write_json(target / "complete.json", {"step": step, "label": label, "binding": binding,
                                              "elapsed_s": saved_state["elapsed_s"]})
        write_json(args.output / "latest_checkpoint.json", {"path": str(target), "step": step})
        emit({"type": "checkpoint_complete", "path": str(target)})
        return target

    def retention():
        if not args.keep_recent_checkpoints:
            return
        from learning.vla.checkpoint_retention import prune_checkpoints
        protected = []
        for name in ("best_checkpoint.json", "latest_checkpoint.json"):
            path = args.output / name
            if path.exists():
                protected.append(json.loads(path.read_text())["path"])
        removed = prune_checkpoints(args.output / "checkpoints", binding, protected,
                                    args.keep_recent_checkpoints)
        if removed:
            emit({"type": "checkpoint_pruned", "removed": removed,
                  "retained": "Initial, best, final and most recent recovery checkpoints"})

    emit({"type": "started", "resumed": bool(args.resume), "cuda_memory_free_gb": torch.cuda.mem_get_info()[0] / 1e9})
    if initial is None:
        initial = evaluate()
        checks.append(initial)
        best = initial["scores"]["validation"]["mean"]
    next_hour = (int(elapsed_prior // 3600) + 1) * 3600
    next_recovery = elapsed_prior + args.checkpoint_seconds
    initial_step = step
    recent_times, reason = [], "step_budget"
    last_checkpoint = None
    while step < args.steps:
        # Reserve final evaluation/checkpoint time; watchdog enforces the outer limit.
        if time.monotonic() >= deadline - 120:
            reason = "wall_time_budget"
            break
        if (args.output / "STOP").exists():
            reason = "stop_file"
            break
        if not training_space_available(shutil.disk_usage(args.output).free, checkpoint_bytes,
                                        final_only=args.final_checkpoint_only):
            reason = "low_disk"
            break
        tick = time.monotonic()
        indices = (sample_indices(sampler, args.batch_size, len(windows["train"]), eligible) if onset_index
                   else [sampler.randrange(len(windows["train"])) for _ in range(args.batch_size)])
        batch = pre(default_collate([datasets["train"][i] for i in indices]))
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = masked_forward(policy, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss; stopping")
        loss.backward()
        if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
            raise RuntimeError("Frozen parameters received gradients")
        norm = torch.nn.utils.clip_grad_norm_(trainable, optim_cfg.grad_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        if step == initial_step or step % 100 == 0:
            assert_fp32_optimizer(policy, optimizer)
        scheduler.step()
        step += 1
        duration = time.monotonic() - tick
        recent_times = (recent_times + [duration])[-100:]
        emit({"type": "update", "loss": float(loss.detach()), "grad_norm": float(norm),
              "lr": scheduler.get_last_lr()[0], "samples": [windows["train"].locate(i) for i in indices],
              "update_seconds": duration, "mean_recent_update_seconds": sum(recent_times) / len(recent_times),
              "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
              "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9})
        elapsed = elapsed_prior + time.monotonic() - started
        hourly = elapsed >= next_hour
        recovery_due = bool(args.checkpoint_seconds and elapsed >= next_recovery)
        initial_recovery = bool(args.initial_checkpoint_step and step == args.initial_checkpoint_step)
        saved_this_cycle = False
        if step % args.eval_every == 0 or hourly or recovery_due:
            check = evaluate()
            checks.append(check)
            value = check["scores"]["validation"]["mean"]
            if value < best:
                best = value
                # Save best only on checkpoint boundaries/hourly, not every loss probe.
                if not args.final_checkpoint_only and (step % args.save_every == 0 or hourly or recovery_due or args.keep_recent_checkpoints):
                    last_checkpoint = save("best")
                    saved_this_cycle = True
                    write_json(args.output / "best_checkpoint.json", {"path": str(last_checkpoint), "validation_mean": value})
        if not args.final_checkpoint_only and (step % args.save_every == 0 or hourly or recovery_due or initial_recovery) and not saved_this_cycle:
            last_checkpoint = save("recovery_initial" if initial_recovery else "periodic")
            saved_this_cycle = True
        if saved_this_cycle:
            retention()
            next_recovery = elapsed_prior + time.monotonic() - started + args.checkpoint_seconds
        if hourly:
            emit({"type": "hourly_check", "finite": True, "best_validation_mean": best,
                  "disk_free_gb": shutil.disk_usage(args.output).free / 1e9})
            next_hour += 3600
        if args.stop_after and step - initial_step >= args.stop_after:
            reason = "qualification_stop_after"
            break
    final = evaluate()
    checks.append(final)
    best = min(best, final["scores"]["validation"]["mean"])
    last_checkpoint = None if args.qualification_only else save("final")
    retention()
    report = {"completed": True, "stop_reason": reason, "updates": step,
              "initial_evaluation": initial, "final_evaluation": final,
              "best_validation_mean": best, "evaluations": checks,
              "final_checkpoint": str(last_checkpoint) if last_checkpoint else None,
              "qualification_only": args.qualification_only, "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
              "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
              "elapsed_s": elapsed_prior + time.monotonic() - started,
              "binding": binding, "limits": "Fixed timeline flow-loss probes, not full validation or autonomous task success. No robot execution."}
    write_json(args.output / "report.json", report)
    emit({"type": "finished", "stop_reason": reason, "final_checkpoint": str(last_checkpoint)})


if __name__ == "__main__":
    main()
