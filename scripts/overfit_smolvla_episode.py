"""Bounded one-training-episode memorization diagnostic, not generalization.

Keeps the original trainer, export, validation split and model immutable.
Writes one inference-only checkpoint (no optimizer-resume claim).
"""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HOME", str(ROOT / "data/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data/hf_cache/datasets"))
from learning.vla.runtime_io import write_json
from learning.vla.smolvla_data import CONTRACT, LEROBOT_VERSION, sha256
from learning.vla.window_training import CommandWindows


def episode_span(receipt, name):
    """Return global export indices without changing/relabeling dataset splits."""
    if receipt["contract"] != CONTRACT or receipt["lerobot_version"] != LEROBOT_VERSION:
        raise ValueError("Unreviewed export")
    records = {s: receipt["datasets"][s]["episodes"] for s in ("train", "validation")}
    names = {s: [r["episode"] for r in rows] for s, rows in records.items()}
    if (set(names["train"]) & set(names["validation"])
            or any(len(set(v)) != len(v) for v in names.values())):
        raise ValueError("Duplicate or leaking parent split")
    if name not in names["train"]:
        raise ValueError("Overfit target must be an existing training parent")
    windows = CommandWindows(records["train"])
    position = names["train"].index(name)
    start = windows.ends[position - 1] if position else 0
    return start, windows.ends[position], records["train"][position]


def probe_indices(start, end, count=32):
    if not 0 <= start < end or count < 1:
        raise ValueError("Invalid probe span")
    return sorted({start + min(end - start - 1, int((i + .5) * (end - start) / count))
                   for i in range(count)})


def train_or_verify(args, receipt, span):
    import numpy as np
    import torch
    from importlib.metadata import version
    from torch.utils.data import default_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from learning.vla.smolvla_policy import validate_openarm_features
    from learning.vla.smolvla_compat import masked_forward
    from learning.vla.training_state import capture_rng, restore_rng
    from safetensors.torch import load_file

    if version("lerobot") != LEROBOT_VERSION or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Pinned CUDA/BF16 runtime required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    start, end, record = span
    bundle = args.output / "pretrained_model" if args.phase == "verify" else args.checkpoint
    cfg = PreTrainedConfig.from_pretrained(str(bundle))
    cfg.vlm_model_name, cfg.load_vlm_weights, cfg.device = str(args.backbone), False, "cuda"
    cfg.push_to_hub = False
    validate_openarm_features(cfg)
    policy = SmolVLAPolicy.from_pretrained(str(bundle), config=cfg)
    from learning.vla.training_precision import promote_trainable_fp32, restore_checkpoint_trainable_dtypes, assert_fp32_optimizer
    restore_checkpoint_trainable_dtypes(policy, bundle)
    promote_trainable_fp32(policy)
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(bundle),
        preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": str(args.backbone)},
                                "device_processor": {"device": "cuda"}})
    datasets = {s: LeRobotDataset(receipt["datasets"][s]["repo_id"], root=args.export / s,
        delta_timestamps={"action": [i / 50 for i in range(50)]}, video_backend="pyav")
        for s in ("train", "validation")}
    for split, dataset in datasets.items():
        if len(dataset) != len(CommandWindows(receipt["datasets"][split]["episodes"])):
            raise ValueError("Export lengths differ")
    plans = {"train_episode": probe_indices(start, end),
             "validation_monitor": [p["index"] for p in
                 CommandWindows(receipt["datasets"]["validation"]["episodes"]).probes(8)]}
    events = args.output / "events.jsonl"
    begun = time.monotonic()

    def emit(row):
        row = {"elapsed_s": time.monotonic() - begun, **row}
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        return row

    def evaluate(step):
        rng = capture_rng()
        policy.eval()
        scores = {}
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for label, indices in plans.items():
                    split = "train" if label == "train_episode" else "validation"
                    rows = []
                    for n, index in enumerate(indices):
                        raw = default_collate([datasets[split][index]])
                        batch = pre(raw)
                        gen = torch.Generator(device="cuda").manual_seed(args.seed + 2000 + n)
                        noise = torch.randn((1, 50, cfg.max_action_dim), device="cuda", generator=gen)
                        loss, _ = masked_forward(policy, batch, noise=noise,
                            time=torch.full((1,), .5, device="cuda"))
                        actor = {k: v for k, v in batch.items() if k not in
                                 ("action", "action_is_pad", "actions_id_pad")}
                        policy.reset()
                        pred = post(policy.predict_action_chunk(actor, noise=noise))
                        if pred.shape != (1, 50, 8) or not torch.isfinite(pred).all() or not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite/invalid probe")
                        target = raw["action"].to(pred.device)
                        valid = (~raw["action_is_pad"]).to(pred.device)
                        prefix = min(5, int(valid.sum()))
                        err = (pred - target).abs()
                        rows.append({"index": index, "loss": float(loss),
                            "prefix5_gripper_mae_mm": float(err[0, :prefix, 7].mean()) * 1000,
                            "prefix5_joint_mae_rad": float(err[0, :prefix, :7].mean()),
                            "chunk_gripper_mae_mm": float(err[valid][:, 7].mean()) * 1000})
                    scores[label] = {"samples": rows, **{key: sum(r[key] for r in rows) / len(rows)
                        for key in ("loss", "prefix5_gripper_mae_mm", "prefix5_joint_mae_rad", "chunk_gripper_mae_mm")}}
        finally:
            restore_rng(rng)
        return emit({"type": "evaluation", "step": step, "scores": scores})

    if args.phase == "verify":
        report = json.loads((args.output / "training_report.json").read_text())
        fresh = evaluate(report["updates"])
        for label in plans:
            if abs(fresh["scores"][label]["loss"] - report["final"]["scores"][label]["loss"]) > 1e-8:
                raise ValueError("Fresh checkpoint loss mismatch")
        for name in ("policy_preprocessor_step_5_normalizer_processor.safetensors",
                     "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
            old, new = load_file(str(args.checkpoint / name)), load_file(str(bundle / name))
            if old.keys() != new.keys() or any(not torch.equal(old[k], new[k]) for k in old):
                raise ValueError("Normalization changed")
        write_json(args.output / "verification.json", {"passed": True, "fresh": fresh,
            "normalization_unchanged": True, "weights_sha256": sha256(bundle / "model.safetensors")})
        return

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optim_cfg = cfg.get_optimizer_preset()
    optim_cfg.lr = args.learning_rate
    optimizer = optim_cfg.build(trainable)
    assert_fp32_optimizer(policy, optimizer)
    sched_cfg = cfg.get_scheduler_preset()
    sched_cfg.peak_lr, sched_cfg.decay_lr = args.learning_rate, args.learning_rate * .025
    sched_cfg.num_decay_steps, sched_cfg.num_warmup_steps = args.steps, min(100, max(1, args.steps // 30))
    scheduler = sched_cfg.build(optimizer, args.steps)
    sampler = random.Random(args.seed + 1)
    initial = evaluate(0)
    step, reason, checks = 0, "step_budget", [initial]
    while step < args.steps:
        if time.monotonic() - begun > args.train_seconds - 120:
            reason = "time_budget"
            break
        if shutil.disk_usage(args.output).free < 25.7 * 2**30:
            reason = "disk_reserve"
            break
        indices = [sampler.randrange(start, end) for _ in range(args.batch_size)]
        raw = default_collate([datasets["train"][i] for i in indices])
        # Independently check the dataset's parent id, not just RNG bounds.
        if not torch.all(raw["episode_index"] == receipt["datasets"]["train"]["episodes"].index(record)):
            raise ValueError("Sampled a different training episode")
        batch = pre(raw)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = masked_forward(policy, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
            raise RuntimeError("Frozen parameters received gradients")
        norm = torch.nn.utils.clip_grad_norm_(trainable, optim_cfg.grad_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        if step == 0:
            assert_fp32_optimizer(policy, optimizer)
        scheduler.step()
        step += 1
        emit({"type": "update", "step": step, "loss": float(loss.detach()), "grad_norm": float(norm),
              "indices": indices, "lr": scheduler.get_last_lr()[0],
              "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9})
        if step % 250 == 0:
            checks.append(evaluate(step))
    final = evaluate(step)
    destination = args.output / "pretrained_model"
    model_bytes = sum(v.numel() * v.element_size() for v in policy.state_dict().values())
    if shutil.disk_usage(args.output).free < 25 * 2**30 + model_bytes + 16 * 2**20:
        raise RuntimeError("Full FP32 checkpoint would violate 25 GiB reserve")
    policy.save_pretrained(destination)
    pre.save_pretrained(destination)
    post.save_pretrained(destination)
    write_json(args.output / "training_report.json", {"completed": True, "updates": step,
        "stop_reason": reason, "initial": initial, "final": final, "evaluations": checks,
        "training_episode": record, "train_index_span": [start, end],
        "batch_size": args.batch_size, "trainable_parameters": sum(p.numel() for p in trainable),
        "elapsed_s": time.monotonic() - begun, "optimizer_resume_supported": False,
        "checkpoint_selection": "fixed final update; not selected on validation",
        "trainable_storage": "float32; BF16 autocast computation",
        "limits": "Deliberate training-parent memorization. No held-out success or generalization claim."})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, default=ROOT / "data/drive_v2c30_20261012/export_all_v1")
    p.add_argument("--episode", type=Path, default=ROOT / "data/drive_v2c30_20261012/merged_all_v1/train_20002")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/v2c30_sft_overnight_v1/checkpoints/step_0009426_best/pretrained_model")
    p.add_argument("--backbone", type=Path, default=ROOT / "data/models/smolvlm_backbone")
    p.add_argument("--snapshot", type=Path, default=ROOT / "third_party/capstone_automateData")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--train-seconds", type=int, default=1800)
    p.add_argument("--phase", choices=("supervise", "train", "verify"), default="supervise")
    args = p.parse_args()
    if not (1 <= args.steps <= 2000 and 1 <= args.batch_size <= 16 and 300 <= args.train_seconds <= 1800
            and 0 < args.learning_rate <= 1e-4 and args.seed >= 0):
        raise ValueError("Diagnostic budget exceeded")
    for key in ("export", "episode", "checkpoint", "backbone", "snapshot", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if any(args.output.is_relative_to(v) or v.is_relative_to(args.output) for v in
           (args.export, args.episode, args.checkpoint, args.backbone, args.snapshot)):
        raise ValueError("Output overlaps input")
    receipt = json.loads((args.export / "openarm_export.json").read_text())
    span = episode_span(receipt, args.episode.name)
    meta = json.loads((args.episode / "metadata.json").read_text())
    if (meta["outcome"] != "success" or meta["data_sha256"] != span[2]["data_sha256"]
            or sha256(args.episode / "episode.npz") != meta["data_sha256"]):
        raise ValueError("Successful source episode binding differs")
    from scripts.rollout_smolvla import verify_snapshot
    verify_snapshot(args.snapshot, meta)
    if args.phase != "supervise":
        train_or_verify(args, receipt, span)
        return
    if shutil.disk_usage(ROOT).free < 26.5 * 2**30:
        raise RuntimeError("Need 26.5 GiB free: one weights-only bundle plus 25 GiB reserve")
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    plan.update(export_sha256=sha256(args.export / "openarm_export.json"),
        base_weights_sha256=sha256(args.checkpoint / "model.safetensors"), source_sha256=meta["data_sha256"],
        train_index_span=span[:2], trainer_sha256=sha256(Path(__file__)),
        original_trainer_sha256=sha256(ROOT / "scripts/finetune_smolvla.py"),
        interpretation="Single seen training reset; fixed final checkpoint, no validation selection",
        normalization="inherited unchanged", robot_execution=False, total_seconds=3600)
    write_json(args.output / "plan.json", plan)
    from scripts.run_smolvla_job import arm_deadline, kill_owned_process
    from scripts.test_recorded_observation_control import check_report
    begun, results = time.monotonic(), {}

    def stage(name, command, seconds):
        remaining = 3600 - (time.monotonic() - begun)
        if remaining < 30:
            raise TimeoutError("Total diagnostic deadline")
        fired = threading.Event()
        print(json.dumps({"started": name}), flush=True)
        with (args.output / (name + ".log")).open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            timer = arm_deadline(child, min(seconds, remaining), fired)
            try:
                code = child.wait()
            except BaseException:
                kill_owned_process(child)
                raise
            finally:
                timer.cancel()
        if code or fired.is_set():
            raise RuntimeError(f"{name} failed: returncode={code}, timeout={fired.is_set()}")
        print(json.dumps({"finished": name}), flush=True)

    def rollout(name, mode, bundle):
        destination = args.output / name
        command = [sys.executable, "-u", str(ROOT / "scripts/rollout_smolvla.py"),
            "--snapshot", str(args.snapshot), "--episode", str(args.episode), "--checkpoint", str(bundle),
            "--backbone", str(args.backbone), "--output", str(destination), "--mode", mode,
            "--max-steps", "2000", "--prefix", "5", "--seed", str(args.seed)]
        stage(name, command, 600)
        report = json.loads((destination / "report.json").read_text())
        if mode == "replay":
            check_report(report, mode, plan["base_weights_sha256"], meta["data_sha256"], args.episode.name, 5, args.seed)
        elif (report["recorded_input_assistance"] or report["simulator_state_forced_to_recording"]
              or report["source_data_sha256"] != meta["data_sha256"]
              or report["checkpoint_weights_sha256"] != sha256(bundle / "model.safetensors")
              or report["input_state_source"] != "live_simulation" or report["input_image_source"] != "live_simulation"
              or not report["pipeline_passed"]):
            raise ValueError("Live evaluation contract differs")
        results[name] = report
        write_json(args.output / "progress.json", results)

    rollout("expert_replay", "replay", args.checkpoint)
    rollout("baseline", "policy", args.checkpoint)
    child_args = [sys.executable, "-u", str(Path(__file__)), *sys.argv[1:], "--phase"]
    stage("training", [*child_args, "train"], args.train_seconds)
    stage("verification", [*child_args, "verify"], 300)
    rollout("overfit", "policy", args.output / "pretrained_model")
    if sha256(args.checkpoint / "model.safetensors") != plan["base_weights_sha256"]:
        raise ValueError("Original model changed")
    write_json(args.output / "report.json", {"completed": True, "plan": plan, "conditions": results,
        "training": json.loads((args.output / "training_report.json").read_text()),
        "verification": json.loads((args.output / "verification.json").read_text()),
        "elapsed_s": time.monotonic() - begun,
        "limits": "One memorized training reset and noise seed; no held-out, real-time or sim2real success claim."})


if __name__ == "__main__":
    main()
