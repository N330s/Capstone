"""Explicit wall-clock extension after a successful campaign; low-LR SFT tail."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HOME", str(ROOT / "data/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data/hf_cache/datasets"))
from learning.vla.runtime_io import read_json, write_json, write_status


def deadline_timestamp(value):
    deadline = datetime.fromisoformat(value)
    if deadline.tzinfo is None:
        raise ValueError("Deadline must include an explicit timezone")
    return deadline.timestamp()


def source_checkpoint(run, export):
    from learning.vla.smolvla_data import sha256
    report = read_json(run / "report.json")
    if not report.get("completed"):
        raise ValueError("Prior training did not complete; refusing unattended continuation")
    checkpoint = Path(report["final_checkpoint"]).resolve()
    if not checkpoint.is_relative_to((run / "checkpoints").resolve()):
        raise ValueError("Source checkpoint outside original run")
    marker = read_json(checkpoint / "complete.json")
    if marker["binding"] != report["binding"]:
        raise ValueError("Source checkpoint/report mismatch")
    if marker["binding"]["export_receipt_sha256"] != sha256(export / "openarm_export.json"):
        raise ValueError("Source dataset changed")
    return checkpoint, marker


def worker(args):
    # All heavy imports occur only after the original campaign has ended.
    import shutil
    import torch
    from importlib.metadata import version
    from torch.utils.data import default_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from learning.vla.smolvla_policy import SmolVLAController
    from learning.vla.smolvla_compat import masked_forward
    from learning.vla.smolvla_data import sha256, LEROBOT_VERSION
    from learning.vla.training_state import capture_rng, restore_rng, cpu_state
    from learning.vla.window_training import CommandWindows
    from learning.vla.checkpoint_retention import prune_checkpoints
    checkpoint, marker = source_checkpoint(args.source_run, args.export)
    if marker["binding"].get("trainable_storage") != "float32":
        raise ValueError("Legacy BF16 optimizer continuation is disabled after precision repair; warm-start weights with a new FP32 optimizer instead")
    if version("lerobot") != LEROBOT_VERSION or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Qualified LeRobot and BF16 GPU required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if shutil.disk_usage(args.output).free < 10 * 1024**3:
        raise RuntimeError("Less than 10 GiB free for continuation")
    controller = SmolVLAController(checkpoint / "pretrained_model", args.backbone)
    policy, pre, post = controller.model, controller.pre, controller.post
    from learning.vla.training_precision import promote_trainable_fp32, assert_fp32_optimizer
    promote_trainable_fp32(policy)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    source_config = read_json(args.source_run / "run_config.json")
    if sum(p.numel() for p in trainable) != source_config["trainable_parameters"]:
        raise ValueError("Trainable parameter set changed")
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    if state["binding"] != marker["binding"] or state["step"] != marker["step"]:
        raise ValueError("Source optimizer state differs from completion marker")
    optimizer = policy.config.get_optimizer_preset().build(trainable)
    optimizer.load_state_dict(state["optimizer"])
    assert_fp32_optimizer(policy, optimizer)
    rates = [group["lr"] for group in optimizer.param_groups]
    if not rates or any(not math.isfinite(rate) or not 0 < rate <= 1e-4 for rate in rates):
        raise ValueError("Invalid inherited learning rate")
    sampler = random.Random()
    sampler.setstate(state["sampler"])
    inherited_rng = state["rng"]
    step = initial_step = state["step"]
    del state
    receipt = read_json(args.export / "openarm_export.json")
    windows = {s: CommandWindows(receipt["datasets"][s]["episodes"]) for s in ("train", "validation")}
    datasets = {s: LeRobotDataset(receipt["datasets"][s]["repo_id"], root=args.export / s,
                                 delta_timestamps={"action": [i / 50 for i in range(50)]},
                                 video_backend="pyav") for s in windows}
    if any(len(datasets[s]) != len(windows[s]) for s in windows):
        raise ValueError("Dataset length mismatch")
    probes = marker["binding"]["probes"]["validation"]
    batch_size = marker["binding"]["batch_size"]
    binding = {**marker["binding"], "trainer_sha256": sha256(Path(__file__)),
               "source_checkpoint": str(checkpoint),
               "source_weights_sha256": sha256(checkpoint / "pretrained_model/model.safetensors"),
               "continuation_until": args.until, "schedule": "constant inherited final learning rate",
               "learning_rates": rates}
    write_json(args.output / "run_config.json", {**binding, "robot_execution": False,
               "trainable_parameters": sum(p.numel() for p in trainable), "gpu": torch.cuda.get_device_name()})
    restore_rng(inherited_rng)
    started = time.monotonic()
    stop_at = time.monotonic() + max(0, deadline_timestamp(args.until) - time.time())
    def emit(kind, **values):
        event = {"type": kind, "step": step, "elapsed_s": time.monotonic() - started, **values}
        with (args.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        write_status(args.output / "status.json", event)
        print(json.dumps(event), flush=True)
        return event
    def evaluate():
        rng = capture_rng()
        policy.eval()
        values = []
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for n, sample in enumerate(probes):
                    batch = pre(default_collate([datasets["validation"][sample["index"]]]))
                    noise = torch.randn((1, 50, policy.config.max_action_dim), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(binding["seed"] + 2000 + n))
                    loss, _ = masked_forward(policy, batch, noise=noise, time=torch.full((1,), .5, device="cuda"))
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite validation loss")
                    values.append(float(loss))
        finally:
            restore_rng(rng)
        return emit("evaluation", scores={"validation": {"mean": sum(values) / len(values), "samples": values}})
    best = float("inf")
    def save(label):
        if shutil.disk_usage(args.output).free < 5 * 1024**3:
            raise RuntimeError("Insufficient checkpoint space")
        target = args.output / "checkpoints" / f"step_{step:07d}_{label}"
        target.mkdir(parents=True, exist_ok=False)
        emit("checkpoint_started", path=str(target))
        bundle = target / "pretrained_model"
        policy.save_pretrained(bundle)
        pre.save_pretrained(bundle)
        post.save_pretrained(bundle)
        saved = {"binding": binding, "step": step, "best_validation": best,
                 "optimizer": cpu_state(optimizer.state_dict()), "rng": capture_rng(),
                 "sampler": sampler.getstate(), "elapsed_s": time.monotonic() - started,
                 "schedule": {"type": "constant", "learning_rates": rates}}
        torch.save(saved, target / "training_state.pt")
        write_json(target / "complete.json", {"step": step, "label": label, "binding": binding,
                                               "elapsed_s": saved["elapsed_s"]})
        write_json(args.output / "latest_checkpoint.json", {"path": str(target), "step": step})
        if label in ("best", "recovery_initial"):
            write_json(args.output / "best_checkpoint.json", {"path": str(target), "validation_mean": best})
        emit("checkpoint_complete", path=str(target))
        protected = [read_json(args.output / n)["path"] for n in ("best_checkpoint.json", "latest_checkpoint.json")]
        removed = prune_checkpoints(args.output / "checkpoints", binding, protected, 2)
        if removed:
            emit("checkpoint_pruned", removed=removed)
        return target
    emit("started", source_step=initial_step, batch_size=batch_size, learning_rates=rates)
    initial = evaluate()
    best = initial["scores"]["validation"]["mean"]
    save("recovery_initial")
    next_save = time.monotonic() + 900
    checks, reason = [initial], "wall_time_budget"
    recent = []
    # Reserve five minutes for final evaluation and durable checkpoint writing.
    while time.monotonic() < stop_at - 300:
        if (args.output / "STOP").exists():
            reason = "stop_file"
            break
        if shutil.disk_usage(args.output).free < 8 * 1024**3:
            reason = "low_disk"
            break
        tick = time.monotonic()
        indices = [sampler.randrange(len(windows["train"])) for _ in range(batch_size)]
        policy.train()
        batch = pre(default_collate([datasets["train"][i] for i in indices]))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = masked_forward(policy, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite continuation loss")
        loss.backward()
        if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
            raise RuntimeError("Frozen parameters received gradients")
        norm = torch.nn.utils.clip_grad_norm_(trainable, policy.config.get_optimizer_preset().grad_clip_norm,
                                              error_if_nonfinite=True)
        optimizer.step()
        step += 1
        recent = (recent + [time.monotonic() - tick])[-100:]
        emit("update", loss=float(loss.detach()), grad_norm=float(norm), lr=rates[0],
             update_seconds=recent[-1], mean_recent_update_seconds=sum(recent) / len(recent),
             peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
             peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
             samples=[windows["train"].locate(i) for i in indices])
        if step == initial_step + 25 or time.monotonic() >= next_save:
            check = evaluate()
            checks.append(check)
            value = check["scores"]["validation"]["mean"]
            improved = value < best
            best = min(best, value)
            save("best" if improved else "periodic")
            next_save = time.monotonic() + 900
    final = evaluate()
    checks.append(final)
    improved = final["scores"]["validation"]["mean"] < best
    best = min(best, final["scores"]["validation"]["mean"])
    target = save("final")
    if improved:
        write_json(args.output / "best_checkpoint.json", {"path": str(target), "validation_mean": best})
    write_json(args.output / "report.json", {"completed": True, "stop_reason": reason, "updates": step,
        "additional_updates": step - initial_step, "binding": binding, "final_checkpoint": str(target),
        "initial_evaluation": initial, "final_evaluation": final, "evaluations": checks,
        "best_validation_mean": best, "robot_execution": False,
        "limits": "Low-LR continuation with inherited optimizer; fixed validation probes; no autonomous success claim."})
    emit("finished", final_checkpoint=str(target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--until", required=True)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--source-campaign", type=Path, required=True)
    p.add_argument("--source-pid", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--export", type=Path, default=ROOT / "data/drive_v2c30_20261012/export_all_v1")
    p.add_argument("--backbone", type=Path, default=ROOT / "data/models/smolvlm_backbone")
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    for name in ("source_run", "source_campaign", "output", "export", "backbone"):
        setattr(args, name, getattr(args, name).resolve())
    until = deadline_timestamp(args.until)
    if not 0 < until - time.time() <= 9 * 3600:
        raise ValueError("Explicit future deadline within nine hours required")
    if args.worker:
        worker(args)
        return
    import psutil
    from scripts.run_smolvla_job import arm_deadline, kill_owned_process, watch_training
    args.output.mkdir(parents=True, exist_ok=False)
    monitor = args.output.with_name(args.output.name + "_monitor")
    monitor.mkdir(exist_ok=False)
    write_json(args.output / "authorization.json", {"until": args.until,
        "supersedes": "eight-hour cumulative training cap", "source_run": str(args.source_run),
        "strategy": "finish original schedule, then constant inherited-LR optimizer continuation",
        "requested_unix": time.time(), "robot_execution": False})
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (monitor / "awake.log").open("w") as awake_log:
        awake = subprocess.Popen([sys.executable, str(ROOT / "scripts/keep_training_awake.py"),
            "--pid", str(os.getpid()), "--max-seconds", str(math.ceil(until - time.time()))],
            stdout=awake_log, stderr=subprocess.STDOUT, creationflags=flags)
        process = None
        try:
            # Bind process identity; an unrelated reused PID cannot delay or be killed by this job.
            try:
                source = psutil.Process(args.source_pid)
                source.create_time()
            except psutil.NoSuchProcess:
                source = None
            while source is not None and source.is_running():
                if time.time() >= until - 600:
                    raise RuntimeError("No continuation time remains before deadline")
                if awake.poll() is not None:
                    raise RuntimeError("Sleep-prevention helper exited unexpectedly")
                write_status(args.output / "status.json", {"state": "waiting_for_original_campaign",
                    "deadline": args.until, "updated_unix": time.time(), "source_pid": args.source_pid})
                time.sleep(30)
            if not (args.source_campaign / "completion.json").exists():
                raise RuntimeError("Original campaign ended without a completion record")
            source_checkpoint(args.source_run, args.export)
            remaining = until - time.time()
            if remaining < 600:
                raise RuntimeError("Insufficient time for continuation and checkpoint")
            command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
            with (monitor / "train.log").open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
                fired = threading.Event()
                timer = arm_deadline(process, max(0, until - time.time()), fired)
                started = time.monotonic()
                write_json(monitor / "launch.json", {"pid": process.pid, "supervisor_pid": os.getpid(),
                    "command": command, "started_unix": time.time(), "deadline": args.until,
                    "max_seconds": remaining, "api_tokens_used": False})
                try:
                    reason = watch_training(process, SimpleNamespace(output=args.output, max_seconds=remaining,
                        heartbeat_timeout_seconds=900), monitor, started, flags)
                finally:
                    if process.poll() is None:
                        kill_owned_process(process)
                    code = process.wait()
                    timer.cancel()
            complete = code == 0 and read_json(args.output / "report.json").get("completed")
            write_json(args.output / "completion.json", {"state": "completed" if complete else "failed",
                       "returncode": code, "reason": reason, "deadline": args.until})
            write_status(monitor / "health.json", {"running": False, "returncode": code})
        except Exception as error:
            if process is not None and process.poll() is None:
                kill_owned_process(process)
            write_json(args.output / "completion.json", {"state": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
