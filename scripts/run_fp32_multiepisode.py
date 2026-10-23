"""Offline, bounded full-data FP32 SFT, exact GPU resume gate, then fixed pickup bank.

No automatic restart, checkpoint sweep, new data collection, API calls or robot
commands. Qualification and SFT share four wall-clock hours; paired evaluation
has another 45 minutes, with five minutes reserved for audit/report generation.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.runtime_io import read_json, write_json, write_status
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source
from learning.vla.job_monitoring import heartbeat_age_seconds, evaluation_heartbeat_paths, heartbeat_stalled
from scripts.benchmark_smolvla_batches import gpu_snapshot
from scripts.run_smolvla_job import arm_deadline, kill_owned_process

EXPORT = ROOT / "data/drive_v2c30_20261012/export_all_v1"
BASE = ROOT / "outputs/v2c30_sft_overnight_v1/checkpoints/step_0009426_best/pretrained_model"
BACKBONE = ROOT / "data/models/smolvlm_backbone"
TRAIN_SECONDS, EVAL_SECONDS, REPORT_SECONDS = 14400, 2700, 300


def compare_qualification(reference, split):
    """Independent-process exact restore, including FP32 Adam state and RNG."""
    import torch
    from learning.vla.training_state import assert_state_equal
    ref = reference / "checkpoints/step_0000012_final"
    res = split / "checkpoints/step_0000012_final"
    if sha256(ref / "pretrained_model/model.safetensors") != sha256(res / "pretrained_model/model.safetensors"):
        from safetensors.torch import load_file
        assert_state_equal(load_file(str(ref / "pretrained_model/model.safetensors")),
                           load_file(str(res / "pretrained_model/model.safetensors")), "model")
    a = torch.load(ref / "training_state.pt", map_location="cpu", weights_only=False)
    b = torch.load(res / "training_state.pt", map_location="cpu", weights_only=False)
    for key in ("binding", "step", "optimizer", "scheduler", "rng", "sampler"):
        assert_state_equal(a[key], b[key], key)
    if a["binding"]["trainable_storage"] != "float32":
        raise ValueError("Legacy low-precision checkpoint is not qualified")
    for state in a["optimizer"]["state"].values():
        for key in ("exp_avg", "exp_avg_sq"):
            if state[key].dtype != torch.float32 or not torch.isfinite(state[key]).all():
                raise ValueError("Nonfinite or non-FP32 Adam moments")
    events = []
    for folder in (reference, split):
        updates = [e for e in map(json.loads, (folder / "events.jsonl").read_text().splitlines()) if e["type"] == "update"]
        events.append([{k: e[k] for k in ("step", "loss", "grad_norm", "lr", "samples")} for e in updates])
    assert_state_equal(events[0], events[1], "exact_update_history")
    result = {"passed": True, "updates": 12, "split_updates": [8, 4],
        "model_sha256": sha256(ref / "pretrained_model/model.safetensors"),
        "compared": ["model", "optimizer", "scheduler", "rng", "sampler", "binding", "loss_grad_lr_data"],
        "trainable_and_adam_storage": "float32"}
    del a, b, events
    gc.collect()
    return result


def discard_new_split_checkpoints(campaign, split):
    """Only this newly created disposable qualification fork; retain reference."""
    from learning.vla.checkpoint_retention import _no_reparse_points
    if split.resolve() != (campaign / "qualification/split").resolve():
        raise ValueError("Not this campaign's disposable qualification fork")
    rows = []
    for step in (8, 12):
        path = split / f"checkpoints/step_{step:07d}_final"
        if path.resolve().parent != (split / "checkpoints").resolve():
            raise ValueError("Cleanup target escaped qualification fork")
        _no_reparse_points(path)
        marker = read_json(path / "complete.json")
        if marker["step"] != step or marker["label"] != "final":
            raise ValueError("Unexpected qualification checkpoint")
        rows.append({"path": str(path.resolve()), "bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
            "model_sha256": sha256(path / "pretrained_model/model.safetensors"),
            "complete_sha256": sha256(path / "complete.json"), "recoverable": False})
    write_json(campaign / "qualification/disposable_checkpoint_retention.json", {
        "exact_resume_gate": read_json(campaign / "qualification/exact_resume.json"),
        "reference_preserved": str(campaign / "qualification/reference"), "targets": rows})
    if os.name != "nt":
        raise RuntimeError("This campaign's native cleanup is qualified for Windows only")
    for row in rows:
        literal = row["path"].replace("'", "''")
        parent = str((split / "checkpoints").resolve()).replace("'", "''")
        script = ("$ErrorActionPreference='Stop'; $target=(Resolve-Path -LiteralPath '" + literal + "').Path; "
            "if ((Split-Path -Parent $target) -ne '" + parent + "') { throw 'Unexpected parent' }; "
            "$items=@(Get-Item -LiteralPath $target)+@(Get-ChildItem -LiteralPath $target -Force -Recurse); "
            "if (@($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Reparse point' }; "
            "Remove-Item -LiteralPath $target -Recurse -Force")
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                       check=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-python", type=Path, required=True, help="Interpreter with matplotlib installed")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.parent != ROOT / "outputs" or not args.output.name.startswith("fp32_multiepisode_"):
        raise ValueError("A new direct outputs/fp32_multiepisode_* directory is required")
    if not args.plot_python.is_file():
        raise ValueError("Plotting interpreter missing")
    if shutil.disk_usage(ROOT).free < 37 * 2**30:
        raise RuntimeError("At least 37 GiB free required for qualification and retained full checkpoints")
    import psutil
    gpu_scripts = ("finetune_smolvla.py", "finetune_recovery_smolvla.py", "train_gripper_feedback.py", "rollout_smolvla.py")
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            argv = proc.info["cmdline"] or []
            if proc.pid != os.getpid() and any(Path(arg).name in gpu_scripts for arg in argv):
                raise RuntimeError(f"Another training/rollout process exists: {proc.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    train_deadline = started + TRAIN_SECONDS
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    env = {**os.environ, "HF_HOME": str(ROOT / "data/hf_cache"),
        "HF_DATASETS_CACHE": str(ROOT / "data/hf_cache/datasets"),
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled"}
    sources = ("scripts/run_fp32_multiepisode.py", "scripts/finetune_smolvla.py",
        "learning/vla/training_precision.py", "learning/vla/training_safety.py",
        "learning/vla/job_monitoring.py",
        "learning/vla/checkpoint_retention.py", "scripts/evaluate_pickup_generalization.py",
        "scripts/report_pickup_generalization.py", "scripts/report_fp32_multiepisode.py")
    archive = {name: archive_source(name) for name in sources}
    gpu_before = gpu_snapshot()
    plan = {"campaign": "corrected-fp32-multiepisode-v1", "started_unix": time.time(),
        "supervisor_pid": os.getpid(), "qualification_plus_training_limit_s": TRAIN_SECONDS,
        "evaluation_limit_s": EVAL_SECONDS, "report_limit_s": REPORT_SECONDS,
        "api_tokens_used": False, "robot_execution": False,
        "warm_start": str(BASE), "base_weights_sha256": sha256(BASE / "model.safetensors"),
        "export_sha256": sha256(EXPORT / "openarm_export.json"), "sources": archive,
        "batch_size": 8, "learning_rate": 1e-5, "steps_ceiling": 20000, "seed": 1000,
        "sampling": "uniform train commands with replacement, masked terminal chunks; all 81 nominal training parents",
        "optimizer": "fresh upstream AdamW, trainable parameters and moments FP32, BF16 forward autocast",
        "disk_reserve_gib": 25, "gpu_before": gpu_before,
        "checkpoint_retention": "Only new campaign: initial, best, latest/recent(1), final. Disposable split qualification bundles removed after exact comparison; reference preserved.",
        "evaluation_selection": "final checkpoint declared a priori, same 5 validation resets and train_20002 control; prefix25/seed1000/1200commands, paired exact expert controls",
        "limits": "Training is not evidence of task success. Existing validation is not an untouched blind test. No insertion/RL promotion."}
    write_json(args.output / "plan.json", plan)
    keeper = subprocess.Popen([sys.executable, str(ROOT / "scripts/keep_training_awake.py"),
        "--pid", str(os.getpid()), "--max-seconds", str(TRAIN_SECONDS + EVAL_SECONDS + REPORT_SECONDS)],
        cwd=ROOT, creationflags=flags) if os.name == "nt" else None
    stages, last_hour, current_child = [], -1, None

    def stage(name, cmd, seconds, heartbeat=None):
        nonlocal last_hour, current_child
        if seconds <= 0:
            raise RuntimeError("Cumulative stage time allowance exhausted")
        event = {"stage": name, "state": "running", "started_unix": time.time(), "command": list(map(str, cmd)), "max_seconds": seconds}
        write_json(args.output / "status.json", event)
        with (args.output / (name + ".log")).open("x", encoding="utf-8") as log:
            process = subprocess.Popen(list(map(str, cmd)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            current_child = process
            fired = threading.Event()
            timer = arm_deadline(process, seconds, fired)
            stage_started = time.monotonic()
            try:
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    stage_elapsed = time.monotonic() - stage_started
                    paths = None
                    status = None
                    if heartbeat:
                        paths = (evaluation_heartbeat_paths(heartbeat) if name == "pickup_evaluation"
                                 else [heartbeat / "events.jsonl", heartbeat / "status.json"])
                        if (heartbeat / "status.json").exists():
                            status = read_json(heartbeat / "status.json")
                    age = heartbeat_age_seconds(paths, stage_elapsed_s=stage_elapsed, now_unix=time.time())
                    gpu = gpu_snapshot()
                    # Thermal and starvation guards fail closed; do not silently retry.
                    try:
                        temp = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=15, creationflags=flags, check=True)
                        temperature = int(temp.stdout.strip().splitlines()[0])
                    except (ValueError, OSError, subprocess.SubprocessError) as error:
                        raise RuntimeError("Cannot verify GPU temperature") from error
                    if temperature >= 85 or psutil.virtual_memory().available < 1024**3:
                        raise RuntimeError("Thermal/RAM safety guard triggered")
                    if shutil.disk_usage(ROOT).free < 25 * 2**30:
                        raise RuntimeError("Protected disk reserve breached")
                    if heartbeat_stalled(age):
                        raise RuntimeError("Training/evaluation heartbeat stalled for 15 minutes")
                    health = {"running": True, "stage": name, "supervisor_pid": os.getpid(), "child_pid": process.pid,
                        "elapsed_s": elapsed, "heartbeat_age_s": age, "gpu": gpu, "temperature_c": temperature,
                        "disk_free_gib": shutil.disk_usage(ROOT).free / 2**30,
                        "available_ram_gib": psutil.virtual_memory().available / 2**30, "training_status": status}
                    write_status(args.output / "health.json", health)
                    hour = int(elapsed // 3600)
                    if hour > last_hour:
                        with (args.output / "hourly_checks.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(health) + "\n")
                        last_hour = hour
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
                code = process.wait()
            finally:
                timer.cancel()
                if process.poll() is None:
                    kill_owned_process(process)
                process.wait()
                current_child = None
        event.update(state="passed" if code == 0 and not fired.is_set() else "failed", returncode=code,
                     timed_out=fired.is_set(), elapsed_s=time.monotonic() - stage_started, ended_unix=time.time())
        stages.append(event)
        write_json(args.output / "stages.json", stages)
        write_json(args.output / "status.json", event)
        if event["state"] != "passed":
            raise RuntimeError(f"{name} failed; no automatic restart")

    def command(script, *extra):
        return [sys.executable, "-u", ROOT / "scripts" / script, *extra]

    def remaining():
        value = train_deadline - time.monotonic()
        if value < 360:
            raise RuntimeError("Insufficient cumulative training allowance for safe completion")
        return value

    try:
        reference, split = args.output / "qualification/reference", args.output / "qualification/split"
        common = ["--export", EXPORT, "--checkpoint", BASE, "--backbone", BACKBONE,
            "--steps", 20000, "--batch-size", 8, "--learning-rate", 1e-5,
            "--eval-every", 500, "--save-every", 20000, "--probes", 4]
        for label, output, count, resume in (("resume_reference", reference, 12, None),
                ("resume_first", split, 8, None),
                ("resume_second", split, 4, split / "checkpoints/step_0000008_final")):
            extra = ["--resume", resume] if resume else []
            stage(label, command("finetune_smolvla.py", *common, "--output", output,
                "--max-seconds", 850, "--stop-after", count, *extra), min(900, remaining()), heartbeat=output)
        qualification = compare_qualification(reference, split)
        report = read_json(reference / "report.json")
        if report["updates"] != 12 or read_json(split / "report.json")["updates"] != 12:
            raise ValueError("Qualification did not complete all expected updates")
        peak_mib = gpu_before["used_mib"] + report["peak_reserved_gb"] * 1e9 / 2**20
        if gpu_before["total_mib"] - peak_mib < 1024:
            raise ValueError("Batch8 lacks a conservative 1 GiB desktop/device headroom")
        qualification.update(batch_size=8, conservative_free_mib=gpu_before["total_mib"] - peak_mib,
            peak_reserved_gb=report["peak_reserved_gb"], elapsed_s=time.monotonic() - started)
        write_json(args.output / "qualification/exact_resume.json", qualification)
        discard_new_split_checkpoints(args.output, split)
        training = args.output / "training"
        allowance = remaining()
        main_args = ["--export", EXPORT, "--checkpoint", BASE, "--backbone", BACKBONE,
            "--output", training, "--steps", 20000, "--batch-size", 8, "--learning-rate", 1e-5,
            "--probes", 64, "--eval-every", 500, "--save-every", 20000,
            "--checkpoint-seconds", 1800, "--initial-checkpoint-step", 25, "--keep-recent-checkpoints", 1,
            "--max-seconds", max(1, allowance - 300)]
        write_json(args.output / "training_gate.json", {"passed": True, "qualification": qualification,
            "training_allowance_s": allowance, "command": list(map(str, command("finetune_smolvla.py", *main_args)))})
        stage("training", command("finetune_smolvla.py", *main_args), allowance, heartbeat=training)
        trained = read_json(training / "report.json")
        if not trained["completed"] or trained["updates"] < 100 or trained["stop_reason"] not in ("step_budget", "wall_time_budget", "low_disk"):
            raise RuntimeError("No sufficiently complete final training checkpoint for evaluation")
        final = Path(trained["final_checkpoint"]) / "pretrained_model"
        evaluation = args.output / "pickup_evaluation"
        stage("pickup_evaluation", command("evaluate_pickup_generalization.py", "--output", evaluation,
            "--checkpoint", final), EVAL_SECONDS + 5, heartbeat=evaluation)
        results = ROOT / "results" / args.output.name
        stage("audit_pickup", command("report_pickup_generalization.py", "--evaluation", evaluation,
            "--output", results, "--verify-only"), 120)
        stage("plot_pickup", [args.plot_python, ROOT / "scripts/report_pickup_generalization.py",
            "--evaluation", evaluation, "--output", results, "--plot-only"], 60)
        stage("plot_training", [args.plot_python, ROOT / "scripts/report_fp32_multiepisode.py",
            "--campaign", args.output, "--output", results], 120)
        write_json(args.output / "completion.json", {"completed": True, "state": "completed",
            "training_seconds_including_qualification": sum(s["elapsed_s"] for s in stages[:4]),
            "total_elapsed_s": time.monotonic() - started, "training_updates": trained["updates"],
            "evaluation": read_json(evaluation / "report.json"), "results": str(results)})
        return 0
    except BaseException as error:
        write_json(args.output / "completion.json", {"completed": False, "state": "failed",
            "error": repr(error), "elapsed_s": time.monotonic() - started,
            "no_automatic_restart": True, "completed_stages": stages})
        raise
    finally:
        if current_child is not None and current_child.poll() is None:
            kill_owned_process(current_child)
        if keeper and keeper.poll() is None:
            keeper.terminate()
            keeper.wait(timeout=15)
        write_status(args.output / "health.json", {"running": False, "supervisor_pid": os.getpid(),
            "completion": read_json(args.output / "completion.json")})


if __name__ == "__main__":
    sys.exit(main())
