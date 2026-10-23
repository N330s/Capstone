"""Watch a single local trainer, enforce an authorized deadline, and record hourly health.

This supervisor uses no AI/API tokens. Its deadline covers the child process's
imports, evaluation and checkpoint writes, not just optimizer updates.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.finetune_smolvla import write_json
from learning.vla.runtime_io import read_json, write_status


def kill_owned_process(process):
    # Windows venv executables can be launchers with a real Python child. Killing
    # just the launcher would leave optimizer updates running past the deadline.
    if hasattr(process, "pid"):
        import psutil
        try:
            descendants = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            descendants = []
        for child in reversed(descendants):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    if process.poll() is None:
        process.kill()


def arm_deadline(process, seconds, fired):
    def terminate_at_deadline():
        if process.poll() is None:
            fired.set()
            kill_owned_process(process)
    timer = threading.Timer(max(0, seconds), terminate_at_deadline)
    timer.daemon = True
    timer.start()
    return timer


def watch_training(process, args, monitor_dir, started, flags):
    last_hour = -1
    while process.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed >= args.max_seconds:
            kill_owned_process(process)
            return "hard_time_limit"
        status_file = args.output / "status.json"
        status = None
        try:
            status = read_json(status_file)
        except (OSError, ValueError):
            pass
        mtimes = []
        for path in (status_file, args.output / "events.jsonl"):
            try:
                mtimes.append(path.stat().st_mtime)
            except OSError:
                pass
        heartbeat_age = time.time() - max(mtimes) if mtimes else elapsed
        health = {"elapsed_s": elapsed, "pid": process.pid, "running": True,
                  "heartbeat_age_s": heartbeat_age, "training_status": status,
                  "warning": "No heartbeat for >10 minutes" if heartbeat_age > 600 else None}
        write_status(monitor_dir / "health.json", health)
        if heartbeat_age > args.heartbeat_timeout_seconds:
            kill_owned_process(process)
            return "stalled_heartbeat"
        hour = int(elapsed // 3600)
        if hour > last_hour:
            try:
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                                      "--format=csv,noheader"], capture_output=True, text=True, timeout=15,
                                     creationflags=flags)
                health["gpu_snapshot"] = gpu.stdout.strip()
            except (OSError, subprocess.TimeoutExpired) as error:
                health["gpu_snapshot_error"] = str(error)
            try:
                with (monitor_dir / "hourly_checks.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(health) + "\n")
            except PermissionError as error:
                print(f"Hourly telemetry lock: {error}", flush=True)
            last_hour = hour
        time.sleep(min(30, max(.1, args.max_seconds - elapsed)))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=14400)
    parser.add_argument("--authorized-total-seconds", type=int, choices=(14400, 28800), default=14400)
    parser.add_argument("--monitor-output", type=Path, help="New monitor directory for a checkpoint-resume session")
    parser.add_argument("--heartbeat-timeout-seconds", type=float, default=900)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= args.authorized_total_seconds or not args.heartbeat_timeout_seconds > 0:
        raise ValueError("Supervisor duration must be within the authorized four/eight-hour ceiling")
    extra = args.trainer_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    if any(a.startswith(("--output", "--max-seconds", "--runtime-ceiling-seconds")) for a in extra):
        raise ValueError("Supervisor owns output and deadline arguments")
    monitor_dir = args.monitor_output.resolve() if args.monitor_output else args.output.resolve().with_name(args.output.name + "_monitor")
    if monitor_dir == args.output.resolve() or monitor_dir.is_relative_to(args.output.resolve()) or args.output.resolve().is_relative_to(monitor_dir):
        raise ValueError("Monitor and training directories must be disjoint")
    monitor_dir.mkdir(parents=True, exist_ok=False)
    prior_elapsed = 0.
    if "--resume" in extra:
        checkpoint = Path(extra[extra.index("--resume") + 1]).resolve()
        if not checkpoint.is_relative_to(args.output.resolve()):
            raise ValueError("Resume checkpoint must belong to this training output")
        prior_elapsed = read_json(checkpoint / "complete.json")["elapsed_s"]
        if not 0 <= prior_elapsed < args.authorized_total_seconds:
            raise ValueError("Invalid cumulative resume time")
    trainer_budget = max(1, args.max_seconds - 300) + prior_elapsed
    if trainer_budget > args.authorized_total_seconds:
        raise ValueError("Cumulative trainer time exceeds the authorized ceiling")
    if prior_elapsed:
        report_path = args.output / "report.json"
        if report_path.exists():
            write_json(monitor_dir / "previous_session_report.json", read_json(report_path))
        write_json(report_path, {"completed": False, "state": "resume_session_starting",
                   "prior_checkpoint_elapsed_s": prior_elapsed,
                   "previous_session_report": str(monitor_dir / "previous_session_report.json")})
    started = time.monotonic()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    command = [sys.executable, "-u", str(ROOT / "scripts/finetune_smolvla.py"),
               "--output", str(args.output.resolve()), "--max-seconds", str(trainer_budget),
               "--runtime-ceiling-seconds", str(args.authorized_total_seconds), *extra]
    with (monitor_dir / "train.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=flags)
        fired = threading.Event()
        timer = arm_deadline(process, args.max_seconds - (time.monotonic() - started), fired)
        reason, monitor_error = None, None
        try:
            write_json(monitor_dir / "launch.json", {"pid": process.pid, "supervisor_pid": os.getpid(),
                       "command": command, "max_seconds": args.max_seconds,
                       "prior_checkpoint_elapsed_s": prior_elapsed,
                       "authorized_total_seconds": args.authorized_total_seconds,
                       "started_unix": time.time(), "api_tokens_used": False})
            reason = watch_training(process, args, monitor_dir, started, flags)
        except Exception as error:
            reason, monitor_error = "monitor_failed", str(error)
        finally:
            if process.poll() is None:
                kill_owned_process(process)
            code = process.wait()
            timer.cancel()
        if fired.is_set():
            reason = "hard_time_limit"
    report_complete = False
    try:
        report_complete = bool(read_json(args.output / "report.json").get("completed"))
    except (OSError, ValueError):
        pass
    completion = {"returncode": code, "elapsed_s": time.monotonic() - started,
                  "reason": reason or ("trainer_complete" if code == 0 else "trainer_failed"),
                  "monitor_error": monitor_error,
                  "report_exists": (args.output / "report.json").exists(), "report_complete": report_complete}
    write_json(monitor_dir / "completion.json", completion)
    write_status(monitor_dir / "health.json", {**completion, "pid": process.pid, "running": False})
    return code


if __name__ == "__main__":
    sys.exit(main())
