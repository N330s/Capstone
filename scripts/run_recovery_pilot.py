"""Supervise an explicitly bounded offline recovery-training experiment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_smolvla_job import arm_deadline
from learning.vla.runtime_io import write_json
from learning.vla.recovery_data import RECOVERY_CONTRACT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export", "checkpoint", "backbone", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--experiment", action="store_true", help="Allow up to 600 updates / 30 minutes, not an overnight run")
    parser.add_argument("--sampling-index", type=Path)
    parser.add_argument("--view", type=Path)
    parser.add_argument("--sampling-mode", choices=("uniform", "correction"), default="uniform")
    parser.add_argument("--eval-every", type=int, default=8)
    parser.add_argument("--probes", type=int, default=8)
    args = parser.parse_args()
    if (not 180 <= args.seconds <= (1800 if args.experiment else 900)
            or not 1 <= args.steps <= (600 if args.experiment else 64) or not 1 <= args.batch_size <= 16):
        raise ValueError("Bounded qualification: <=64 updates/15min; explicit experiment: <=600 updates/30min")
    if bool(args.sampling_index) != bool(args.view) or (args.sampling_mode == "correction" and not args.sampling_index):
        raise ValueError("Correction sampling requires an index and raw view")
    if min(args.eval_every, args.probes) < 1:
        raise ValueError("Positive evaluation/probe settings required")
    receipt = json.loads((args.export / "openarm_export.json").read_text())
    if receipt.get("contract") != RECOVERY_CONTRACT:
        raise ValueError("Verified recovery-segment export required")
    args.output = args.output.resolve()
    inputs = [args.export, args.checkpoint, args.backbone]
    if args.sampling_index:
        inputs += [args.view, args.sampling_index]
    if any(args.output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(args.output)
           for p in inputs):
        raise ValueError("Pilot outputs must be disjoint from inputs")
    # Two complete best/final copies are ~3 GiB. Keep the intake's 25 GiB reserve.
    if shutil.disk_usage(args.export).free < 29 * 1024**3:
        raise RuntimeError("At least 29 GiB free required before this checkpointed experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", str(ROOT / "scripts/finetune_recovery_smolvla.py"),
               "--export", str(args.export.resolve()), "--checkpoint", str(args.checkpoint.resolve()),
               "--backbone", str(args.backbone.resolve()), "--output", str(args.output / "training"),
               "--steps", str(args.steps), "--batch-size", str(args.batch_size),
               "--max-seconds", str(args.seconds - 60), "--sampling-mode", args.sampling_mode,
               "--eval-every", str(args.eval_every), "--probes", str(args.probes),
               "--save-every", str(args.steps)]
    if args.sampling_index:
        command += ["--sampling-index", str(args.sampling_index.resolve()), "--view", str(args.view.resolve())]
    write_json(args.output / "command.json", {"argv": command, "hard_limit_seconds": args.seconds,
               "scope": "Bounded offline SFT experiment; not task-success or sim2real evidence"})
    fired, started = threading.Event(), time.monotonic()
    with (args.output / "train.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        timer = arm_deadline(process, args.seconds, fired)
        try:
            code = process.wait()
        except BaseException:
            from scripts.run_smolvla_job import kill_owned_process
            kill_owned_process(process)
            raise
        finally:
            timer.cancel()
    report_path = args.output / "training/report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    passed = code == 0 and report.get("completed") and report.get("updates") == args.steps and not fired.is_set()
    result = {"passed": bool(passed), "returncode": code, "hard_deadline_fired": fired.is_set(),
              "elapsed_s": time.monotonic() - started, "training_report": str(report_path),
              "robot_execution": False, "limits": "Finite bounded training, not proof of recovery-task success"}
    write_json(args.output / "pilot_report.json", result)
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError("Recovery pilot did not meet its completion checks; inspect train.log")


if __name__ == "__main__":
    main()
