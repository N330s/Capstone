"""Bounded live-input actuator-resolution diagnostic; no training or oracle inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_smolvla_job import arm_deadline, kill_owned_process
from scripts.train_gripper_feedback import BASE, BACKBONE, EPISODE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolution-mm", type=float, choices=(.25, .5, 1.), default=1.)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fired = threading.Event()
    command = [sys.executable, "-u", str(ROOT / "scripts/rollout_smolvla.py"),
        "--snapshot", str(ROOT / "third_party/capstone_automateData"), "--episode", str(EPISODE),
        "--checkpoint", str(BASE), "--backbone", str(BACKBONE), "--output", str(args.output / "rollout"),
        "--mode", "policy", "--max-steps", "2000", "--prefix", "5", "--seed", "1000",
        "--gripper-resolution-mm", str(args.resolution_mm)]
    with (args.output / "run.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        timer = arm_deadline(child, 600, fired)
        try:
            code = child.wait()
        except BaseException:
            kill_owned_process(child)
            raise
        finally:
            timer.cancel()
    if code or fired.is_set():
        raise RuntimeError("Incomplete filtered rollout")
    print(json.dumps(json.loads((args.output / "rollout/report.json").read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
