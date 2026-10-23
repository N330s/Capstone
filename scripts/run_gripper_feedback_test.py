"""One-hour supervised paired feedback experiment, preserving all existing data."""
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
from learning.vla.runtime_io import write_json
from learning.vla.smolvla_data import sha256
from scripts.run_smolvla_job import arm_deadline, kill_owned_process
from scripts.train_gripper_feedback import BASE, BACKBONE, EPISODE, EXPORT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    if any(args.output.is_relative_to(v) or v.is_relative_to(args.output) for v in (BASE, BACKBONE, EPISODE, EXPORT)):
        raise ValueError("Experiment output overlaps immutable inputs")
    if shutil.disk_usage(ROOT).free < 25 * 2**30 + 2 * 399523968 + 64 * 2**20:
        raise RuntimeError("Need room for both 399.5 MB expert bundles above 25 GiB reserve")
    args.output.mkdir(parents=True, exist_ok=False)
    started, results = time.monotonic(), {}
    plan = {"base_sha256": sha256(BASE / "model.safetensors"), "seed": 1000,
        "steps_per_condition": 800, "total_seconds": 3600, "training_stage_seconds": 1200,
        "files": {name: sha256(ROOT / name) for name in ("scripts/train_gripper_feedback.py",
            "learning/vla/gripper_feedback.py", "learning/vla/expert_overlay.py", "learning/vla/smolvla_policy.py",
            "scripts/rollout_smolvla.py", "scripts/run_gripper_feedback_test.py")}}
    write_json(args.output / "plan.json", plan)
    # Process-lifetime-only sleep prevention, no power-setting or display changes.
    keeper = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/keep_training_awake.py"),
        "--pid", str(os.getpid()), "--max-seconds", "3600"], cwd=ROOT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0) if os.name == "nt" else None

    def stage(name, command, seconds):
        remaining = 3600 - (time.monotonic() - started)
        if remaining < 30:
            raise TimeoutError("One-hour experiment cap")
        print(json.dumps({"started": name}), flush=True)
        fired = threading.Event()
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
            raise RuntimeError(f"{name} failed: {code}, timeout={fired.is_set()}")
        print(json.dumps({"finished": name}), flush=True)

    try:
        for mode in ("nominal", "augmented"):
            train = args.output / mode
            command = [sys.executable, "-u", str(ROOT / "scripts/train_gripper_feedback.py"),
                       "--output", str(train), "--mode", mode]
            stage(mode, command, 1200)
            stage(mode + "_verify", [*command, "--verify"], 240)
            report = json.loads((train / "report.json").read_text())
            if report["updates"] != 800 or report["stop_reason"] != "step_budget":
                raise ValueError("Incomplete fixed-budget condition; do not compare uneven runs")
        schedules = [[json.loads(line)["indices"] for line in
                      (args.output / mode / "updates.jsonl").read_text().splitlines()] for mode in ("nominal", "augmented")]
        if schedules[0] != schedules[1]:
            raise ValueError("Paired training sample schedules differ")
        initial = [json.loads((args.output / mode / "initial.json").read_text()) for mode in ("nominal", "augmented")]
        if initial[0] != initial[1]:
            raise ValueError("Paired runs did not start from identical deterministic predictions")
        for mode in ("nominal", "augmented"):
            dest = args.output / (mode + "_rollout")
            stage(mode + "_rollout", [sys.executable, "-u", str(ROOT / "scripts/rollout_smolvla.py"),
                "--snapshot", str(ROOT / "third_party/capstone_automateData"), "--episode", str(EPISODE),
                "--checkpoint", str(BASE), "--backbone", str(BACKBONE), "--output", str(dest),
                "--expert-overlay", str(args.output / mode / "expert"), "--mode", "policy",
                "--max-steps", "2000", "--prefix", "5", "--seed", "1000"], 600)
            run = json.loads((dest / "report.json").read_text())
            weights = json.loads((args.output / mode / "expert/overlay.json").read_text())["weights_sha256"]
            if (not run["pipeline_passed"] or run["recorded_input_assistance"] or run["simulator_state_forced_to_recording"]
                    or run["expert_overlay_weights_sha256"] != weights or run["checkpoint_weights_sha256"] != plan["base_sha256"]):
                raise ValueError("Live overlay evaluation binding differs")
            results[mode] = run
            write_json(args.output / "progress.json", results)
        if sha256(BASE / "model.safetensors") != plan["base_sha256"]:
            raise ValueError("Original base changed")
        for name, expected in plan["files"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError("Implementation changed during experiment")
        write_json(args.output / "report.json", {"completed": True, "plan": plan,
            "rollouts": results, "elapsed_s": time.monotonic() - started,
            "limits": "One training reset and policy seed, no new-scene, held-out or sim2real success claim."})
    finally:
        if keeper and keeper.poll() is None:
            keeper.terminate()
            keeper.wait(timeout=10)


if __name__ == "__main__":
    main()
