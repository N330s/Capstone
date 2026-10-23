"""Bounded precision repair and declared physical-grasp-metric qualification."""
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
from learning.vla.runtime_io import write_json
from learning.vla.smolvla_data import sha256
from scripts.train_gripper_feedback import BASE, EPISODE, BACKBONE
from scripts.run_smolvla_job import arm_deadline, kill_owned_process


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    files = ("scripts/train_gripper_feedback.py", "learning/vla/training_precision.py",
             "learning/vla/expert_overlay.py", "learning/vla/physical_grasp.py",
             "learning/vla/gripper_feedback.py", "learning/vla/smolvla_policy.py", "scripts/rollout_smolvla.py")
    plan = {"implementation": {name: sha256(ROOT / name) for name in files},
        "base_sha256": sha256(BASE / "model.safetensors"), "total_seconds": 3600,
        "steps": 800, "training_seconds": 1500, "forward": "BF16 autocast", "trainable_storage": "FP32",
        "augmentation": "two-mm open-hold sensor perturbations; no deployed observation assistance",
        "grasp_metrics": "physical-contact-settle-v1; declared bookkeeping extension, unchanged physics/guards"}
    write_json(args.output / "plan.json", plan)
    started = time.monotonic()
    keeper = subprocess.Popen([sys.executable, str(ROOT / "scripts/keep_training_awake.py"),
        "--pid", str(os.getpid()), "--max-seconds", "3600"], cwd=ROOT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0) if os.name == "nt" else None

    def stage(name, command, seconds):
        remaining = 3600 - (time.monotonic() - started)
        if remaining < 30:
            raise TimeoutError("One-hour repair cap")
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
            raise RuntimeError(f"{name} failed: {code}, timeout={fired.is_set()}")
        print(json.dumps({"finished": name}), flush=True)

    def rollout(name, mode, prefix=5, overlay=None):
        command = [sys.executable, "-u", str(ROOT / "scripts/rollout_smolvla.py"),
            "--snapshot", str(ROOT / "third_party/capstone_automateData"), "--episode", str(EPISODE),
            "--checkpoint", str(BASE), "--backbone", str(BACKBONE), "--output", str(args.output / name),
            "--mode", mode, "--max-steps", "2000", "--prefix", str(prefix), "--seed", "1000",
            "--physical-grasp-metrics"]
        if overlay:
            command.extend(["--expert-overlay", str(overlay)])
        stage(name, command, 600)
        report = json.loads((args.output / name / "report.json").read_text())
        if (not report["pipeline_passed"] or report["recorded_input_assistance"] or report["simulator_state_forced_to_recording"]
                or report["grasp_metrics"]["revision"] != "physical-contact-settle-v1"
                or report["grasp_metrics"]["physics_or_guard_thresholds_changed"]
                or report["checkpoint_weights_sha256"] != plan["base_sha256"]):
            raise ValueError("Repair evaluation bindings differ")
        return report

    try:
        expert = rollout("expert_physical_metrics", "replay")
        if not expert["task_success"] or not expert["grasp_established"] or max(expert["replay_errors"].values()) > 1e-8:
            raise ValueError("New bookkeeping failed exact physical expert replay")
        baseline = rollout("baseline_physical_metrics", "policy")
        training = [sys.executable, "-u", str(ROOT / "scripts/train_gripper_feedback.py"),
            "--output", str(args.output / "training"), "--mode", "augmented", "--steps", "800", "--max-seconds", "1500"]
        stage("training", training, 1500)
        stage("verification", [*training, "--verify"], 240)
        trained = json.loads((args.output / "training/report.json").read_text())
        if trained["updates"] != 800 or trained["stop_reason"] != "step_budget":
            raise ValueError("Fixed schedule did not finish")
        rollouts = {}
        for prefix in (5, 25, 50):
            run = rollout("repaired_prefix" + str(prefix), "policy", prefix, args.output / "training/expert")
            if run["expert_overlay_weights_sha256"] != trained["overlay"]["weights_sha256"]:
                raise ValueError("Wrong repaired expert")
            rollouts[str(prefix)] = run
            write_json(args.output / "progress.json", {"rollouts": rollouts})
            if run["task_success"]:
                break
        for name, expected in plan["implementation"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError("Implementation changed during repair test")
        write_json(args.output / "report.json", {"completed": True, "plan": plan,
            "expert": expert, "baseline": baseline, "rollouts": rollouts,
            "elapsed_s": time.monotonic() - started,
            "limits": "One training reset and seed, multiple declared prefix tests; no held-out, generalization or sim2real success claim."})
    finally:
        if keeper and keeper.poll() is None:
            keeper.terminate()
            keeper.wait(timeout=10)


if __name__ == "__main__":
    main()
