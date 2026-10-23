"""Matched original-weight control for a completed repaired-policy prefix trial."""
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
from scripts.run_smolvla_job import arm_deadline, kill_owned_process
from scripts.train_gripper_feedback import BASE, EPISODE, BACKBONE
from scripts.report_precision_repair import audit_rollout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repair", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefix", type=int, choices=(25, 50), required=True)
    args = p.parse_args()
    repair = json.loads((args.repair / "report.json").read_text())
    if not repair["completed"] or str(args.prefix) not in repair["rollouts"]:
        raise ValueError("A completed repaired trial with this exact prefix is required")
    # Include this control within the earlier experiment's one-hour allowance.
    allowance = min(600., 3600. - repair["elapsed_s"] - 120.)
    if allowance < 240:
        raise ValueError("Insufficient allowance for an additional matched control")
    if sha256(BASE / "model.safetensors") != repair["plan"]["base_sha256"]:
        raise ValueError("Baseline weights changed")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", str(ROOT / "scripts/rollout_smolvla.py"),
        "--snapshot", str(ROOT / "third_party/capstone_automateData"), "--episode", str(EPISODE),
        "--checkpoint", str(BASE), "--backbone", str(BACKBONE), "--output", str(args.output / "rollout"),
        "--mode", "policy", "--max-steps", "2000", "--prefix", str(args.prefix), "--seed", "1000",
        "--physical-grasp-metrics"]
    write_json(args.output / "plan.json", {"command": command, "max_seconds": allowance,
        "repair_elapsed_s": repair["elapsed_s"], "purpose": "Same-prefix original-weight control; no training"})
    started = time.monotonic()
    fired = threading.Event()
    with (args.output / "rollout.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        timer = arm_deadline(child, allowance, fired)
        try:
            code = child.wait()
        except BaseException:
            kill_owned_process(child)
            raise
        finally:
            timer.cancel()
    if code or fired.is_set():
        raise RuntimeError(f"Matched control failed: {code}, timeout={fired.is_set()}")
    baseline = json.loads((args.output / "rollout/report.json").read_text())
    trained = json.loads((args.repair / "training/report.json").read_text())
    audit_rollout(baseline, label="original_prefix" + str(args.prefix),
        base_sha=repair["plan"]["base_sha256"], source_sha=trained["binding"]["episode"]["data_sha256"],
        overlay_sha=trained["overlay"]["weights_sha256"], grasp_metrics=repair["expert"]["grasp_metrics"])
    write_json(args.output / "report.json", {"completed": True, "prefix": args.prefix,
        "baseline": baseline, "repaired": repair["rollouts"][str(args.prefix)],
        "repair_report_sha256": sha256(args.repair / "report.json"),
        "elapsed_s": time.monotonic() - started,
        "limits": "One training reset/seed. Combined training/precision repair, not an isolated augmentation comparison."})
    print(json.dumps({"completed": True, "baseline_steps": baseline["steps"],
        "baseline_outcome": baseline["outcome"], "baseline_grasp": baseline["grasp_established"]}), flush=True)


if __name__ == "__main__":
    main()
