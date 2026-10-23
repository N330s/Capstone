"""Bounded source-matched expert replay and privileged-input SmolVLA diagnostics.

Only actor observations are overridden. All physical motion uses normal env.step
and unchanged guards. No training, hardware control, or Hub upload occurs.
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
from scripts.rollout_smolvla import verify_snapshot
from scripts.run_smolvla_job import arm_deadline, kill_owned_process
from learning.vla.runtime_io import write_json
from learning.vla.smolvla_data import sha256


def check_report(report, mode, weights_hash, recording_hash, episode, prefix, seed):
    if (report["mode"] != mode or report["checkpoint_weights_sha256"] != weights_hash
            or report["source_data_sha256"] != recording_hash or report["source_episode"] != episode
            or report["execution_prefix"] != prefix or report["policy_seed"] != seed
            or not report["pipeline_passed"] or report["simulator_state_forced_to_recording"]
            or report["recorded_input_assistance"] != mode.startswith("recorded-")
            or report["autonomous_task_success"]):
        raise ValueError("Diagnostic bindings/assistance flags differ")
    if mode == "replay" and (not report["task_success"] or not report["grasp_established"]
                             or max(report["replay_errors"].values()) > 1e-8):
        raise ValueError("Expert replay control failed; do not interpret policy conditions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "third_party/capstone_automateData")
    parser.add_argument("--episode", type=Path, default=ROOT / "data/drive_v2c30_20261012/merged_all_v1/train_20004")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/v2c30_sft_overnight_v1/checkpoints/step_0009426_best/pretrained_model")
    parser.add_argument("--backbone", type=Path, default=ROOT / "data/models/smolvlm_backbone")
    parser.add_argument("--closed-loop", type=Path, default=ROOT / "outputs/recovery_sampling_20261019/old_scene_transfer/baseline_train_20004")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--stage-seconds", type=int, default=900)
    parser.add_argument("--total-seconds", type=int, default=2400)
    args = parser.parse_args()
    if not (1 <= args.prefix <= 50 and args.seed >= 0 and 60 <= args.stage_seconds <= 1200
            and 180 <= args.total_seconds <= 3600):
        raise ValueError("Invalid bounded diagnostic settings")
    args.output = args.output.resolve()
    for key in ("snapshot", "episode", "checkpoint", "backbone", "closed_loop"):
        value = getattr(args, key).resolve()
        setattr(args, key, value)
        if args.output.is_relative_to(value) or value.is_relative_to(args.output):
            raise ValueError("Output must be disjoint from every input")
    meta = json.loads((args.episode / "metadata.json").read_text())
    if sha256(args.episode / "episode.npz") != meta["data_sha256"]:
        raise ValueError("Recorded source changed")
    verify_snapshot(args.snapshot, meta)
    weights_hash = sha256(args.checkpoint / "model.safetensors")
    closed = json.loads((args.closed_loop / "report.json").read_text())
    closed_manifest = json.loads((args.closed_loop / "manifest.json").read_text())
    if (closed["mode"] != "policy" or closed["checkpoint_weights_sha256"] != weights_hash
            or closed["source_episode"] != args.episode.name or closed["execution_prefix"] != args.prefix
            or closed["policy_seed"] != args.seed or not closed["pipeline_passed"]):
        raise ValueError("Existing closed-loop comparison differs")
    for key in ("scene_sha256", "source_hashes", "robot_config", "contact_config", "action_names", "state_order"):
        if closed_manifest[key] != meta[key]:
            raise ValueError("Existing closed-loop provenance differs: " + key)
    args.output.mkdir(parents=True, exist_ok=False)
    modes = ("replay", "recorded-state", "recorded-observations")
    plan = {"episode": args.episode.name, "source_sha256": meta["data_sha256"],
            "checkpoint": str(args.checkpoint), "checkpoint_weights_sha256": weights_hash,
            "scene_sha256": meta["scene_sha256"], "source_hashes": meta["source_hashes"],
            "modes": modes, "prefix": args.prefix, "seed": args.seed,
            "stage_seconds": args.stage_seconds, "total_seconds": args.total_seconds,
            "closed_loop_baseline": {"path": str(args.closed_loop), "report": closed},
            "implementation_sha256": {name: sha256(ROOT / name) for name in (
                "scripts/rollout_smolvla.py", "scripts/test_recorded_observation_control.py",
                "learning/vla/rollout_video.py", "learning/vla/smolvla_policy.py")},
            "robot_execution": False, "training": False,
            "interpretation": "Privileged input intervention, no simulator teleportation; not autonomous task success"}
    write_json(args.output / "plan.json", plan)
    started, rows = time.monotonic(), []
    for mode in modes:
        remaining = args.total_seconds - (time.monotonic() - started)
        if remaining < 30:
            raise TimeoutError("Total diagnostic allowance exhausted")
        destination = args.output / mode
        command = [sys.executable, "-u", str(ROOT / "scripts/rollout_smolvla.py"),
                   "--snapshot", str(args.snapshot), "--episode", str(args.episode),
                   "--checkpoint", str(args.checkpoint), "--backbone", str(args.backbone),
                   "--output", str(destination), "--max-steps", "2000", "--prefix", str(args.prefix),
                   "--seed", str(args.seed), "--mode", mode]
        fired = threading.Event()
        print(json.dumps({"started": mode}), flush=True)
        with (args.output / (mode + ".log")).open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            timer = arm_deadline(process, min(args.stage_seconds, remaining), fired)
            try:
                code = process.wait()
            except BaseException:
                kill_owned_process(process)
                raise
            finally:
                timer.cancel()
        if code or fired.is_set():
            raise RuntimeError(f"Incomplete diagnostic {mode}: code={code}, timeout={fired.is_set()}")
        report = json.loads((destination / "report.json").read_text())
        check_report(report, mode, weights_hash, meta["data_sha256"], args.episode.name, args.prefix, args.seed)
        rows.append({"mode": mode, "directory": str(destination), "report": report})
        write_json(args.output / "progress.json", {"completed_conditions": len(rows), "conditions": rows})
        print(json.dumps({"finished": mode, "steps": report["steps"], "outcome": report["outcome"],
                          "grasp": report["grasp_established"], "simulated_task_success": report["task_success"]}), flush=True)
    write_json(args.output / "report.json", {**plan, "completed": True, "conditions": rows,
               "elapsed_s": time.monotonic() - started,
               "limits": "One held-out reset, one noise seed, original source-matched scene. Assisted inputs can be inconsistent with actual simulator state; no success generalization or autonomous claim."})


if __name__ == "__main__":
    main()
