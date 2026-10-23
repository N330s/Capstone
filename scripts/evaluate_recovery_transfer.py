"""Paired old-scene rollout comparison; never trains or relaxes provenance guards."""
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


def validate_pair(baseline, candidate):
    baseline, candidate = Path(baseline).resolve(), Path(candidate).resolve()
    marker = json.loads((candidate.parent / "complete.json").read_text())
    if marker["binding"]["base_weights_sha256"] != sha256(baseline / "model.safetensors"):
        raise ValueError("Candidate did not warm-start from this baseline")
    return {"baseline": sha256(baseline / "model.safetensors"),
            "recovery_pilot": sha256(candidate / "model.safetensors")}


def check_rollout(report, expected_hash, episode, prefix, seed):
    if (report["checkpoint_weights_sha256"] != expected_hash or report["source_episode"] != episode
            or report["execution_prefix"] != prefix or report["policy_seed"] != seed
            or report["mode"] != "policy" or not report["pipeline_passed"]):
        raise ValueError("Rollout binding/execution mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/v2c30_sft_overnight_v1/checkpoints/step_0009426_best/pretrained_model")
    parser.add_argument("--candidate", type=Path, default=ROOT / "outputs/recovery_pipeline_20261019/pilot_v1/training/checkpoints/step_0000016_final/pretrained_model")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "third_party/capstone_automateData")
    parser.add_argument("--collection", type=Path, default=ROOT / "data/drive_v2c30_20261012/merged_all_v1")
    parser.add_argument("--backbone", type=Path, default=ROOT / "data/models/smolvlm_backbone")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", default=["train_20004", "train_21010", "train_21036"])
    parser.add_argument("--prefix", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--stage-seconds", type=int, default=600)
    parser.add_argument("--total-seconds", type=int, default=1800)
    args = parser.parse_args()
    if (not 1 <= args.prefix <= 50 or not 1 <= args.max_steps <= 2000
            or not 60 <= args.stage_seconds <= 900 or not 60 <= args.total_seconds <= 3600
            or args.seed < 0 or not 1 <= len(args.episodes) <= 6 or len(set(args.episodes)) != len(args.episodes)):
        raise ValueError("Invalid bounded evaluation settings")
    args.output = args.output.resolve()
    for source in (args.baseline, args.candidate, args.snapshot, args.collection, args.backbone):
        source = source.resolve()
        if args.output.is_relative_to(source) or source.is_relative_to(args.output):
            raise ValueError("Evaluation outputs must be disjoint from inputs")
    hashes = validate_pair(args.baseline, args.candidate)
    metadata = {}
    for name in args.episodes:
        path = (args.collection / name).resolve()
        if path.parent != args.collection.resolve():
            raise ValueError("Unsafe episode path")
        meta = json.loads((path / "metadata.json").read_text())
        if sha256(path / "episode.npz") != meta["data_sha256"]:
            raise ValueError("Source checksum differs")
        verify_snapshot(args.snapshot, meta)
        metadata[name] = {"source_sha256": meta["data_sha256"], "scene_sha256": meta["scene_sha256"],
                          "seed": meta["seed"], "source_hashes": meta["source_hashes"]}
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {"scope": "Cross-release transfer diagnostic: both policies evaluated in the original matched environment",
            "not_new_recovery_scene": True, "checkpoints": {"baseline": str(args.baseline), "recovery_pilot": str(args.candidate)},
            "weights_sha256": hashes, "episodes": metadata, "prefix": args.prefix, "policy_seed": args.seed,
            "max_steps": args.max_steps, "guards": "Unchanged collector guards; source hashes verified",
            "robot_execution": False}
    write_json(args.output / "plan.json", plan)
    started, rows = time.monotonic(), []
    checkpoints = {"baseline": args.baseline, "recovery_pilot": args.candidate}
    for episode in args.episodes:
        for label, checkpoint in checkpoints.items():
            remaining = args.total_seconds - (time.monotonic() - started)
            if remaining < 30:
                raise TimeoutError("Total paired-evaluation budget exhausted")
            destination = args.output / f"{label}_{episode}"
            command = [sys.executable, "-u", str(ROOT / "scripts/rollout_smolvla.py"),
                       "--snapshot", str(args.snapshot.resolve()), "--episode", str((args.collection / episode).resolve()),
                       "--checkpoint", str(checkpoint.resolve()), "--backbone", str(args.backbone.resolve()),
                       "--output", str(destination), "--max-steps", str(args.max_steps),
                       "--prefix", str(args.prefix), "--seed", str(args.seed)]
            fired = threading.Event()
            print(json.dumps({"started": label, "episode": episode}), flush=True)
            with (args.output / f"{label}_{episode}.log").open("x", encoding="utf-8") as log:
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
                raise RuntimeError(f"Rollout incomplete: {label}, {episode}, code={code}, timeout={fired.is_set()}")
            report = json.loads((destination / "report.json").read_text())
            check_rollout(report, hashes[label], episode, args.prefix, args.seed)
            rows.append({"model": label, "episode": episode, "directory": str(destination),
                         "report": report})
            write_json(args.output / "progress.json", {"completed_rollouts": len(rows), "rollouts": rows})
            print(json.dumps({"finished": label, "episode": episode, "steps": report["steps"],
                              "outcome": report["outcome"], "grasp": report["grasp_established"]}), flush=True)
    write_json(args.output / "report.json", {**plan, "completed": True, "rollouts": rows,
               "elapsed_s": time.monotonic() - started,
               "candidate_updates": json.loads((args.candidate.resolve().parent / "complete.json").read_text())["step"],
               "limits": "Selected old validation resets, one policy seed, bounded recovery candidate. Not new-scene validation or generalization proof."})


if __name__ == "__main__":
    main()
