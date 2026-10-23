"""Explicit restart after a failed inference gate, reusing qualified data/model artifacts.

No export, benchmark or short optimizer run is repeated. Previous outputs remain
unchanged, and prior GPU training time is deducted from the four-hour allowance.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.finetune_smolvla import write_json
from scripts.run_smolvla_job import arm_deadline
from scripts.benchmark_smolvla_batches import gpu_snapshot
from learning.vla.smolvla_data import CONTRACT, LEROBOT_VERSION, sha256


def remaining_budget(prior_seconds, retry_seconds, ceiling=14400):
    if ceiling not in (14400, 28800):
        raise ValueError("Explicit four/eight-hour authorization required")
    if not all(math.isfinite(s) and s >= 0 for s in (prior_seconds, retry_seconds)):
        raise ValueError("Training ledger durations must be finite and nonnegative")
    budget = min(ceiling - 600, math.floor(ceiling - prior_seconds - retry_seconds - 60))
    if budget < 3600:
        raise ValueError("Less than one hour remains in the authorized training allowance")
    return budget


def planned_updates(update_seconds, supervisor_seconds):
    if not update_seconds or any(not math.isfinite(s) or s <= 0 for s in update_seconds):
        raise ValueError("Finite positive measured update durations required")
    median = statistics.median(update_seconds[-100:])
    # Budget imports, fixed-probe evaluations and checkpoint writes before sizing
    # the cosine horizon. The independent supervisor is still the hard limit.
    return max(1000, min(60000, int((supervisor_seconds - 600) / median))), median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous-campaign", "release", "output", "training-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prior-training-report", type=Path, action="append", default=[])
    parser.add_argument("--extra-preflight-seconds", type=float, default=0,
                        help="Conservative debit for manual diagnostic model runs")
    args = parser.parse_args()
    if not math.isfinite(args.extra_preflight_seconds) or args.extra_preflight_seconds < 0:
        raise ValueError("Nonnegative diagnostic debit required")
    args.output = args.output.resolve()
    args.training_output = args.training_output.resolve()
    if args.training_output.exists() or args.training_output.with_name(args.training_output.name + "_monitor").exists():
        raise ValueError("Main training/monitor output already exists; no duplicate training launch")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stages, ledger = [], []
    env = {**os.environ, "HF_HOME": str(ROOT / "data/hf_cache"),
           "HF_DATASETS_CACHE": str(ROOT / "data/hf_cache/datasets"),
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled"}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    def command(script, *extra):
        return [sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, extra)]

    def stage(name, cmd, timeout=None):
        event = {"stage": name, "state": "running", "started_unix": time.time(), "command": cmd}
        write_json(args.output / "status.json", event)
        with (args.output / (name + ".log")).open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, creationflags=flags)
            fired = threading.Event()
            timer = arm_deadline(process, timeout, fired) if timeout else None
            try:
                code = process.wait()
            finally:
                if timer:
                    timer.cancel()
        event.update(state="passed" if code == 0 and not fired.is_set() else "failed",
                     returncode=code, ended_unix=time.time(), timed_out=fired.is_set())
        stages.append(event)
        write_json(args.output / "stages.json", stages)
        write_json(args.output / "status.json", event)
        if event["state"] != "passed":
            raise RuntimeError(f"Gate {name} failed; see its log. Main training is not automatically retried.")

    try:
        previous = args.previous_campaign.resolve()
        completion = json.loads((previous / "completion.json").read_text())
        if completion["state"] != "failed" or completion["long_run_started"]:
            raise ValueError("Only an explicit failed preflight campaign with no main training can be restarted")
        prior_stages = json.loads((previous / "stages.json").read_text())
        for name in ("batch_benchmark", "video_readback", "short_training", "expert_control"):
            if not any(s["stage"] == name and s["state"] == "passed" for s in prior_stages):
                raise ValueError(f"Missing previously passed gate: {name}")
        merged = args.release.resolve() / "merged_all_v1"
        export = args.release.resolve() / "export_all_v1"
        receipt_path = export / "openarm_export.json"
        receipt = json.loads(receipt_path.read_text())
        readback = json.loads((previous / "readback.json").read_text())
        manifest_hash = sha256(merged / "manifest.json")
        if (not readback["passed"] or readback["source_manifest_sha256"] != manifest_hash
                or receipt["source_manifest_sha256"] != manifest_hash
                or receipt["contract"] != CONTRACT or receipt["lerobot_version"] != LEROBOT_VERSION):
            raise ValueError("Passed readback does not bind this completed release")
        report = json.loads((previous / "training_preflight/report.json").read_text())
        benchmark = json.loads((previous / "batch_benchmark/report.json").read_text())
        batch = benchmark["selected_batch_size"]
        if (not benchmark["passed"] or report["binding"]["batch_size"] != batch
                or not any(c["batch_size"] == batch and c["safe"] for c in benchmark["candidates"])):
            raise ValueError("Selected batch does not match the passed training/memory qualification")
        before, after = [report[k]["scores"]["validation"]["mean"]
                         for k in ("initial_evaluation", "final_evaluation")]
        if (not report["completed"] or report["updates"] < 100 or not math.isfinite(after)
                or not after < before or report["peak_allocated_gb"] > 5.8):
            raise ValueError("Prior short training did not satisfy qualification")
        base, backbone = ROOT / "data/models/smolvla_base", ROOT / "data/models/smolvlm_backbone"
        for key, path in (("export_receipt_sha256", receipt_path),
                          ("base_weights_sha256", base / "model.safetensors"),
                          ("base_config_sha256", base / "config.json"),
                          ("trainer_sha256", ROOT / "scripts/finetune_smolvla.py")):
            if report["binding"][key] != sha256(path):
                raise ValueError(f"Qualification artifact changed: {key}")
        saved = Path(report["final_checkpoint"]).resolve()
        if not saved.is_relative_to(previous):
            raise ValueError("Qualification checkpoint is outside the previous campaign")
        marker = json.loads((saved / "complete.json").read_text())
        if marker["binding"] != report["binding"] or marker["step"] != report["updates"]:
            raise ValueError("Incomplete or mismatched qualification checkpoint")
        expert = json.loads((previous / "expert_control/report.json").read_text())
        if not expert["pipeline_passed"] or expert["safety_abort"] or any(expert["replay_errors"].values()):
            raise ValueError("Expert replay did not reproduce the source exactly")
        for s in prior_stages:
            if s["stage"] in ("batch_benchmark", "short_training", "expert_control", "base_policy", "preflight_policy"):
                duration = s["ended_unix"] - s["started_unix"]
                if s["stage"] == "short_training":
                    duration = max(600, duration)  # debit the full original qualification cap
                ledger.append({"source": str(previous / "stages.json"), "stage": s["stage"], "seconds": duration})
        for path in args.prior_training_report:
            old = json.loads(path.read_text())
            if not old["completed"]:
                raise ValueError("Additional training report must be complete")
            ledger.append({"source": str(path.resolve()), "seconds": old["elapsed_s"]})
        ledger.append({"source": "manual diagnostic model runs (conservative allowance)",
                       "seconds": args.extra_preflight_seconds})
        prior_seconds = sum(item["seconds"] for item in ledger)
        remaining_budget(prior_seconds, time.monotonic() - started)
        manifest = json.loads((merged / "manifest.json").read_text())
        heldout = next(r for r in manifest["episodes"] if r["split"] == "validation")
        if expert["source_episode"] != heldout["episode"]:
            raise ValueError("Replay and policy checks must use the same reset")
        write_json(args.output / "reused_qualification.json", {"previous_campaign": str(previous),
                   "export_receipt_sha256": sha256(receipt_path), "batch_size": batch,
                   "preflight_updates": report["updates"], "validation_before": before,
                   "validation_after": after, "checkpoint": str(saved), "prior_budget_ledger": ledger})
        for label, model, extra in (("base_policy", base, ["--base-export", export]),
                                   ("preflight_policy", saved / "pretrained_model", [])):
            stage(label, command("rollout_smolvla.py", "--snapshot", ROOT / "third_party/capstone_automateData",
                  "--episode", merged / heldout["episode"], "--checkpoint", model, "--backbone", backbone,
                  "--output", args.output / label, "--mode", "policy", "--max-steps", 100, "--prefix", 5, *extra),
                  timeout=180)
            inference = json.loads((args.output / label / "report.json").read_text())
            if not inference["pipeline_passed"] or inference["steps"] < 1 or inference["reset_state_max_error"] > 1e-5:
                raise RuntimeError(f"{label} inference-interface check failed")
            if inference["checkpoint_weights_sha256"] != sha256(model / "model.safetensors"):
                raise RuntimeError("Inference checkpoint hash mismatch")
        gpu = gpu_snapshot()
        candidate = next(c for c in benchmark["candidates"] if c["batch_size"] == batch)
        headroom = gpu["total_mib"] - gpu["used_mib"] - candidate["peak_reserved_gb"] * 1e9 / 1024**2
        if headroom < 1024:
            raise RuntimeError("Current desktop/other GPU usage leaves less than 1 GiB conservative headroom")
        retry_seconds = time.monotonic() - started
        budget = remaining_budget(prior_seconds, retry_seconds)
        events = [json.loads(line) for line in (previous / "training_preflight/events.jsonl").read_text().splitlines()]
        steps, median = planned_updates([e["update_seconds"] for e in events if e["type"] == "update"], budget)
        write_json(args.output / "preflight_gate.json", {"passed": True, "batch_size": batch,
                   "prior_budget_ledger": ledger, "prior_seconds": prior_seconds,
                   "retry_qualification_seconds": retry_seconds, "main_supervisor_seconds": budget,
                   "planned_steps": steps, "measured_median_update_seconds": median,
                   "conservative_gpu_headroom_mib": headroom, "source_episodes": len(manifest["episodes"]),
                   "limits": "Reused learning qualification plus fresh inference checks, not task success"})
        stage("bounded_training", command("run_smolvla_job.py", "--output", args.training_output,
              "--max-seconds", budget, "--", "--export", export, "--checkpoint", base,
              "--backbone", backbone, "--steps", steps, "--batch-size", batch,
              "--eval-every", 1000, "--save-every", steps, "--probes", 64))
        trained = json.loads((args.training_output / "report.json").read_text())
        stage("trained_policy", command("rollout_smolvla.py", "--snapshot", ROOT / "third_party/capstone_automateData",
              "--episode", merged / heldout["episode"], "--checkpoint",
              Path(trained["final_checkpoint"]) / "pretrained_model", "--backbone", backbone,
              "--output", args.output / "trained_policy", "--mode", "policy", "--max-steps", 2000, "--prefix", 5),
              timeout=600)
        write_json(args.output / "completion.json", {"state": "completed", "training_report": str(args.training_output / "report.json"),
                   "limits": "One source-matched held-out reset, not generalization or sim-to-real qualification"})
    except Exception as error:
        write_json(args.output / "completion.json", {"state": "failed", "error": str(error),
                   "long_run_started": any(s["stage"] == "bounded_training" for s in stages) or
                                       args.training_output.with_name(args.training_output.name + "_monitor").exists()})
        raise


if __name__ == "__main__":
    main()
