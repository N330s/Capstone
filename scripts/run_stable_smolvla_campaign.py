"""User-authorized overnight training after explicit I/O and exact-resume qualification.

Preserves previous campaigns; uses a new output root, bounded checkpoint retention
and one cumulative four/eight-hour ledger. No API tokens or hardware commands.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.runtime_io import write_json, read_json
from learning.vla.smolvla_data import sha256, CONTRACT, LEROBOT_VERSION
from scripts.run_smolvla_job import arm_deadline, kill_owned_process
from scripts.resume_smolvla_campaign import remaining_budget, planned_updates
from scripts.benchmark_smolvla_batches import gpu_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous-campaign", "failed-training", "release", "output", "training-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--authorized-total-seconds", type=int, choices=(14400, 28800), default=14400)
    args = parser.parse_args()
    for name in ("previous_campaign", "failed_training", "release", "output", "training_output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.training_output.exists():
        raise ValueError("New training output required; never overwrite previous training")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stages = []
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    env = {**os.environ, "HF_HOME": str(ROOT / "data/hf_cache"),
           "HF_DATASETS_CACHE": str(ROOT / "data/hf_cache/datasets"),
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled"}

    def command(script, *extra):
        return [sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, extra)]

    def stage(name, cmd, timeout):
        event = {"stage": name, "state": "running", "started_unix": time.time(), "command": cmd}
        write_json(args.output / "status.json", event)
        with (args.output / (name + ".log")).open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, creationflags=flags)
            fired = threading.Event()
            timer = arm_deadline(process, timeout, fired)
            try:
                code = process.wait()
            finally:
                timer.cancel()
                if process.poll() is None:
                    kill_owned_process(process)
        event.update(state="passed" if code == 0 and not fired.is_set() else "failed",
                     returncode=code, ended_unix=time.time(), timed_out=fired.is_set())
        stages.append(event)
        write_json(args.output / "stages.json", stages)
        write_json(args.output / "status.json", event)
        if event["state"] != "passed":
            raise RuntimeError(f"{name} failed; no silent training restart")

    try:
        previous = args.previous_campaign
        old_gate = read_json(previous / "preflight_gate.json")
        old_completion = read_json(previous / "completion.json")
        failed_monitor = args.failed_training.with_name(args.failed_training.name + "_monitor")
        failure = read_json(failed_monitor / "completion.json")
        launch = read_json(failed_monitor / "launch.json")
        if (not old_gate["passed"] or old_completion["state"] != "failed"
                or not old_completion["long_run_started"] or failure["reason"] != "trainer_failed"
                or failure["returncode"] == 0):
            raise ValueError("Expected the explicitly authorized, failed previous training campaign")
        import psutil
        if psutil.pid_exists(launch["pid"]):
            raise ValueError("Previous trainer may still be alive; refusing another GPU job")
        old_binding = read_json(args.failed_training / "run_config.json")
        export, merged = args.release / "export_all_v1", args.release / "merged_all_v1"
        base, backbone = ROOT / "data/models/smolvla_base", ROOT / "data/models/smolvlm_backbone"
        receipt = read_json(export / "openarm_export.json")
        qualification_dir = Path(read_json(previous / "reused_qualification.json")["previous_campaign"])
        readback = read_json(qualification_dir / "readback.json")
        if (not readback["passed"] or receipt["contract"] != CONTRACT or receipt["lerobot_version"] != LEROBOT_VERSION
                or readback["source_manifest_sha256"] != sha256(merged / "manifest.json")
                or receipt["source_manifest_sha256"] != readback["source_manifest_sha256"]):
            raise ValueError("Completed release no longer matches passed readback")
        for key, path in (("export_receipt_sha256", export / "openarm_export.json"),
                          ("base_weights_sha256", base / "model.safetensors"),
                          ("base_config_sha256", base / "config.json")):
            if old_binding[key] != sha256(path):
                raise ValueError(f"Previous training input changed: {key}")
        for label in ("base_policy", "preflight_policy"):
            result = read_json(previous / label / "report.json")
            if not result["pipeline_passed"] or result["safety_abort"] or result["steps"] != 100:
                raise ValueError("Previous source-matched inference checks did not pass")
        batch = old_gate["batch_size"]
        benchmark = read_json(qualification_dir / "batch_benchmark/report.json")
        candidate = next(c for c in benchmark["candidates"] if c["batch_size"] == batch and c["safe"])
        gpu = gpu_snapshot()
        headroom = gpu["total_mib"] - gpu["used_mib"] - candidate["peak_reserved_gb"] * 1e9 / 1024**2
        if headroom < 1024:
            raise ValueError("Insufficient current GPU headroom for the qualified batch")
        prior_seconds = old_gate["prior_seconds"] + old_gate["retry_qualification_seconds"] + failure["elapsed_s"]
        write_json(args.output / "authorization.json", {"authorized_total_seconds": args.authorized_total_seconds,
                   "prior_seconds": prior_seconds, "previous_gate": str(previous / "preflight_gate.json"),
                   "failed_training_completion": str(failed_monitor / "completion.json"),
                   "original_trainer_sha256": old_binding["trainer_sha256"],
                   "new_trainer_sha256": sha256(ROOT / "scripts/finetune_smolvla.py"),
                   "compatibility": "New trainer qualified afresh; no mismatched old optimizer state loaded",
                   "batch_size": batch, "export_receipt_sha256": sha256(export / "openarm_export.json")})
        remaining_budget(prior_seconds, time.monotonic() - started, args.authorized_total_seconds)
        # Actual independent GPU processes verify restore, not just a mock test.
        common = ["--export", export, "--checkpoint", base, "--backbone", backbone,
                  "--steps", 24, "--batch-size", batch, "--eval-every", 24, "--save-every", 24, "--probes", 4]
        reference, split = args.output / "resume_reference", args.output / "resume_split"
        for label, output, count, resume in (("resume_reference", reference, 12, None),
                                            ("resume_first", split, 8, None),
                                            ("resume_second", split, 4, split / "checkpoints/step_0000008_final")):
            extra = ["--resume", resume] if resume else []
            stage(label, command("finetune_smolvla.py", *common, "--output", output,
                  "--max-seconds", 600, "--stop-after", count, *extra), timeout=600)
        ref = reference / "checkpoints/step_0000012_final"
        res = split / "checkpoints/step_0000012_final"
        import torch
        from learning.vla.training_state import assert_state_equal
        if sha256(ref / "pretrained_model/model.safetensors") != sha256(res / "pretrained_model/model.safetensors"):
            from safetensors.torch import load_file
            assert_state_equal(load_file(str(ref / "pretrained_model/model.safetensors")),
                               load_file(str(res / "pretrained_model/model.safetensors")), "model")
        a = torch.load(ref / "training_state.pt", map_location="cpu", weights_only=False)
        b = torch.load(res / "training_state.pt", map_location="cpu", weights_only=False)
        for key in ("binding", "step", "optimizer", "scheduler", "rng", "sampler"):
            assert_state_equal(a[key], b[key], key)
        del a, b
        write_json(args.output / "exact_resume.json", {"passed": True, "updates": 12,
                   "split_updates": [8, 4], "compared": ["model", "optimizer", "scheduler", "rng", "sampler", "binding"]})
        budget = remaining_budget(prior_seconds, time.monotonic() - started, args.authorized_total_seconds)
        old_events = [json.loads(line) for line in (qualification_dir / "training_preflight/events.jsonl").read_text().splitlines()]
        steps, median = planned_updates([e["update_seconds"] for e in old_events if e["type"] == "update"], budget)
        training_args = ["--export", export, "--checkpoint", base, "--backbone", backbone,
                         "--steps", steps, "--batch-size", batch, "--eval-every", 1000,
                         "--save-every", steps, "--probes", 64, "--initial-checkpoint-step", 25,
                         "--keep-recent-checkpoints", 2]
        # Verify live monitoring for longer than the previous four-minute failure.
        # A deliberately non-delete-sharing reader reproduces the real WinError5,
        # while normal monitor reads use delete-sharing handles.
        stop, stress = threading.Event(), {"reads": 0, "errors": [], "forced_lock": False}
        def stress_reader():
            path = args.training_output / "status.json"
            while not stop.is_set():
                if path.exists():
                    try:
                        value = read_json(path)
                        stress["reads"] += 1
                        if os.name == "nt" and value["step"] >= 40 and not stress["forced_lock"]:
                            with path.open("r") as locked:
                                locked.read()
                                stress["forced_lock"] = True
                                time.sleep(3.5)
                    except Exception as error:
                        stress["errors"].append(str(error))
                stop.wait(.1)
        reader = threading.Thread(target=stress_reader, daemon=True)
        reader.start()
        qual_monitor = args.output / "stability_monitor"
        try:
            stage("stability_training", command("run_smolvla_job.py", "--output", args.training_output,
                  "--monitor-output", qual_monitor, "--max-seconds", 900,
                  "--authorized-total-seconds", args.authorized_total_seconds, "--", *training_args,
                  "--checkpoint-seconds", 60, "--stop-after", 160), timeout=920)
        finally:
            stop.set()
            reader.join(timeout=5)
        report = read_json(args.training_output / "report.json")
        stress["telemetry_warning_count"] = (qual_monitor / "train.log").read_text(errors="replace").count('"type": "telemetry_warning"')
        write_json(args.output / "io_stress.json", stress)
        before, after = [report[k]["scores"]["validation"]["mean"] for k in ("initial_evaluation", "final_evaluation")]
        if (not report["completed"] or report["updates"] < 120 or not after < before
                or stress["errors"] or (os.name == "nt" and not stress["forced_lock"])):
            raise RuntimeError("Live I/O/training qualification did not pass")
        saved = Path(report["final_checkpoint"])
        if not saved.is_relative_to(args.training_output) or not (saved / "complete.json").is_file():
            raise ValueError("Qualification checkpoint incomplete or outside this training run")
        gpu = gpu_snapshot()
        if gpu["total_mib"] - gpu["used_mib"] - candidate["peak_reserved_gb"] * 1e9 / 1024**2 < 1024:
            raise ValueError("Other GPU usage changed; insufficient headroom for unattended main training")
        elapsed = time.monotonic() - started
        main_budget = remaining_budget(prior_seconds, elapsed, args.authorized_total_seconds)
        write_json(args.output / "preflight_gate.json", {"passed": True,
                   "authorized_total_seconds": args.authorized_total_seconds, "prior_seconds": prior_seconds,
                   "new_qualification_seconds": elapsed, "main_supervisor_seconds": main_budget,
                   "planned_steps": steps, "batch_size": batch, "measured_median_update_seconds": median,
                   "qualification_updates": report["updates"], "validation_before": before, "validation_after": after,
                   "resume_checkpoint": str(saved), "checkpoint_seconds": 900, "keep_recent_checkpoints": 2,
                   "initial_checkpoint_step": 25, "api_tokens_used": False,
                   "limits": "Concurrent I/O and restore qualification, not task success or a bug-free guarantee"})
        stage("bounded_training", command("run_smolvla_job.py", "--output", args.training_output,
              "--max-seconds", main_budget, "--authorized-total-seconds", args.authorized_total_seconds,
              "--", *training_args, "--checkpoint-seconds", 900, "--resume", saved), timeout=main_budget + 10)
        trained = read_json(args.training_output / "report.json")
        manifest = read_json(merged / "manifest.json")
        heldout = next(r for r in manifest["episodes"] if r["split"] == "validation")
        stage("trained_policy", command("rollout_smolvla.py", "--snapshot", ROOT / "third_party/capstone_automateData",
              "--episode", merged / heldout["episode"], "--checkpoint",
              Path(trained["final_checkpoint"]) / "pretrained_model", "--backbone", backbone,
              "--output", args.output / "trained_policy", "--max-steps", 2000, "--prefix", 5), timeout=600)
        write_json(args.output / "completion.json", {"state": "completed", "training_report": str(args.training_output / "report.json"),
                   "limits": "One validation reset rollout, not generalization or sim-to-real qualification"})
    except Exception as error:
        write_json(args.output / "completion.json", {"state": "failed", "error": str(error),
                   "main_run_started": any(s["stage"] == "bounded_training" for s in stages) or
                                       args.training_output.with_name(args.training_output.name + "_monitor").exists()})
        raise


if __name__ == "__main__":
    main()
