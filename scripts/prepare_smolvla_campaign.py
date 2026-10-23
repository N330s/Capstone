"""Fail-closed preparation and preflight, then a token-free bounded local job.

Requires completed public-download archives. Never retries training after a failed
gate or overwrites partially prepared outputs. All child output is retained.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.finetune_smolvla import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New campaign control directory")
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--wait-for-export", type=Path, help="Avoid competing with an existing export")
    parser.add_argument("--reuse-preparation", type=Path, help="Previous campaign still exporting this release; new control/output required")
    parser.add_argument("--existing-export-pid", type=int, help="Existing exporter preserved during coordinator handover")
    parser.add_argument("--benchmark-export", type=Path, help="Completed export used to qualify batches 4/8/16 before the main run")
    args = parser.parse_args()
    args.release, args.output, args.training_output = [p.resolve() for p in
                                                     (args.release, args.output, args.training_output)]
    args.output.mkdir(parents=True, exist_ok=False)
    stages = []
    env = {**os.environ, "HF_HOME": str(ROOT / "data/hf_cache"),
           "HF_DATASETS_CACHE": str(ROOT / "data/hf_cache/datasets"),
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled"}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    def stage(name, command):
        event = {"stage": name, "state": "running", "started_unix": time.time(), "command": command}
        write_json(args.output / "status.json", event)
        with (args.output / (name + ".log")).open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, creationflags=flags)
        event.update(state="passed" if result.returncode == 0 else "failed",
                     returncode=result.returncode, ended_unix=time.time())
        stages.append(event)
        write_json(args.output / "stages.json", stages)
        write_json(args.output / "status.json", event)
        if result.returncode:
            raise RuntimeError(f"Gate {name} failed; see its log")

    def command(script, *extra, python=None):
        return [str(python or sys.executable), "-u", str(ROOT / "scripts" / script), *map(str, extra)]

    try:
        missing = [n for n in ("a", "b", "c") if not (args.release / "archives" / f"v2c30_{n}.rar").is_file()]
        if missing:
            raise ValueError(f"Download is incomplete: {missing}")
        if args.wait_for_export:
            deadline = time.monotonic() + 3600
            write_json(args.output / "status.json", {"stage": "wait_for_existing_export", "state": "waiting"})
            while not args.wait_for_export.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Existing export did not finish within one hour")
                time.sleep(15)
        merged, export = args.release / "merged_all_v1", args.release / "export_all_v1"
        checkpoint, backbone = ROOT / "data/models/smolvla_base", ROOT / "data/models/smolvlm_backbone"
        benchmark_seconds = 0.
        if args.reuse_preparation:
            previous = args.reuse_preparation.resolve()
            prior_stages = json.loads((previous / "stages.json").read_text())
            for name in ("merge_validate", "stage_camera_audit"):
                event = next(s for s in prior_stages if s["stage"] == name)
                if event["state"] != "passed" or str(merged) not in event["command"]:
                    raise ValueError("Prior preparation does not match this release")
            prior_status = json.loads((previous / "status.json").read_text())
            if prior_status["stage"] != "export" or str(export) not in prior_status["command"]:
                raise ValueError("Handover is only supported at the export stage")
            audit_path = previous / "data_audit/report.json"
            write_json(args.output / "preparation_reuse.json", {"previous_campaign": str(previous),
                       "merged": str(merged), "export": str(export), "export_pid": args.existing_export_pid})
        else:
            sources = []
            for name in ("a", "b", "c"):
                extracted = args.release / f"extracted_{name}"
                if not (extracted / "extraction_receipt.json").is_file():
                    stage(f"extract_{name}", command("extract_collection_rar.py", "--archive",
                          args.release / "archives" / f"v2c30_{name}.rar", "--output", extracted,
                          python=ROOT / ".venv-download/Scripts/python.exe"))
                sources.append(extracted / f"v2c30_{name}")
            stage("merge_validate", command("merge_collection_release.py", "--sources", *sources,
                                           "--output", merged))
            stage("stage_camera_audit", command("inspect_collection_training.py", "--dataset", merged,
                                                "--output", args.output / "data_audit"))
            audit_path = args.output / "data_audit/report.json"
        audit = json.loads(audit_path.read_text())
        if not all(audit["stage_presence_all_episodes"].values()):
            raise RuntimeError("Some episodes lack grasp/lift/insert/hold stage labels")
        if args.benchmark_export:
            stage("batch_benchmark", command("benchmark_smolvla_batches.py", "--export", args.benchmark_export,
                  "--checkpoint", checkpoint, "--backbone", backbone, "--output", args.output / "batch_benchmark"))
            selection = json.loads((args.output / "batch_benchmark/report.json").read_text())
            if not selection["passed"]:
                raise RuntimeError("Batch qualification failed")
            args.batch_size, benchmark_seconds = selection["selected_batch_size"], selection["elapsed_s"]
        if args.reuse_preparation:
            receipt_path = export / "openarm_export.json"
            if not receipt_path.is_file():
                import psutil
                exporter = psutil.Process(args.existing_export_pid)
                if str(export) not in exporter.cmdline() or not any("prepare_smolvla.py" in s for s in exporter.cmdline()):
                    raise ValueError("Export process identity mismatch")
                write_json(args.output / "status.json", {"stage": "wait_for_preserved_export", "state": "running",
                           "export_pid": exporter.pid, "selected_batch_size": args.batch_size})
                deadline = time.monotonic() + 7200
                while not receipt_path.is_file():
                    if not exporter.is_running() or exporter.status() == psutil.STATUS_ZOMBIE:
                        raise RuntimeError("Preserved exporter exited without a completed receipt")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Preserved export did not complete within two hours")
                    time.sleep(5)
        else:
            stage("export", command("prepare_smolvla.py", "--dataset", merged, "--export", export,
                                    "--repo-prefix", "local/openarm_v2c30_release"))
        stage("video_readback", command("verify_smolvla_export.py", "--dataset", merged,
                                        "--export", export, "--report", args.output / "readback.json"))
        if not json.loads((args.output / "readback.json").read_text())["passed"]:
            raise RuntimeError("Video/action readback did not pass")
        qualification_started = time.monotonic()
        preflight = args.output / "training_preflight"
        stage("short_training", command("finetune_smolvla.py", "--export", export, "--checkpoint", checkpoint,
              "--backbone", backbone, "--output", preflight, "--steps", 400, "--max-seconds", 600,
              "--batch-size", args.batch_size, "--eval-every", 200, "--save-every", 400, "--probes", 32))
        report = json.loads((preflight / "report.json").read_text())
        before = report["initial_evaluation"]["scores"]["validation"]["mean"]
        after = report["final_evaluation"]["scores"]["validation"]["mean"]
        if not report["completed"] or report["updates"] < 100 or not after < before:
            raise RuntimeError("Short run did not show held-out probe improvement; long run not started")
        if report["peak_allocated_gb"] > 5.8:
            raise RuntimeError("Too little GPU headroom at the selected batch size")
        saved = Path(report["final_checkpoint"]) / "pretrained_model"
        manifest = json.loads((merged / "manifest.json").read_text())
        heldout = next(r for r in manifest["episodes"] if r["split"] == "validation")
        # Fresh process loads saved processors/model and runs the exact source-matched
        # simulator. A guarded task failure is valid preflight evidence, not success.
        for label, model, mode in (("expert_control", checkpoint, "replay"),
                                   ("base_policy", checkpoint, "policy"),
                                   ("preflight_policy", saved, "policy")):
            adaptation = ["--base-export", export] if label == "base_policy" else []
            stage(label, command("rollout_smolvla.py", "--snapshot", ROOT / "third_party/capstone_automateData",
                  "--episode", merged / heldout["episode"], "--checkpoint", model, "--backbone", backbone,
                  "--output", args.output / label, "--mode", mode, "--max-steps", 100,
                  "--prefix", 5, *adaptation))
        qualification_elapsed = benchmark_seconds + time.monotonic() - qualification_started
        budget = min(13800, int(14400 - qualification_elapsed - 60))
        if budget < 3600:
            raise RuntimeError("Not enough authorized training time remains for the main run")
        write_json(args.output / "preflight_gate.json", {"passed": True, "episodes": len(manifest["episodes"]),
                   "validation_before": before, "validation_after": after,
                   "batch_size": args.batch_size, "qualification_seconds": qualification_elapsed,
                   "batch_benchmark_seconds": benchmark_seconds,
                   "main_supervisor_seconds": budget,
                   "limits": "Probe improvement and guarded interface checks, not autonomous task success."})
        stage("bounded_training", command("run_smolvla_job.py", "--output", args.training_output,
              "--max-seconds", budget, "--", "--export", export, "--checkpoint", checkpoint,
              "--backbone", backbone, "--steps", 60000, "--batch-size", args.batch_size,
              "--eval-every", 1000, "--save-every", 60000, "--probes", 64))
        trained = json.loads((args.training_output / "report.json").read_text())
        # Controlled post-training check. This is simulation inference, not more
        # optimizer updates or permission to command the physical robot.
        stage("trained_policy", command("rollout_smolvla.py", "--snapshot", ROOT / "third_party/capstone_automateData",
              "--episode", merged / heldout["episode"], "--checkpoint",
              Path(trained["final_checkpoint"]) / "pretrained_model", "--backbone", backbone,
              "--output", args.output / "trained_policy", "--mode", "policy", "--max-steps", 2000,
              "--prefix", 5))
        write_json(args.output / "completion.json", {"state": "completed", "training_report": str(args.training_output / "report.json"),
                   "limits": "One held-out reset rollout; no generalization or sim-to-real claim"})
    except Exception as error:
        write_json(args.output / "completion.json", {"state": "failed", "error": str(error),
                   "long_run_started": any(s["stage"] == "bounded_training" for s in stages)})
        raise


if __name__ == "__main__":
    main()
