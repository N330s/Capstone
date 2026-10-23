"""Resume the documented false-stall evaluation, never training or replacing outcomes."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.checkpoint_retention import _no_reparse_points
from learning.vla.evaluation_recovery import charged_external_seconds, validate_external_interruption
from learning.vla.job_monitoring import heartbeat_age_seconds, heartbeat_stalled, evaluation_heartbeat_paths
from learning.vla.runtime_io import read_json, write_json, write_status
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source
from scripts.benchmark_smolvla_batches import gpu_snapshot
from scripts.run_smolvla_job import arm_deadline, kill_owned_process


def preserve_interrupted_attempt(evaluation, condition, archive):
    """Only the next unfinished condition, never a completed physical outcome."""
    folder = evaluation / condition
    if not folder.is_dir() or (folder / "report.json").exists() or (folder / "trajectory.npz").exists():
        raise ValueError("An unfinished terminal trace needs artifact recovery, not a new policy attempt")
    if list(folder.iterdir()):
        raise ValueError("Expected the documented empty model-initialization folder")
    archive.mkdir(parents=True, exist_ok=False)
    for source in (folder, evaluation / (condition + ".log")):
        if source.resolve().parent != evaluation or not source.exists():
            raise ValueError("Interrupted target escaped evaluation")
        _no_reparse_points(source)
        target = archive / source.name
        if target.exists() or not target.resolve().is_relative_to(evaluation):
            raise ValueError("Unsafe interrupted-attempt archive target")
        literal, destination = str(source).replace("'", "''"), str(target).replace("'", "''")
        script = ("$ErrorActionPreference='Stop'; $source=(Resolve-Path -LiteralPath '" + literal + "').Path; "
            "if ($source -ne '" + literal + "') { throw 'Unexpected resolved target' }; "
            "Move-Item -LiteralPath $source -Destination '" + destination + "'")
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--plot-python", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    evaluation = campaign / "pickup_evaluation"
    if campaign.parent != ROOT / "outputs" or campaign.name != "fp32_multiepisode_20261020":
        raise ValueError("This recovery is scoped to the documented interrupted campaign")
    plan, failure, status = (read_json(campaign / name) for name in ("plan.json", "completion.json", "status.json"))
    external_seconds = charged_external_seconds(plan, failure, status)
    trained = read_json(campaign / "training/report.json")
    final = Path(trained["final_checkpoint"]) / "pretrained_model"
    frozen = read_json(evaluation / "plan.json")
    rows = read_json(evaluation / "progress.json")["rows"]
    expected = [(r["episode"], mode) for r in frozen["parents"] for mode in ("replay", "policy")]
    if (not trained["completed"] or not 0 < len(rows) < len(expected)
            or [(r["episode"], r["mode"]) for r in rows] != expected[:len(rows)]
            or sha256(final / "model.safetensors") != frozen["base_weights_sha256"]
            or (evaluation / "report.json").exists()):
        raise ValueError("Completed training and the original preserved partial evaluation are required")
    condition = "_".join(expected[len(rows)])
    if args.prepare_only:
        print(json.dumps({"external_seconds_charged": external_seconds, "remaining_evaluation_s": 2700-external_seconds,
            "preserved_trials": len(rows), "unfinished_condition_to_restart": condition,
            "checkpoint": str(final), "training_restarted": False}, indent=2))
        return
    if not args.plot_python.is_file() or os.name != "nt" or shutil.disk_usage(ROOT).free < 25.25*2**30:
        raise RuntimeError("Windows, plotting interpreter and protected disk headroom required")
    import psutil
    for proc in psutil.process_iter(["cmdline"]):
        try:
            if any(Path(a).name in ("finetune_smolvla.py", "rollout_smolvla.py", "evaluate_pickup_generalization.py")
                    for a in proc.info["cmdline"] or []):
                raise RuntimeError(f"Another GPU training/evaluation process exists: {proc.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    recovery = campaign / "evaluation_recovery_001"
    recovery.mkdir(exist_ok=False)
    started = time.monotonic()
    archive_source("scripts/resume_fp32_evaluation.py")
    archive_source("learning/vla/evaluation_recovery.py")
    archive_source("learning/vla/job_monitoring.py")
    archive_source("scripts/run_fp32_multiepisode.py")
    archive_source("scripts/evaluate_pickup_generalization.py")
    archive_source("scripts/report_pickup_generalization.py")
    shutil.copy2(evaluation / "progress.json", recovery / "original_progress.json")
    preserved = {}
    for row in rows:
        for path in (evaluation / row["directory"]).rglob("*"):
            if path.is_file():
                preserved[str(path.relative_to(evaluation))] = sha256(path)
    attempt = evaluation / "interrupted_attempt_001"
    preserve_interrupted_attempt(evaluation, condition, attempt)
    for path in attempt.rglob("*"):
        if path.is_file():
            preserved[str(path.relative_to(evaluation))] = sha256(path)
    marker = {"original_plan_sha256": sha256(evaluation / "plan.json"),
        "campaign_sources": {key: {"path": str(campaign / filename), "sha256": sha256(campaign / filename)}
            for key, filename in (("plan", "plan.json"), ("completion", "completion.json"), ("status", "status.json"))},
        "external_elapsed_s": external_seconds, "preparation_s": math.ceil(time.monotonic() - started),
        "completed_rows": rows, "preserved_files": preserved, "interrupted_condition": condition,
        "completed_trials_preserved": True, "incomplete_attempt_restart_disclosed": True,
        "interrupted_attempt_archive": attempt.name,
        "recovery_script_sha256": sha256(Path(__file__)),
        "recovery_helper_sha256": sha256(ROOT / "learning/vla/evaluation_recovery.py"),
        "reason": "Original supervisor mistook evaluation duration for stale heartbeat; seven completed conditions retained, unfinished model-loading attempt archived and restarted"}
    marker["elapsed_s"] = external_seconds + marker["preparation_s"]
    write_json(evaluation / "external_interruption_001.json", marker)
    prior = validate_external_interruption(evaluation, marker)
    stages, child, keeper = [], None, None
    report_seconds_spent = 0.
    flags = subprocess.CREATE_NO_WINDOW
    env = {**os.environ, "HF_HOME": str(ROOT / "data/hf_cache"),
        "HF_DATASETS_CACHE": str(ROOT / "data/hf_cache/datasets"),
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled"}
    write_json(recovery / "launch.json", {"pid": os.getpid(), "started_unix": time.time(),
        "original_elapsed_s": prior, "remaining_evaluation_s": 2700-prior,
        "report_limit_s": 300, "checkpoint": str(final), "training_restarted": False,
        "implementation_sha256": sha256(Path(__file__))})

    def stage(name, command, limit, heartbeat=None):
        nonlocal child
        if limit <= 0:
            raise RuntimeError("Recovery allowance exhausted")
        phase_started = time.monotonic()
        event = {"stage": name, "state": "running", "command": list(map(str, command)), "max_seconds": limit}
        write_json(recovery / "status.json", event)
        fired = threading.Event()
        code, error = None, None
        with (recovery / (name + ".log")).open("x", encoding="utf-8") as log:
            child = subprocess.Popen(event["command"], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            timer = arm_deadline(child, limit, fired)
            try:
                while child.poll() is None:
                    age = heartbeat_age_seconds(evaluation_heartbeat_paths(heartbeat) if heartbeat else None,
                        stage_elapsed_s=time.monotonic()-phase_started, now_unix=time.time())
                    if heartbeat_stalled(age):
                        raise RuntimeError("Actual evaluation progress heartbeat is stale")
                    gpu = gpu_snapshot()
                    query = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                        check=True, capture_output=True, text=True, timeout=15, creationflags=flags)
                    temperature = int(query.stdout.strip())
                    if temperature >= 85 or psutil.virtual_memory().available < 2**30 or shutil.disk_usage(ROOT).free < 25*2**30:
                        raise RuntimeError("Thermal/RAM/disk safety guard triggered")
                    write_status(recovery / "health.json", {"running": True, "stage": name,
                        "child_pid": child.pid, "supervisor_pid": os.getpid(), "gpu": gpu, "temperature_c": temperature,
                        "heartbeat_age_s": age, "phase_elapsed_s": time.monotonic()-phase_started,
                        "disk_free_gib": shutil.disk_usage(ROOT).free/2**30})
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                code = child.wait()
            except BaseException as exc:
                error = repr(exc)
                raise
            finally:
                timer.cancel()
                if child.poll() is None:
                    kill_owned_process(child)
                child.wait()
                child = None
                event.update(state="passed" if code == 0 and not fired.is_set() and error is None else "failed",
                    returncode=code, error=error, timed_out=fired.is_set(), elapsed_s=time.monotonic()-phase_started)
                stages.append(event)
                write_json(recovery / "stages.json", stages)
                write_json(recovery / "status.json", event)
        if event["state"] != "passed":
            raise RuntimeError(name + " failed; retained evidence, no automatic retry")
        return event["elapsed_s"]

    results = ROOT / "results" / campaign.name
    try:
        keeper = subprocess.Popen([sys.executable, str(ROOT / "scripts/keep_training_awake.py"),
            "--pid", str(os.getpid()), "--max-seconds", str(math.ceil(2700-prior+300))], cwd=ROOT, creationflags=flags)
        report_seconds_spent += stage("plot_training", [args.plot_python, ROOT / "scripts/report_fp32_multiepisode.py",
            "--campaign", campaign, "--output", results / "training"], 120)
        stage("pickup_evaluation", [sys.executable, "-u", ROOT / "scripts/evaluate_pickup_generalization.py",
            "--output", evaluation, "--checkpoint", final, "--resume"], 2700-prior, heartbeat=evaluation)
        report_seconds_spent += stage("audit_pickup", [sys.executable, ROOT / "scripts/report_pickup_generalization.py",
            "--evaluation", evaluation, "--output", results / "pickup", "--verify-only"], min(120, 300-report_seconds_spent))
        report_seconds_spent += stage("plot_pickup", [args.plot_python, ROOT / "scripts/report_pickup_generalization.py",
            "--evaluation", evaluation, "--output", results / "pickup", "--plot-only"], min(60, 300-report_seconds_spent))
        write_json(recovery / "completion.json", {"completed": True, "state": "completed",
            "training_restarted": False, "elapsed_s": time.monotonic()-started,
            "evaluation": read_json(evaluation / "report.json"), "report_seconds": report_seconds_spent,
            "results": str(results), "original_failure_preserved": True})
        return 0
    except BaseException as error:
        write_json(recovery / "completion.json", {"completed": False, "state": "failed", "error": repr(error),
            "elapsed_s": time.monotonic()-started, "stages": stages, "no_automatic_retry": True})
        raise
    finally:
        if child is not None and child.poll() is None:
            kill_owned_process(child)
        if keeper and keeper.poll() is None:
            keeper.terminate()
            keeper.wait(timeout=15)
        write_status(recovery / "health.json", {"running": False, "completion": read_json(recovery / "completion.json")})


if __name__ == "__main__":
    sys.exit(main())
