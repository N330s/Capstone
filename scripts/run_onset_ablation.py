"""Bounded paired continuation: uniform versus initial-survey exclusion.

No API calls, automatic restarts, old-file cleanup or physical robot commands.
Final checkpoints only; all native actor/physics/contact guards stay unchanged.
"""
import argparse
import gc
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
from learning.vla.runtime_io import read_json, write_json, write_status
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source
from learning.vla.job_monitoring import heartbeat_age_seconds
from scripts.run_smolvla_job import arm_deadline, kill_owned_process

EXPORT = ROOT / "data/drive_v2c30_20261012/export_all_v1"
COLLECTION = ROOT / "data/drive_v2c30_20261012/merged_all_v1"
BASE = ROOT / "outputs/fp32_multiepisode_20261020/training/checkpoints/step_0011300_final/pretrained_model"
BACKBONE = ROOT / "data/models/smolvlm_backbone"
SNAPSHOT = ROOT / "third_party/capstone_automateData"
MODES = ("uniform", "skip-survey")
FILES = ("scripts/run_onset_ablation.py", "scripts/finetune_smolvla.py",
         "learning/vla/onset_sampling.py", "learning/vla/pickup_onset.py",
         "learning/vla/training_precision.py", "learning/vla/smolvla_compat.py",
         "learning/vla/training_safety.py", "scripts/rollout_smolvla.py",
         "learning/vla/physical_grasp.py", "learning/vla/pickup_evaluation.py",
         "learning/vla/smolvla_policy.py", "scripts/evaluate_pickup_generalization.py",
         "scripts/report_onset_ablation.py")


def check_updates(folder, expected, index, mode):
    report = read_json(folder / "report.json")
    updates = [e for e in map(json.loads, (folder / "events.jsonl").read_text().splitlines()) if e["type"] == "update"]
    if (not report["completed"] or report["updates"] != expected or len(updates) != expected
            or report["binding"]["onset_sampling"] != mode
            or report["binding"]["trainable_storage"] != "float32"
            or not all(math.isfinite(e[k]) for e in updates for k in ("loss", "grad_norm", "lr"))):
        raise ValueError("Incomplete, nonfinite or differently bound comparison")
    from learning.vla.window_training import CommandWindows
    receipt = read_json(EXPORT / "openarm_export.json")
    windows = CommandWindows(receipt["datasets"]["train"]["episodes"])
    excluded, counts = set(index["excluded"]), {}
    for event in updates:
        for sample in event["samples"]:
            if sample != windows.locate(sample["index"]) or (mode == "skip-survey" and sample["index"] in excluded):
                raise ValueError("Invalid or excluded training anchor")
            counts[sample["episode"]] = counts.get(sample["episode"], 0)+1
    weights = None
    if report["qualification_only"]:
        if expected > 12 or report["final_checkpoint"] is not None or (folder / "checkpoints").exists():
            raise ValueError("Preflight must retain logs/metrics only, not another model copy")
    else:
        checkpoint = Path(report["final_checkpoint"])
        if not (checkpoint / "complete.json").is_file() or len(list((folder / "checkpoints").iterdir())) != 1:
            raise ValueError("Exactly one complete final-only checkpoint required")
        weights = sha256(checkpoint / "pretrained_model/model.safetensors")
    return {"updates": len(updates), "sampled_parents": len(counts), "parent_counts": counts,
            "final_model_sha256": weights,
            "peak_allocated_gb": report["peak_allocated_gb"], "report_sha256": sha256(folder / "report.json")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plot-python", type=Path, required=True)
    p.add_argument("--supersedes", type=Path, help="Manual diagnostic restart only; charge previous campaign wall time, preserve all artifacts")
    args = p.parse_args()
    out = args.output.resolve()
    if out.parent != ROOT / "outputs" or not out.name.startswith("onset_ablation_"):
        raise ValueError("Dedicated direct outputs/onset_ablation_* directory required")
    if (out / "plan.json").exists() or not (out / "anchor_index.json").is_file():
        raise ValueError("Prepared index and a new, never-launched campaign required")
    if shutil.disk_usage(ROOT).free < 30*2**30:
        raise RuntimeError("30 GiB free required for two final-only bundles, conservative write estimates and 25 GiB reserve")
    prior_s, previous = 0., None
    if args.supersedes:
        previous = args.supersedes.resolve()
        if (previous.parent != ROOT / "outputs" or previous == out
                or not (previous / "manual_stop.json").is_file()
                or not (previous / "failure.json").is_file() or (previous / "completion.json").exists()):
            raise ValueError("Manual restart requires preserved failed launch and explicit stop provenance")
        old = read_json(previous / "plan.json")
        first_started = old.get("aggregate_started_unix", old["started_unix"])
        prior_s = max(0., time.time()-first_started)
        if prior_s > 1200:
            raise ValueError("Restart preparation exceeded 20 minutes; do not extend the original allowance")
    import psutil
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.pid != os.getpid() and proc.info["name"].lower().startswith("python"):
                argv = proc.info["cmdline"] or []
                if any(Path(arg).name in ("finetune_smolvla.py", "rollout_smolvla.py", "run_onset_ablation.py") for arg in argv):
                    # A Windows venv launcher may parent this same supervisor.
                    if proc.pid not in [parent.pid for parent in psutil.Process().parents()]:
                        raise RuntimeError(f"Another training/rollout supervisor exists: {proc.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    from learning.vla.onset_sampling import verify_index
    index = read_json(out / "anchor_index.json")
    verify_index(index, COLLECTION, EXPORT)
    if sha256(BASE / "model.safetensors") != "9a6341162b9df3e8735a1e15d8bbde4acef4e11c79955d9a3b4a80cdfe5316f3":
        raise ValueError("Frozen continuation base changed")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled",
           "HF_HOME": str(ROOT / "data/hf_cache"), "HF_DATASETS_CACHE": str(ROOT / "data/hf_cache/datasets")}
    if previous:
        # Include index/model verification and restart preparation too.
        prior_s = max(0., time.time()-first_started)
    started, stages, rows = time.monotonic(), [], []
    deadline = started+9000-prior_s
    plan = {"protocol": "onset-sampler-ablation-v1", "supervisor_pid": os.getpid(), "started_unix": time.time(),
            "base": str(BASE), "base_weights_sha256": sha256(BASE / "model.safetensors"),
            "index_sha256": sha256(out / "anchor_index.json"), "export_sha256": sha256(EXPORT / "openarm_export.json"),
            "steps_per_arm": 1000, "preflight_steps_per_arm": 12, "batch_size": 8, "learning_rate": 1e-5,
            "seed": 1000, "modes_in_fixed_order": list(MODES), "prefix": 25,
            "training_stage_limit_s": 2400, "preflight_stage_limit_s": 600, "total_limit_s": 9000,
            "aggregate_started_unix": first_started if previous else time.time(), "previous_wall_time_charged_s": prior_s,
            "superseded_attempt": str(previous) if previous else None,
            "superseded_failure_sha256": sha256(previous / "failure.json") if previous else None,
            "restart_reason": "Manual final-only disk-budget correction, with prior artifacts preserved; no training-algorithm or objective change" if previous else None,
            "selection": "Final step 1000 only, same frozen initial weights and fresh AdamW/scheduler; no validation selection",
            "conditioning": "No phase/time/history input; only the eligible training anchor pool changes",
            "checkpoint_retention": "Two full-run final-only bundles. Preflights retain logs/metrics without model copies. No deletion or cleanup.",
            "validation_gate": "Both training controls tested; held-out bank only for arms that pass their control, if sufficient time remains",
            "limits": "One seed, small continuation pilot, not a powered generalization study. Native loss unchanged. Validation previously inspected, not blind test. No API or physical robot use.",
            "sources": {name: archive_source(name) for name in FILES}}
    write_json(out / "plan.json", plan)
    keeper = subprocess.Popen([sys.executable, ROOT / "scripts/keep_training_awake.py", "--pid", str(os.getpid()),
                               "--max-seconds", str(int(9000-prior_s))], cwd=ROOT, creationflags=flags) if os.name == "nt" else None

    def assert_sources():
        for name, record in plan["sources"].items():
            if sha256(ROOT / name) != record["sha256"]:
                raise ValueError("Implementation changed during campaign: "+name)

    def stage(name, command, seconds, heartbeat=None, live_qualification=False):
        assert_sources()
        seconds = min(seconds, deadline-time.monotonic())
        if seconds < 60:
            raise RuntimeError("Overall allowance exhausted; no automatic restart")
        event = {"name": name, "command": list(map(str, command)), "started_unix": time.time(), "hard_limit_s": seconds}
        tick, qualified = time.monotonic(), False
        with (out / (name+".log")).open("x", encoding="utf-8") as log:
            child = subprocess.Popen(event["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            fired = threading.Event()
            timer = arm_deadline(child, seconds, fired)
            try:
                while child.poll() is None:
                    if shutil.disk_usage(ROOT).free < 25*2**30 or psutil.virtual_memory().available < 1024**3:
                        raise RuntimeError("Disk reserve or available RAM safety guard")
                    gpu = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.free", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=15, check=True, creationflags=flags)
                    temp, free = map(int, gpu.stdout.strip().splitlines()[0].split(","))
                    if temp >= 85:
                        raise RuntimeError("GPU thermal safety guard")
                    paths = ([heartbeat / "events.jsonl", heartbeat / "status.json"] if heartbeat else [out / (name+".log")])
                    age = heartbeat_age_seconds(paths, stage_elapsed_s=time.monotonic()-tick, now_unix=time.time())
                    if age > 600:
                        raise RuntimeError("No stage heartbeat for ten minutes")
                    status = None
                    if heartbeat and (heartbeat / "status.json").exists():
                        status = read_json(heartbeat / "status.json")
                        if live_qualification and not qualified and status.get("step", 0) >= 12:
                            updates = [e for e in map(json.loads, (heartbeat / "events.jsonl").read_text().splitlines()) if e["type"] == "update"][:12]
                            if len(updates) != 12 or free < 1024 or not all(math.isfinite(e[k]) for e in updates for k in ("loss", "grad_norm", "lr")):
                                raise RuntimeError("First-12 GPU qualification failed")
                            qualified = True
                            write_json(out / (name+"_live_qualification.json"), {"passed": True, "updates": 12,
                                "temperature_c": temp, "free_gpu_mib": free, "native_loss_unchanged": True})
                    health = {"running": True, "stage": name, "supervisor_pid": os.getpid(), "child_pid": child.pid,
                        "elapsed_s": prior_s+time.monotonic()-started, "remaining_total_s": deadline-time.monotonic(),
                        "temperature_c": temp, "free_gpu_mib": free, "heartbeat_age_s": age,
                        "disk_free_gib": shutil.disk_usage(ROOT).free/2**30, "training_status": status}
                    write_status(out / "health.json", health)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                code = child.wait()
            finally:
                timer.cancel()
                if child.poll() is None:
                    kill_owned_process(child)
                child.wait()
        event.update(returncode=code, elapsed_s=time.monotonic()-tick, timed_out=fired.is_set())
        stages.append(event)
        write_json(out / "stages.json", stages)
        if code or fired.is_set():
            raise RuntimeError(f"Stage failed: {name}; no automatic retry")

    def command(script, *extra):
        return [sys.executable, "-u", ROOT / "scripts" / script, *extra]

    def train(mode, preflight):
        name = ("preflight_" if preflight else "training_")+mode
        folder = out / name
        seconds = 600 if preflight else 2400
        extra = ["--stop-after", "12", "--qualification-only"] if preflight else []
        stage(name, command("finetune_smolvla.py", "--export", EXPORT, "--checkpoint", BASE, "--backbone", BACKBONE,
            "--output", folder, "--steps", "1000", "--batch-size", "8", "--learning-rate", "1e-5", "--seed", "1000",
            "--probes", "8", "--eval-every", "500", "--save-every", "1000", "--max-seconds", str(seconds-60),
            "--onset-index", out / "anchor_index.json", "--onset-collection", COLLECTION, "--onset-sampling", mode,
            "--final-checkpoint-only", *extra), seconds, folder, True)
        audit = check_updates(folder, 12 if preflight else 1000, index, mode)
        write_json(out / (name+"_audit.json"), audit)
        if not preflight:
            a = [e for e in map(json.loads, (out / ("preflight_"+mode) / "events.jsonl").read_text().splitlines()) if e["type"] == "update"]
            b = [e for e in map(json.loads, (folder / "events.jsonl").read_text().splitlines()) if e["type"] == "update"][:12]
            keys = ("step", "loss", "grad_norm", "lr", "samples")
            if [{k:e[k] for k in keys} for e in a] != [{k:e[k] for k in keys} for e in b]:
                raise ValueError("Fresh replay did not reproduce the qualified first 12 updates")
            write_json(out / (name+"_preflight_reproduction.json"), {"passed": True, "compared": list(keys),
                "updates": 12, "not_an_optimizer_resume_test": True})
        return None if preflight else Path(read_json(folder / "report.json")["final_checkpoint"]) / "pretrained_model"

    def control(mode, checkpoint, replay=False):
        from learning.vla.pickup_evaluation import audit_pickup_trace
        import numpy as np
        episode, kind = "train_20002", "replay" if replay else "policy"
        name = "control_"+("expert" if replay else mode)
        folder = out / name
        stage(name, command("rollout_smolvla.py", "--snapshot", SNAPSHOT, "--episode", COLLECTION / episode,
            "--checkpoint", checkpoint, "--backbone", BACKBONE, "--output", folder, "--mode", kind,
            "--max-steps", "1200", "--prefix", "25", "--seed", "1000", "--physical-grasp-metrics", "--pickup-only"), 420)
        r = read_json(folder / "report.json")
        source = read_json(COLLECTION / episode / "metadata.json")
        if (r["recorded_input_assistance"] or r["simulator_state_forced_to_recording"] or not r["pipeline_passed"]
                or r["reset_state_max_error"] != 0 or r["execution_prefix"] != 25 or r["policy_seed"] != 1000
                or r["mode"] != kind or r["source_data_sha256"] != source["data_sha256"]
                or r["checkpoint_weights_sha256"] != sha256(checkpoint / "model.safetensors")
                or r["expert_overlay_weights_sha256"] is not None
                or r["actuation_filter"] != {"gripper_resolution_mm": 0., "reference_or_phase_inputs": False, "arm_targets_changed": False}
                or r["grasp_metrics"]["extension_sha256"] != plan["sources"]["learning/vla/physical_grasp.py"]["sha256"]
                or r["pickup_evaluation"]["helper_sha256"] != plan["sources"]["learning/vla/pickup_evaluation.py"]["sha256"]):
            raise ValueError("Physical control binding differs")
        with np.load(folder / "trajectory.npz", allow_pickle=False) as trace:
            metrics = audit_pickup_trace(r, trace, read_json(folder / "diagnostics.json"))
            qpos = trace["qpos"][:201].copy()
        # Offline FK uses separate MjData after execution, never teleports a rollout.
        from scripts.rollout_smolvla import verify_snapshot
        sys.path.insert(0, str(SNAPSHOT))
        from envs.openarm_insert import OpenArmInsertEnv
        import mujoco
        env_fk = OpenArmInsertEnv(images=False, workspace=verify_snapshot(SNAPSHOT, source))
        try:
            env_fk.reset(seed=source["seed"], options=source["reset_options"])
            independent = mujoco.MjData(env_fk.model)
            with np.load(COLLECTION / episode / "episode.npz", allow_pickle=False) as data:
                close = int(np.flatnonzero(data["phase"] == "close")[0])
                reference_qpos = data["qpos"][:close+1].copy()
            poses = []
            for sequence in (qpos, reference_qpos):
                path = []
                for q in sequence:
                    independent.qpos[:] = q
                    mujoco.mj_forward(env_fk.model, independent)
                    path.append(independent.site_xpos[env_fk.grasp_site].copy())
                poses.append(np.asarray(path))
            tcp, reference = poses
            distance = np.linalg.norm(tcp[:, None]-reference[None], axis=2)
            measurements = [{"command": i, "nearest_expert_path_mm": float(distance[i].min()*1000),
                "nearest_expert_command": int(distance[i].argmin()),
                "same_index_tcp_error_mm": float(np.linalg.norm(tcp[i]-reference[i])*1000)}
                for i in sorted({min(100, len(tcp)-1), len(tcp)-1})]
            np.savez_compressed(folder / "tcp_path.npz", actual_tcp=tcp, expert_preclose_tcp=reference)
        finally:
            env_fk.close()
        if replay and max(r["replay_errors"].values()) != 0:
            raise ValueError("Expert command replay diverged")
        import av
        with av.open(str(folder / "rollout.mp4")) as video:
            frames = sum(1 for _ in video.decode(video=0))
        if frames != r["steps"]+1:
            raise ValueError("Control video incomplete")
        row = {"variant": "expert" if replay else mode, "episode": episode, "directory": str(folder),
               "steps": r["steps"], "outcome": r["outcome"], "safety_abort": r["safety_abort"],
               "early_approach_tcp": measurements,
               "fk_limit": "Offline separate-data FK; Cartesian path position only, not orientation or goal error. Matching-index error includes timing drift.", **metrics}
        rows.append(row)
        write_json(out / "controls.json", rows)
        return row

    try:
        for mode in MODES:
            train(mode, True)
        write_json(out / "qualification.json", {"passed": True, "variants": list(MODES), "updates_each": 12})
        checkpoints = {mode: train(mode, False) for mode in MODES}
        expert = control("expert", BASE, True)
        if not expert["pickup_success"]:
            raise RuntimeError("Expert control failed; do not interpret policy controls")
        policies = {mode: control(mode, checkpoints[mode]) for mode in MODES}
        heldout = {}
        for mode in MODES:
            if not policies[mode]["pickup_success"]:
                heldout[mode] = {"state": "not_run_training_control_failed", "trials": 0}
            elif deadline-time.monotonic() < 2820:
                heldout[mode] = {"state": "not_run_insufficient_total_budget", "trials": 0}
            else:
                destination = out / ("heldout_"+mode)
                stage("heldout_"+mode, command("evaluate_pickup_generalization.py", "--output", destination,
                      "--checkpoint", checkpoints[mode]), 2820)
                r = read_json(destination / "report.json")
                heldout[mode] = {"state": "completed", "successes": r["heldout_successes"],
                    "trials": r["heldout_trials"], "report": str(destination / "report.json"),
                    "note": "Existing evaluator repeats its training control; all attempts retained, no favorable outcome substitution."}
        assert_sources()
        completion = {"completed": True, "training_comparison_completed": True, "controls": rows,
            "heldout": heldout, "elapsed_s": prior_s+time.monotonic()-started, "limits": plan["limits"]}
        write_json(out / "completion.json", completion)
        stage("report", [args.plot_python, ROOT / "scripts/report_onset_ablation.py", "--campaign", out,
              "--output", ROOT / "results" / out.name], 180)
        write_json(out / "report_generation.json", {"completed": True, "elapsed_s": prior_s+time.monotonic()-started,
            "completion_sha256": sha256(out / "completion.json")})
    except BaseException as error:
        write_json(out / "failure.json", {"completed": False, "error": repr(error), "controls": rows,
            "stages": stages, "elapsed_s": prior_s+time.monotonic()-started, "no_automatic_retry": True})
        raise
    finally:
        write_status(out / "health.json", {"running": False, "supervisor_pid": os.getpid(),
            "completed": (out / "completion.json").exists(), "elapsed_s": prior_s+time.monotonic()-started})
        if keeper and keeper.poll() is None:
            keeper.terminate()
            keeper.wait(timeout=10)


if __name__ == "__main__":
    main()
