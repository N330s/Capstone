"""Frozen-policy, source-matched pickup validation pilot; never trains/tunes."""
import argparse
import csv
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
from learning.vla.source_archive import archive_source
from learning.vla.pickup_evaluation import select_validation_parents, audit_pickup_trace, wilson_interval
from learning.vla.evaluation_recovery import validate_external_interruption
from scripts.train_gripper_feedback import BASE, BACKBONE, EXPORT, EPISODE
from scripts.rollout_smolvla import verify_snapshot
from scripts.run_smolvla_job import arm_deadline, kill_owned_process

REPAIR = ROOT / "outputs/gripper_control_fix_20261019/precision_repair"
COLLECTION = ROOT / "data/drive_v2c30_20261012/merged_all_v1"
SNAPSHOT = ROOT / "third_party/capstone_automateData"
FILES = ("scripts/evaluate_pickup_generalization.py", "scripts/rollout_smolvla.py",
    "learning/vla/pickup_evaluation.py", "learning/vla/physical_grasp.py",
    "learning/vla/smolvla_policy.py", "learning/vla/training_precision.py",
    "learning/vla/expert_overlay.py", "learning/vla/rollout_video.py")


def validate_resume_plan(previous, current, rows):
    """Only the supervisor implementation may change; never policy/protocol."""
    if ({k: v for k, v in previous.items() if k != "implementation"}
            != {k: v for k, v in current.items() if k != "implementation"}):
        raise ValueError("Frozen pickup plan changed")
    supervisor = "scripts/evaluate_pickup_generalization.py"
    if ({k: v for k, v in previous["implementation"].items() if k != supervisor}
            != {k: v for k, v in current["implementation"].items() if k != supervisor}):
        raise ValueError("Actor, physical metric or video implementation changed")
    expected = [(r["episode"], r["evaluation_group"], mode)
                for r in previous["parents"] for mode in ("replay", "policy")]
    if not 0 < len(rows) < len(expected) or [(r["episode"], r["group"], r["mode"]) for r in rows] != expected[:len(rows)]:
        raise ValueError("Completed conditions must be the original ordered prefix")


def terminal_trace_saved(folder):
    """True only after controller execution ended and complete motion was saved."""
    import numpy as np
    try:
        with np.load(folder / "trajectory.npz", allow_pickle=False) as trace:
            n = len(trace["action_requested"])
            return (0 < n <= 1200 and trace["qpos"].shape[0] == n + 1
                and len(json.loads((folder / "diagnostics.json").read_text())) == n + 1
                and sum(c["executed_steps"] for c in json.loads((folder / "chunks.json").read_text())) == n)
    except (OSError, ValueError, KeyError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--checkpoint", type=Path,
        help="Completed FP32 full-data final checkpoint; default is the historical one-parent overlay")
    p.add_argument("--resume", action="store_true", help="Verify and retain completed stages; continue same frozen plan/budget")
    args = p.parse_args()
    receipt = json.loads((EXPORT / "openarm_export.json").read_text())
    candidate, overlay, training_binding = BASE, REPAIR / "training/expert", None
    if args.checkpoint:
        candidate, overlay = args.checkpoint.resolve(), None
        run = candidate.parent.parent.parent
        complete = json.loads((candidate.parent / "complete.json").read_text())
        trained = json.loads((run / "report.json").read_text())
        if (not trained["completed"] or trained["binding"]["trainable_storage"] != "float32"
                or complete["label"] != "final" or complete["binding"] != trained["binding"]
                or Path(trained["final_checkpoint"]).resolve() != candidate.parent
                or trained["binding"]["export_receipt_sha256"] != sha256(EXPORT / "openarm_export.json")):
            raise ValueError("A completed, source-matched FP32 final checkpoint is required")
        training_binding = {"report": str(run / "report.json"), "report_sha256": sha256(run / "report.json"),
            "complete_sha256": sha256(candidate.parent / "complete.json"), "selection": "final checkpoint, declared before training"}
    else:
        trained = json.loads((REPAIR / "training/report.json").read_text())
        if not json.loads((REPAIR / "training/verification.json").read_text())["passed"]:
            raise ValueError("Repaired weights must pass reload first")
    parents = [{**next(r for r in receipt["datasets"]["train"]["episodes"] if r["episode"] == EPISODE.name),
                "evaluation_group": "memorized_positive_control"},
               *[{**r, "evaluation_group": "heldout_validation"} for r in select_validation_parents(receipt)]]
    for record in parents:
        folder = COLLECTION / record["episode"]
        meta = json.loads((folder / "metadata.json").read_text())
        if (sha256(folder / "episode.npz") != record["data_sha256"] or meta["data_sha256"] != record["data_sha256"]
                or meta["seed"] != record["seed"] or meta["outcome"] != "success"):
            raise ValueError("Parent source/seed/outcome differs")
        verify_snapshot(SNAPSHOT, meta)
        record["metadata_sha256"] = sha256(folder / "metadata.json")
        record["reset_options"] = meta["reset_options"]
    plan = {"protocol": "pickup-validation-pilot-v1", "prefix": 25, "policy_seed": 1000,
        "max_commands": 1200, "stage_timeout_s": 300, "total_timeout_s": 2700,
        "selection": "5 evenly spaced parents from lexicographically sorted original validation split; no outcome selection",
        "parents": parents, "base_weights_sha256": sha256(candidate / "model.safetensors"),
        "overlay_weights_sha256": trained["overlay"]["weights_sha256"] if overlay else None,
        "export_sha256": sha256(EXPORT / "openarm_export.json"),
        "implementation": {name: sha256(ROOT / name) for name in FILES},
        "pickup_gate": {"lift_m": .02, "continuous_carry_s": 1., "contacts": 2, "no_prior_drops": True},
        "progression_rule": "5/5 heldout pickup successes and no pre-pickup safety aborts pass the pilot screen only; fresh larger test still required",
        "limits": "Validation influenced original checkpoint selection. Same-scene validation generalization pilot, not an untouched blind test, object/scene OOD, real-time or sim2real evidence. No retraining or hyperparameter sweep."}
    if training_binding:
        plan["candidate_training"] = training_binding
    if overlay and plan["base_weights_sha256"] != trained["binding"]["base_sha256"]:
        raise ValueError("Base checkpoint changed")
    if overlay and sha256(overlay / "expert.safetensors") != plan["overlay_weights_sha256"]:
        raise ValueError("Adapted weights changed")
    runtime_implementation = plan["implementation"]
    args.output = args.output.resolve()
    rows, elapsed_prior, continuations, recoveries, failure_name = [], 0., [], [], "failure.json"
    interruptions = []
    if args.resume:
        if (args.output / "report.json").exists():
            raise ValueError("Pilot already complete; do not rerun outcomes")
        previous = json.loads((args.output / "plan.json").read_text())
        rows = json.loads((args.output / "progress.json").read_text())["rows"]
        validate_resume_plan(previous, plan, rows)
        failures = [args.output / "failure.json", *sorted(args.output.glob("failure_continuation_*.json"))]
        elapsed_values = [json.loads(f.read_text())["elapsed_s"] for f in failures if f.exists()]
        interruptions = [json.loads(f.read_text()) for f in sorted(args.output.glob("external_interruption_*.json"))]
        elapsed_values.extend(validate_external_interruption(args.output, item) for item in interruptions)
        if not elapsed_values:
            raise ValueError("Resume requires an audited prior time ledger")
        elapsed_prior = max(elapsed_values)
        recoveries = [json.loads(f.read_text()) for f in sorted(args.output.glob("artifact_recovery_*.json"))]
        if recoveries:
            elapsed_prior = max(elapsed_prior, max(r["cumulative_elapsed_s"] for r in recoveries))
        continuations = [json.loads(f.read_text()) for f in sorted(args.output.glob("continuation_*.json"))]
        plan = previous
    if elapsed_prior >= 2670:
        raise RuntimeError("Original 45-minute evaluation allowance exhausted")
    if args.prepare_only:
        print(json.dumps({"plan": plan, "resume_completed_stages": len(rows),
            "elapsed_prior_s": elapsed_prior, "remaining_allowance_s": 2700 - elapsed_prior}, indent=2))
        return
    if shutil.disk_usage(ROOT).free < 25.25 * 2**30:
        raise RuntimeError("Retain at least 25 GiB plus artifact headroom")
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
    else:
        sequence = len(continuations) + 1
        continuation = {"sequence": sequence, "original_plan_sha256": sha256(args.output / "plan.json"),
            "previous_elapsed_s": elapsed_prior, "remaining_allowance_s": 2700 - elapsed_prior,
            "completed_stages": len(rows), "implementation": runtime_implementation,
            "artifact_only_grace_s": 120,
            "unchanged_actor_and_protocol": True, "reason": "Resume preserved completed stages after interruption; no retraining or completed outcome replacement",
            "external_interruption_files": {f.name: sha256(f) for f in sorted(args.output.glob("external_interruption_*.json"))}}
        write_json(args.output / f"continuation_{sequence:03}.json", continuation)
        continuations.append(continuation)
        failure_name = f"failure_continuation_{sequence:03}.json"
    for name in FILES:
        archive_source(name)
    started = time.monotonic()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    keeper = subprocess.Popen([sys.executable, str(ROOT / "scripts/keep_training_awake.py"),
        "--pid", str(os.getpid()), "--max-seconds", str(int(2700-elapsed_prior))], cwd=ROOT, creationflags=flags) if os.name == "nt" else None
    completed = {(row["episode"], row["mode"]): row for row in rows}

    def stage(record, mode):
        if (record["episode"], mode) in completed:
            saved = audit_result(record, mode)
            if saved != completed[record["episode"], mode]:
                raise ValueError("Saved completed condition changed")
            print(json.dumps({"verified_preserved": record["episode"] + "_" + mode}), flush=True)
            return saved
        name = record["episode"] + "_" + mode
        if (args.output / name / "report.json").exists():
            # A recorder may finish/recover terminal artifacts after its parent
            # stopped. Audit and admit the original outcome, never rerun policy.
            row = audit_result(record, mode)
            rows.append(row)
            write_json(args.output / "progress.json", {"completed_stages": len(rows), "rows": rows})
            print(json.dumps({"admitted_saved_terminal": name}), flush=True)
            return row
        remaining = 2700 - elapsed_prior - (time.monotonic() - started)
        if remaining < 30:
            raise RuntimeError("Original total time allowance reached; retain partial results")
        if shutil.disk_usage(ROOT).free < 25 * 2**30:
            raise RuntimeError("Disk reserve below 25 GiB; retain partial results")
        command = [sys.executable, "-u", str(ROOT / "scripts/rollout_smolvla.py"),
            "--snapshot", str(SNAPSHOT), "--episode", str(COLLECTION / record["episode"]),
            "--checkpoint", str(candidate), "--backbone", str(BACKBONE), "--output", str(args.output / name),
            "--mode", mode, "--max-steps", "1200", "--prefix", "25", "--seed", "1000",
            "--physical-grasp-metrics", "--pickup-only"]
        if mode == "policy" and overlay:
            command.extend(["--expert-overlay", str(overlay)])
        print(json.dumps({"started": name}), flush=True)
        with (args.output / (name + ".log")).open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            try:
                try:
                    code = child.wait(timeout=min(300, remaining))
                except subprocess.TimeoutExpired:
                    if remaining <= 300 or not terminal_trace_saved(args.output / name):
                        raise
                    grace = min(120., remaining - 300)
                    write_json(args.output / (name + "_artifact_grace.json"), {
                        "original_plan_sha256": sha256(args.output / "plan.json"),
                        "controller_execution_finished": True, "artifact_only_grace_s": grace,
                        "no_extra_commands_or_retraining": True})
                    print(json.dumps({"artifact_only_grace": name, "seconds": grace}), flush=True)
                    code = child.wait(timeout=grace)
            except BaseException:
                kill_owned_process(child)
                raise
        if code:
            raise RuntimeError(f"{name}: code={code}")
        row = audit_result(record, mode)
        rows.append(row)
        write_json(args.output / "progress.json", {"completed_stages": len(rows), "rows": rows})
        print(json.dumps({"finished": name, "pickup_success": row["pickup_success"], "outcome": row["outcome"]}), flush=True)
        return row

    def audit_result(record, mode):
        name = record["episode"] + "_" + mode
        folder = args.output / name
        report = json.loads((folder / "report.json").read_text())
        if (report["recorded_input_assistance"] or report["simulator_state_forced_to_recording"]
                or not report["pipeline_passed"] or report["reset_state_max_error"] != 0
                or report["execution_prefix"] != 25 or report["policy_seed"] != 1000 or report["mode"] != mode
                or report["source_data_sha256"] != record["data_sha256"]
                or report["checkpoint_weights_sha256"] != plan["base_weights_sha256"]
                or report["expert_overlay_weights_sha256"] != (plan["overlay_weights_sha256"] if mode == "policy" else None)
                or report["grasp_metrics"]["extension_sha256"] != plan["implementation"]["learning/vla/physical_grasp.py"]
                or report["pickup_evaluation"]["helper_sha256"] != plan["implementation"]["learning/vla/pickup_evaluation.py"]
                or report["actuation_filter"] != {"gripper_resolution_mm": 0., "reference_or_phase_inputs": False, "arm_targets_changed": False}):
            raise ValueError("Actor/source/reset/metric/filter binding changed")
        import numpy as np
        import av
        with np.load(folder / "trajectory.npz", allow_pickle=False) as trace:
            metrics = audit_pickup_trace(report, trace, json.loads((folder / "diagnostics.json").read_text()))
        if mode == "replay" and max(report["replay_errors"].values()) != 0:
            raise ValueError("Expert physical replay diverged")
        with av.open(str(folder / report.get("video", "rollout.mp4"))) as video:
            frames = sum(1 for _ in video.decode(video=0))
        if frames != report["steps"] + 1:
            raise ValueError("Incomplete physical rollout video")
        row = {"episode": record["episode"], "group": record["evaluation_group"], "mode": mode,
            "steps": report["steps"], "outcome": report["outcome"], "safety_abort": report["safety_abort"],
            **metrics, "decoded_video_frames": frames, "directory": name}
        return row

    try:
        for record in parents:
            expert, policy = stage(record, "replay"), stage(record, "policy")
            if overlay and record["evaluation_group"] == "memorized_positive_control" and not (expert["pickup_success"] and policy["pickup_success"]):
                raise ValueError("New pickup gate failed positive control; do not call heldout failures generalization evidence")
        for name, expected in runtime_implementation.items():
            if sha256(ROOT / name) != expected:
                raise ValueError("Implementation changed during pilot")
        heldout = [r for r in rows if r["group"] == "heldout_validation" and r["mode"] == "policy"]
        expert_ok = all(r["pickup_success"] for r in rows if r["mode"] == "replay")
        controls_ok = all(r["pickup_success"] for r in rows if r["group"] == "memorized_positive_control")
        successes = sum(r["pickup_success"] for r in heldout)
        report = {"completed": True, "plan": plan, "rows": rows, "heldout_successes": successes,
            "heldout_trials": len(heldout), "reset_level_wilson_95": wilson_interval(successes, len(heldout)),
            "all_expert_controls_pass": expert_ok,
            "positive_controls_pass": controls_ok,
            "pilot_screen_passed": expert_ok and controls_ok and successes == 5 and not any(r["safety_abort"] for r in heldout),
            "elapsed_s": elapsed_prior + time.monotonic() - started, "limits": plan["limits"],
            "continuations": continuations, "artifact_recoveries": recoveries, "external_interruptions": interruptions}
        write_json(args.output / "report.json", report)
        with (args.output / "conditions.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps({"completed": True, "heldout_successes": successes, "trials": len(heldout)}), flush=True)
    except BaseException as error:
        write_json(args.output / failure_name, {"completed": False, "error": str(error), "rows": rows,
            "elapsed_s": elapsed_prior + time.monotonic() - started, "no_overall_generalization_result": True})
        raise
    finally:
        if keeper and keeper.poll() is None:
            keeper.terminate()
            keeper.wait(timeout=10)


if __name__ == "__main__":
    main()
