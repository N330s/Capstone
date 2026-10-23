"""Independently audit frozen pickup pilot and render reset-level evidence."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.pickup_evaluation import audit_pickup_trace, select_validation_parents, wilson_interval
from learning.vla.source_archive import verified_source
from learning.vla.smolvla_data import sha256
from learning.vla.evaluation_recovery import validate_external_interruption


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--verify-only", action="store_true")
    choice.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    report = json.loads((args.evaluation / "report.json").read_text())
    plan = json.loads((args.evaluation / "plan.json").read_text())
    if not report["completed"] or report["plan"] != plan or len(report["rows"]) != 12:
        raise ValueError("Complete, immutable six-reset paired-expert pilot required")
    if not 0 < report["elapsed_s"] <= 2710:
        raise ValueError("Original cumulative evaluation allowance exceeded")
    interruptions = [json.loads(f.read_text()) for f in sorted(args.evaluation.glob("external_interruption_*.json"))]
    if report.get("external_interruptions", []) != interruptions:
        raise ValueError("External interruption provenance differs")
    for interruption in interruptions:
        prior_s = validate_external_interruption(args.evaluation, interruption)
        if report["elapsed_s"] < prior_s or report["rows"][:len(interruption["completed_rows"])] != interruption["completed_rows"]:
            raise ValueError("Completed outcomes or original time ledger replaced")
    if plan.get("candidate_training"):
        binding = plan["candidate_training"]
        training_report = Path(binding["report"])
        training = json.loads(training_report.read_text())
        final = Path(training["final_checkpoint"])
        if (sha256(training_report) != binding["report_sha256"] or not training["completed"]
                or sha256(final / "complete.json") != binding["complete_sha256"]
                or sha256(final / "pretrained_model/model.safetensors") != plan["base_weights_sha256"]
                or plan["overlay_weights_sha256"] is not None):
            raise ValueError("Full-data candidate training/checkpoint changed")
    for name, expected in plan["implementation"].items():
        verified_source(name, expected)
    continuations = [json.loads(f.read_text()) for f in sorted(args.evaluation.glob("continuation_*.json"))]
    if report.get("continuations", []) != continuations:
        raise ValueError("Continuation provenance differs")
    for continuation in continuations:
        for filename, expected_hash in continuation.get("external_interruption_files", {}).items():
            if Path(filename).name != filename or sha256(args.evaluation / filename) != expected_hash:
                raise ValueError("External interruption binding changed")
        if (continuation["original_plan_sha256"] != sha256(args.evaluation / "plan.json")
                or not continuation["unchanged_actor_and_protocol"]
                or abs(continuation["remaining_allowance_s"] + continuation["previous_elapsed_s"] - 2700) > 1e-8
                or {k: v for k, v in continuation["implementation"].items() if k != "scripts/evaluate_pickup_generalization.py"}
                != {k: v for k, v in plan["implementation"].items() if k != "scripts/evaluate_pickup_generalization.py"}):
            raise ValueError("Continuation changed protocol, actor or cumulative budget")
        for name, expected in continuation["implementation"].items():
            verified_source(name, expected)
    recoveries = [json.loads(f.read_text()) for f in sorted(args.evaluation.glob("artifact_recovery_*.json"))]
    if report.get("artifact_recoveries", []) != recoveries:
        raise ValueError("Artifact recovery provenance differs")
    for recovery in recoveries:
        if (not recovery["no_policy_rerun"] or not recovery["video_rendered_from_saved_qpos"]
                or recovery["original_plan_sha256"] != sha256(args.evaluation / "plan.json")):
            raise ValueError("Unqualified artifact recovery")
        verified_source("scripts/recover_pickup_recording.py", recovery["recovery_script_sha256"])
        recovered_folder = args.evaluation / (recovery["case"] + "_policy")
        for name, expected in recovery["original_file_sha256"].items():
            if sha256(recovered_folder / name) != expected:
                raise ValueError("Original physical execution artifact changed")
    for path in args.evaluation.glob("*_artifact_grace.json"):
        grace = json.loads(path.read_text())
        if (grace["original_plan_sha256"] != sha256(args.evaluation / "plan.json")
                or not grace["controller_execution_finished"] or not grace["no_extra_commands_or_retraining"]
                or not 0 < grace["artifact_only_grace_s"] <= 120):
            raise ValueError("Artifact grace changed physical execution")
    export = ROOT / "data/drive_v2c30_20261012/export_all_v1/openarm_export.json"
    if sha256(export) != plan["export_sha256"]:
        raise ValueError("Export split changed")
    receipt = json.loads(export.read_text())
    selected = select_validation_parents(receipt)
    if ([r["episode"] for r in plan["parents"][1:]] != [r["episode"] for r in selected]
            or [r["episode"] for r in plan["parents"][:1]] != ["train_20002"]):
        raise ValueError("Wrong predeclared parent selection")
    if args.verify_only:
        import av
        parents = {r["episode"]: r for r in plan["parents"]}
        seen = set()
        previews = {}
        for row in report["rows"]:
            key = (row["episode"], row["mode"])
            if key in seen or row["mode"] not in ("policy", "replay"):
                raise ValueError("Duplicate or unknown condition")
            seen.add(key)
            parent = parents[row["episode"]]
            source = ROOT / "data/drive_v2c30_20261012/merged_all_v1" / row["episode"]
            if (sha256(source / "episode.npz") != parent["data_sha256"]
                    or sha256(source / "metadata.json") != parent["metadata_sha256"]):
                raise ValueError("Source changed")
            folder = args.evaluation / row["directory"]
            actual = json.loads((folder / "report.json").read_text())
            if (actual["source_episode"] != row["episode"] or actual["source_data_sha256"] != parent["data_sha256"]
                    or actual["mode"] != row["mode"] or actual["execution_prefix"] != 25 or actual["policy_seed"] != 1000
                    or actual["recorded_input_assistance"] or actual["simulator_state_forced_to_recording"]
                    or actual["reset_state_max_error"] != 0 or actual["checkpoint_weights_sha256"] != plan["base_weights_sha256"]
                    or actual["expert_overlay_weights_sha256"] != (plan["overlay_weights_sha256"] if row["mode"] == "policy" else None)
                    or actual["pickup_evaluation"]["helper_sha256"] != plan["implementation"]["learning/vla/pickup_evaluation.py"]
                    or actual["grasp_metrics"]["extension_sha256"] != plan["implementation"]["learning/vla/physical_grasp.py"]
                    or row["group"] != parent["evaluation_group"]):
                raise ValueError("Condition bindings differ")
            with np.load(folder / "trajectory.npz", allow_pickle=False) as trace:
                metrics = audit_pickup_trace(actual, trace, json.loads((folder / "diagnostics.json").read_text()))
            if any(row[k] != v for k, v in metrics.items()):
                raise ValueError("Physical outcome metrics differ")
            if row["mode"] == "replay" and max(actual["replay_errors"].values()) != 0:
                raise ValueError("Expert replay mismatch")
            with av.open(str(folder / actual.get("video", "rollout.mp4"))) as video:
                frames, last_frame = 0, None
                for frame in video.decode(video=0):
                    frames += 1
                    last_frame = frame
                if row["mode"] == "policy" and last_frame is not None:
                    previews[row["episode"]] = last_frame.to_image()
            if frames != actual["steps"] + 1 or frames != row["decoded_video_frames"]:
                raise ValueError("Video frame audit mismatch")
        heldout = [r for r in report["rows"] if r["group"] == "heldout_validation" and r["mode"] == "policy"]
        count = sum(r["pickup_success"] for r in heldout)
        if count != report["heldout_successes"] or len(heldout) != 5:
            raise ValueError("Wrong heldout denominator")
        args.output.mkdir(parents=True, exist_ok=False)
        for episode, preview in previews.items():
            preview.save(args.output / (episode + "_policy_last.png"))
        summary = {**report, "evaluation_report_sha256": sha256(args.evaluation / "report.json"),
            "independent_motion_and_video_audit_passed": True,
            "reset_level_wilson_95": wilson_interval(count, 5)}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        with (args.output / "conditions.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=report["rows"][0].keys())
            writer.writeheader()
            writer.writerows(report["rows"])
        print(json.dumps({"heldout_successes": count, "heldout_trials": 5,
            "expert_controls_pass": report["all_expert_controls_pass"], "wilson_95": summary["reset_level_wilson_95"]}))
        return
    summary = json.loads((args.output / "summary.json").read_text())
    if summary["evaluation_report_sha256"] != sha256(args.evaluation / "report.json"):
        raise ValueError("Audit before plotting")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in summary["rows"] if r["mode"] == "policy"]
    labels = [r["episode"] + ("\nTRAINING CONTROL" if i == 0 else "\nvalidation") for i, r in enumerate(rows)]
    colors = ["#68737d" if i == 0 else "#22865a" if r["pickup_success"] else "#c45e35" for i, r in enumerate(rows)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.subplots_adjust(top=.79, bottom=.17, left=.09, right=.98, hspace=.28)
    for axis, key, threshold, ylabel in ((axes[0], "maximum_lift_mm", 20., "Maximum actual plug lift (mm)"),
                                        (axes[1], "maximum_continuous_carry_s", 1., "Longest valid carry interval (s)")):
        bars = axis.bar(np.arange(len(rows)), [r[key] for r in rows], color=colors)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        axis.axhline(threshold, color="black", linestyle="--", linewidth=1, label="Required threshold")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, max(threshold, max(r[key] for r in rows)) * 1.3)
        axis.grid(axis="y", alpha=.2)
        axis.legend(loc="upper right", fontsize=8)
    axes[1].set_xticks(np.arange(len(rows)), labels, fontsize=8)
    low, high = summary["reset_level_wilson_95"]
    fig.suptitle(f"Frozen SmolVLA pickup validation pilot: {summary['heldout_successes']}/5 held-out resets\n"
        f"95% reset-level interval {low:.0%}--{high:.0%} | prefix 25 | no retraining\n"
        "Training control excluded from success rate; not an untouched blind test", fontsize=12)
    fig.text(.09, .035, "Success requires grasp + both finger contacts + 20 mm lift + 1 s continuous carry, with no prior drops/abort.\n"
        "Green: pickup passed; orange: failed; gray: memorized control. Insertion is not tested.", fontsize=9)
    fig.savefig(args.output / "pickup_generalization.png", dpi=160)
    fig.savefig(args.output / "pickup_generalization.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
