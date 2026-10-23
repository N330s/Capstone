"""Aggregate completed recovery qualification evidence, without starting a model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.smolvla_data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, default=Path("outputs/recovery_pipeline_20261019"))
    parser.add_argument("--view", type=Path, default=Path("data/recovery_pipeline_20261019/qualification_view"))
    parser.add_argument("--export", type=Path, default=Path("data/recovery_pipeline_20261019/qualification_export"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    verification = json.loads((args.pipeline / "export_verification.json").read_text())
    reload = json.loads((args.pipeline / "checkpoint_verification.json").read_text())
    pilot = json.loads((args.pipeline / "pilot_v1/pilot_report.json").read_text())
    training = json.loads((args.pipeline / "pilot_v1/training/report.json").read_text())
    receipt = json.loads((args.export / "openarm_export.json").read_text())
    command = json.loads((args.pipeline / "pilot_v1/command.json").read_text())["argv"]
    baseline = Path(command[command.index("--checkpoint") + 1])
    if not all(report.get("passed") for report in (verification, reload, pilot)):
        raise ValueError("Export, training and reload qualifications must all pass")
    records = {r["episode"]: r for r in receipt["datasets"]["train"]["episodes"]}
    labels = {}
    for record in records.values():
        parent = record["parent_episode"]
        if parent not in labels:
            with np.load(args.view / parent / "episode.npz", allow_pickle=False) as arrays:
                labels[parent] = arrays["step_label"]
    events = [json.loads(line) for line in (args.pipeline / "pilot_v1/training/events.jsonl").read_text().splitlines()]
    updates = [event for event in events if event["type"] == "update"]
    counts = Counter()
    for event in updates:
        if not np.isfinite([event["loss"], event["grad_norm"]]).all():
            raise ValueError("Non-finite loss or gradient norm")
        for sample in event["samples"]:
            record = records[sample["episode"]]
            counts[str(labels[record["parent_episode"]][record["parent_start"] + sample["step"]])] += 1
    if counts["fault"] or not counts["recovery"]:
        raise ValueError("Pilot must exercise recovery anchors and exclude faults")
    result = {"passed": True, "scope": "Four-parent offline pipeline qualification only",
              "split_plan": str(args.pipeline / "parent_splits.json"),
              "validated_segments": len(verification["segment_provenance"]),
              "eligible_commands": verification["eligible_commands"],
              "excluded_fault_commands": verification["excluded_fault_commands"],
              "optimizer_updates": len(updates), "sampled_training_anchor_labels": dict(counts),
              "all_logged_losses_and_gradient_norms_finite": True,
              "initial_validation_probe_mean": training["initial_evaluation"]["scores"]["validation"]["mean"],
              "final_validation_probe_mean": training["final_evaluation"]["scores"]["validation"]["mean"],
              "validation_probe_count": len(training["binding"]["probes"]["validation"]),
              "validation_parent_count": len(receipt["parent_splits"]["validation"]),
              "elapsed_s": pilot["elapsed_s"], "peak_torch_allocated_gb": training["peak_allocated_gb"],
              "checkpoint_reload_loss_difference": reload["loss_difference"],
              "normalization_unchanged": all(reload["normalization_tensors_unchanged"].values()),
              "final_checkpoint": training["final_checkpoint"],
              "checkpoint_weights_sha256": reload["weights_sha256"],
              "baseline_weights_sha256": training["binding"]["base_weights_sha256"],
              "baseline_weights_still_match": sha256(baseline / "model.safetensors") == training["binding"]["base_weights_sha256"],
              "autonomous_simulation_tested": False, "full_release_validated": False,
              "recovery_optimizer_resume_qualified": False, "robot_execution": False,
              "next_gate": "Complete release + matching new environment source, replay and controlled rollout comparison"}
    if not result["baseline_weights_still_match"]:
        raise ValueError("Baseline weights changed")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
