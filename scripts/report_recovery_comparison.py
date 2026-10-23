"""Compare equal-budget recovery SFT experiments; offline metrics are not success."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def behavior_summary(report):
    result = dict(report["summary"])
    # Same >0.2 mm criterion as the sampler, including regrasp-phase opening.
    rows = [r for r in report.get("predictions", []) if r["group"] == "new_recovery"
            and r["expert_opening_delta_mm"] > .2]
    if rows:
        def metrics(values):
            return {"predictions": len(values), **{key: fmean(r[key] for r in values) for key in (
                "first_gripper_mae_mm", "prefix5_gripper_mae_mm", "chunk_gripper_mae_mm",
                "predicted_opening_delta_mm", "expert_opening_delta_mm")},
                "positive_opening_prediction_fraction": fmean(r["predicted_opening_delta_mm"] > 0 for r in values)}
        result["new_active_reopening"] = metrics(rows)
        result["active_reopening_by_parent"] = {parent: metrics([r for r in rows if r["episode"] == parent])
                                               for parent in sorted({r["episode"] for r in rows})}
    return result


def compare(uniform, correction):
    roots = {"uniform": Path(uniform), "correction": Path(correction)}
    reports = {k: read(p / "training/report.json") for k, p in roots.items()}
    behaviors = {k: read(p / "behavior/report.json") for k, p in roots.items()}
    for key in ("export_receipt_sha256", "base_weights_sha256", "base_config_sha256",
                "steps", "batch_size", "learning_rate", "seed", "probes",
                "sampling_index_sha256", "trainer_sha256", "sampling_implementation_sha256"):
        if reports["uniform"]["binding"][key] != reports["correction"]["binding"][key]:
            raise ValueError("Unequal experiment binding: " + key)
    for key in ("sources", "cases", "batch_size", "all_heldout_recovery_frames", "heldout_recovery_parent_count"):
        if behaviors["uniform"][key] != behaviors["correction"][key]:
            raise ValueError("Unequal recorded-observation probes: " + key)
    result = {}
    for name, root in roots.items():
        report = reports[name]
        supervisor = read(root / "pilot_report.json")
        audit = read(root / "sampling_audit.json")
        if (report["updates"] != report["binding"]["steps"]
                or audit["counts"]["total_windows"] != report["updates"] * report["binding"]["batch_size"]
                or not report["binding"]["sampling"].startswith(name + ";")):
            raise ValueError("Budget/exposure/sampling mode mismatch")
        if not supervisor["passed"] or not report["completed"] or not read(root / "checkpoint_verification.json")["passed"]:
            raise ValueError("Experiment/reload qualification failed: " + name)
        if audit["counts"]["fault_targets"]:
            raise ValueError("Fault target exposure")
        behavior = behaviors[name]
        model = Path(report["final_checkpoint"]) / "pretrained_model/model.safetensors"
        with model.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if behavior["weights_sha256"] != digest:
            raise ValueError("Behavior probe used another checkpoint")
        result[name] = {"updates": report["updates"], "elapsed_s": report["elapsed_s"],
            "batch_size": report["binding"]["batch_size"], "sampling": report["binding"]["sampling"],
            "initial": report["initial_evaluation"]["scores"], "final": report["final_evaluation"]["scores"],
            "exposure": audit["counts"], "behavior": behavior_summary(behavior),
            "state_sensitivity": behavior["state_sensitivity"], "checkpoint": str(model.parent),
            "weights_sha256": digest, "peak_allocated_gb": report["peak_allocated_gb"],
            "evaluations": report["evaluations"]}
    if result["correction"]["exposure"]["active_reopening_anchors"] * 4 < result["correction"]["exposure"]["total_windows"]:
        raise ValueError("Correction training did not meet its exposure quota")
    return {"experiments": result, "matched_fields_checked": True,
            "limits": "One seed, partial training release, fixed sampled validation probes. Recorded-observation improvement is not closed-loop grasp/insertion success. New-scene simulator source remains required."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("uniform", "correction", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-behavior", type=Path)
    parser.add_argument("--transfer", type=Path, help="Paired old-scene diagnostic report, not new-scene evaluation")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.uniform, args.correction)
    if args.baseline_behavior:
        baseline, uniform = read(args.baseline_behavior), read(args.uniform / "behavior/report.json")
        for key in ("sources", "cases", "batch_size", "all_heldout_recovery_frames", "heldout_recovery_parent_count"):
            if baseline[key] != uniform[key]:
                raise ValueError("Baseline probe mismatch: " + key)
        binding = read(args.uniform / "training/report.json")["binding"]
        if baseline["weights_sha256"] != binding["base_weights_sha256"]:
            raise ValueError("Baseline probe used different warm-start weights")
        result["baseline"] = {"weights_sha256": baseline["weights_sha256"],
                              "behavior": behavior_summary(baseline), "state_sensitivity": baseline["state_sensitivity"]}
    if args.transfer:
        transfer = read(args.transfer)
        if (not transfer.get("completed") or not transfer.get("not_new_recovery_scene")
                or transfer["weights_sha256"]["recovery_pilot"] != result["experiments"]["correction"]["weights_sha256"]):
            raise ValueError("Transfer checkpoint/scope mismatch")
        result["old_scene_transfer"] = {"not_new_recovery_scene": True,
            "candidate_updates": transfer["candidate_updates"],
            "rollouts": [{"model": row["model"], "episode": row["episode"],
                "video": str(Path(row["directory"]) / "rollout.mp4"),
                **{key: row["report"][key] for key in ("steps", "outcome", "grasp_established", "task_success")}}
                for row in transfer["rollouts"]]}
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = []
    for name, run in result["experiments"].items():
        for event in run["evaluations"]:
            for group, metrics in event["scores"].items():
                rows.append({"experiment": name, "step": event["step"], "group": group,
                             **{k: metrics[k] for k in ("mean", "gripper_mae_mm", "first_gripper_mae_mm",
                                                        "prefix5_gripper_mae_mm", "predicted_opening_delta_mm")}})
    with (args.output / "validation_metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.subplots_adjust(left=.065, right=.985, bottom=.14, top=.80, wspace=.30)
    colors = {"uniform": "#d9822b", "correction": "#2471a3"}
    for name, run in result["experiments"].items():
        events = run["evaluations"]
        axes[0].plot([e["step"] for e in events], [e["scores"]["validation"]["mean"] for e in events],
                     label=name, color=colors[name], marker=".")
        axes[1].plot([e["step"] for e in events],
                     [e["scores"]["validation_active_reopening"]["first_gripper_mae_mm"] for e in events],
                     label=name, color=colors[name], marker=".")
    names = list(result["experiments"])
    exposure = [result["experiments"][n]["exposure"] for n in names]
    axes[2].bar(names, [100 * e["active_reopening_anchors"] / e["total_windows"] for e in exposure],
                color=[colors[n] for n in names])
    axes[0].set(title="Fixed timeline validation loss", xlabel="Optimizer updates", ylabel="Flow loss")
    axes[1].set(title="Held-out active reopening probes", xlabel="Optimizer updates", ylabel="First gripper target MAE (mm)")
    axes[2].set(title="Verified optimizer exposure", ylabel="Active reopening anchors (%)")
    axes[2].axhline(25, color="gray", linestyle="--", linewidth=1)
    for axis in axes[:2]:
        axis.legend()
        axis.grid(alpha=.2)
    fig.suptitle("Equal-budget recovery SFT: sampling comparison\nOffline measurements, not autonomous task success", fontsize=12)
    fig.savefig(args.output / "sampling_comparison.png", dpi=160)
    fig.savefig(args.output / "sampling_comparison.pdf")
    plt.close(fig)
    conditions = ({"baseline": result["baseline"]} if "baseline" in result else {}) | result["experiments"]
    colors["baseline"] = "#68737d"
    labels = list(conditions)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.subplots_adjust(left=.065, right=.985, bottom=.14, top=.80, wspace=.30)
    for axis, group, metric, title, unit in (
        (axes[0], "new_active_reopening", "prefix5_gripper_mae_mm", "Active reopening: all recorded frames", "Five-command gripper MAE (mm)"),
        (axes[1], "new_active_reopening", "positive_opening_prediction_fraction", "Correct opening direction", "Predictions commanding opening (%)"),
        (axes[2], "old_clean", "prefix5_gripper_mae_mm", "Original clean behavior probes", "Five-command gripper MAE (mm)")):
        values = [conditions[n]["behavior"][group][metric] * (100 if metric.endswith("fraction") else 1) for n in labels]
        axis.bar(labels, values, color=[colors[n] for n in labels])
        axis.set(title=title, ylabel=unit)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Held-out recorded-observation behavior, three matched noise seeds\nOffline measurements, not autonomous task success", fontsize=12)
    fig.savefig(args.output / "behavior_comparison.png", dpi=160)
    fig.savefig(args.output / "behavior_comparison.pdf")
    plt.close(fig)
    behavior_rows = [{"condition": name, "group": group,
                      **{k: value for k, value in metrics.items() if isinstance(value, (int, float))}}
                     for name, condition in conditions.items() for group, metrics in condition["behavior"].items()
                     if group != "active_reopening_by_parent"]
    columns = ["condition", "group"] + sorted({k for row in behavior_rows for k in row} - {"condition", "group"})
    with (args.output / "behavior_metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(behavior_rows)
    print(json.dumps({"output": str(args.output), "matched_fields_checked": True,
                      "runs": {n: {"updates": r["updates"], "exposure": r["exposure"]} for n, r in result["experiments"].items()}}))


if __name__ == "__main__":
    main()
