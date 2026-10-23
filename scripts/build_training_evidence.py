"""Slide-ready plots from the completed v2c30 runs and actual expert recordings."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scripts.visualize_smolvla_dataset import page, footer, first_phase_frame, PHASES, PHASE_COLORS, BLUE, GREEN, ORANGE
from learning.vla.smolvla_data import sha256


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    collection = ROOT / "data/drive_v2c30_20261012/merged_all_v1"
    receipt_path = ROOT / "data/drive_v2c30_20261012/export_all_v1/openarm_export.json"
    receipt = load(receipt_path)
    runs = [ROOT / "outputs" / name for name in ("v2c30_sft_overnight_v1", "v2c30_sft_until_0800_v1")]
    reports = [load(run / "report.json") for run in runs]
    if not all(report["completed"] for report in reports):
        raise ValueError("Training incomplete")
    updates, evaluations = [], []
    for stage, run in enumerate(runs):
        for line in (run / "events.jsonl").read_text().splitlines():
            event = json.loads(line)
            event["stage"] = stage
            if event["type"] == "update":
                if not np.isfinite([event["loss"], event["grad_norm"]]).all():
                    raise ValueError("Nonfinite training history")
                updates.append(event)
            elif event["type"] == "evaluation":
                evaluations.append(event)
    if len({e["step"] for e in updates}) != len(updates):
        raise ValueError("Duplicate optimizer updates")
    split_counts = receipt["splits"]
    phase_counts, lengths = Counter(), {"train": [], "validation": []}
    for split in lengths:
        for record in receipt["datasets"][split]["episodes"]:
            lengths[split].append(record["steps"])
            with np.load(collection / record["episode"] / "episode.npz", allow_pickle=False) as raw:
                labels, counts = np.unique(raw["phase"], return_counts=True)
                phase_counts.update(dict(zip(labels.tolist(), counts.tolist())))
    example = receipt["datasets"]["train"]["episodes"][0]
    example_path = collection / example["episode"] / "episode.npz"
    if sha256(example_path) != example["data_sha256"]:
        raise ValueError("Storyboard source checksum mismatch")
    with np.load(example_path, allow_pickle=False) as raw:
        phases, timestamps = raw["phase"], raw["timestamp_s"]
        actions, states = raw["action_applied"], raw["observation_state"]
        stages = [("survey", "Table start"), ("descend", "Reach"), ("close", "Close gripper"),
                  ("lift", "Lift"), ("transit_place", "Transport"), ("insert", "Insert")]
        selected = [first_phase_frame(phases, phase) for phase, _ in stages]
        images = {camera: raw[camera][selected].copy() for camera in ("image_scene_rgb", "image_wrist_rgb")}
    assets = []
    pdf = PdfPages(args.output / "training_evidence.pdf")
    def save(fig, name, caption):
        fig.savefig(args.output / (name + ".png"), dpi=150)
        pdf.savefig(fig)
        assets.append({"file": name + ".png", "caption": caption})
        plt.close(fig)
    fig = page("Dataset | 101 simulation demonstrations", "OpenArm v1: right arm active, left arm parked | table start → grasp → insert")
    ax = fig.add_axes([.07, .36, .38, .43])
    bars = ax.bar(["Training", "Validation"], [81, 20], color=[BLUE, ORANGE], width=.55)
    ax.bar_label(bars, padding=8, fontsize=20)
    ax.set_ylim(0, 100); ax.set_ylabel("Complete episodes")
    ax.set_title("Split by episode")
    ax2 = fig.add_axes([.56, .36, .38, .43])
    ax2.hist([lengths["train"], lengths["validation"]], bins=12, stacked=True,
             label=["Train", "Validation"], color=[BLUE, ORANGE])
    ax2.set_xlabel("Commands per episode"); ax2.set_ylabel("Episodes"); ax2.legend(frameon=False)
    fig.text(.07, .24, "147,499 training commands   |   37,104 validation commands   |   50 Hz", fontsize=20, weight="bold")
    fig.text(.07, .17, "Inputs: fixed RGB + wrist RGB (320 × 240), 16 robot-state values, task instruction", fontsize=15)
    fig.text(.07, .11, "Targets: 50 future commands × 8 values (7 right-arm joints + gripper travel)", fontsize=15)
    footer(fig, "Actual merged release and audited export receipt. Validation episodes never supply optimizer updates or normalization statistics.")
    save(fig, "01_dataset_overview", "101 expert simulation episodes; episode-level train/validation split.")
    fig = page("Dataset | Synchronized camera observations", f"Actual expert demonstration: {example['episode']} | These frames are training data, not VLA rollout results")
    grid = fig.add_gridspec(2, 6, left=.06, right=.98, top=.79, bottom=.18, hspace=.25, wspace=.05)
    for row, camera in enumerate(images):
        for column, index in enumerate(selected):
            ax = fig.add_subplot(grid[row, column]); ax.imshow(images[camera][column]); ax.axis("off")
            if row == 0:
                ax.set_title(stages[column][1], fontsize=12)
            ax.text(.5, -.08, f"{timestamps[index]:.2f} s", transform=ax.transAxes, ha="center", fontsize=10)
    fig.text(.018, .62, "Fixed camera", rotation=90)
    fig.text(.018, .28, "Wrist camera", rotation=90)
    footer(fig, "Original lossless NPZ pixels. Phase labels annotate this figure; they are not VLA actor inputs.")
    save(fig, "02_expert_camera_storyboard", "Real expert frames from both cameras at the same timestamps.")
    fig = page("Dataset | Coverage and prediction targets", "Every training command is eligible for sampling; terminal action chunks use a padding mask")
    ax = fig.add_axes([.13, .22, .33, .57])
    ax.barh([s.replace("_", " ") for s in PHASES], [phase_counts[s] for s in PHASES], color=[PHASE_COLORS[s] for s in PHASES])
    ax.invert_yaxis(); ax.set_xlabel("Commands across all 101 episodes"); ax.set_title("Recorded phase coverage")
    anchor = selected[2]
    chunk = actions[anchor:anchor + 50]
    ax2 = fig.add_axes([.57, .44, .37, .35])
    for joint in range(7):
        ax2.plot(np.arange(len(chunk)) / 50, chunk[:, joint], label=f"J{joint + 1}")
    ax2.set_ylabel("Absolute joint target [rad]"); ax2.set_xlabel("Future simulated time [s]")
    ax2.legend(ncol=4, fontsize=9, frameon=False)
    ax3 = fig.add_axes([.57, .18, .37, .16])
    ax3.plot(np.arange(len(chunk)) / 50, chunk[:, 7] * 1000, color=GREEN)
    ax3.set_ylabel("Finger travel [mm]"); ax3.set_xlabel("Future simulated time [s]")
    footer(fig, f"Target example: {example['episode']}, command {anchor}. Gripper targets are stored in metres; converted to mm here.")
    save(fig, "03_phase_coverage_and_targets", "Phase coverage from all raw recordings and one real 50-command target chunk.")
    fig = page("Training | Loss history across 19,128 updates", "SmolVLA action expert + state projection | frozen vision/language backbone | batch 16, BF16, RTX 4060 Ti")
    steps = np.array([e["step"] for e in updates]); losses = np.array([e["loss"] for e in updates])
    boundary = reports[0]["updates"]
    ax = fig.add_axes([.075, .22, .40, .58])
    ax.plot(steps, losses, color="#b9ccec", alpha=.4, linewidth=.5, label="Raw stochastic loss")
    ax.plot(steps[99:], np.convolve(losses, np.ones(100) / 100, mode="valid"), color=BLUE, label="100-update mean")
    ax.set_yscale("log"); ax.set_xlabel("Cumulative optimizer updates"); ax.set_ylabel("Training flow loss (log scale)")
    ax.axvline(boundary, color=ORANGE, linestyle="--", label="Low-LR continuation")
    ax.legend(frameon=False, fontsize=10); ax.grid(alpha=.15)
    ax2 = fig.add_axes([.56, .22, .39, .58])
    evsteps = [e["step"] for e in evaluations]
    vals = [e["scores"]["validation"]["mean"] for e in evaluations]
    ax2.plot(evsteps, vals, color=GREEN, marker=".", label="64 held-out fixed probes")
    ax2.set_yscale("log"); ax2.set_xlabel("Cumulative optimizer updates"); ax2.set_ylabel("Validation flow loss (log scale)")
    ax2.axvline(boundary, color=ORANGE, linestyle="--"); ax2.grid(alpha=.15); ax2.legend(frameon=False, fontsize=10)
    fig.text(.075, .12, f"Initial validation: {vals[0]:.5f}   |   Best saved probe loss: {reports[0]['best_validation_mean']:.5f}", fontsize=17, weight="bold")
    footer(fig, "Validation uses 64 fixed windows, fixed flow time/noise; not full validation. Training loss uses random batches/noise. Low loss does not prove task success.")
    save(fig, "04_training_validation_loss", "Actual per-update training loss and 64-window fixed-probe validation; continuation starts at update 11,830.")
    fig = page("Validation | Late-training behavior", "Fixed held-out probes let us compare saved checkpoints; lower is better")
    ax = fig.add_axes([.10, .24, .83, .54])
    late = [(e["step"], e["scores"]["validation"]["mean"]) for e in evaluations if e["step"] >= 6000]
    ax.plot(*zip(*late), color=GREEN, marker="o", markersize=4)
    ax.axvline(boundary, color=ORANGE, linestyle="--", label="Constant final-LR continuation")
    for run, color, label in zip(runs, [BLUE, ORANGE], ["Original best", "Continuation best"]):
        best = load(run / "best_checkpoint.json")
        marker = load(Path(best["path"]) / "complete.json")
        ax.scatter([marker["step"]], [best["validation_mean"]], s=90, color=color, zorder=4, label=f"{label}: {best['validation_mean']:.6f}")
    ax.set_xlabel("Cumulative optimizer updates"); ax.set_ylabel("Fixed-probe validation loss"); ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=11)
    fig.text(.1, .13, "The extra training did not beat the earlier best probe score; evaluate the earlier best in simulation.", fontsize=14)
    footer(fig, "Small score differences are not a statistically established ranking of task performance. These validation resets are not an untouched test set.")
    save(fig, "05_validation_detail", "Late validation curve with best checkpoints identified, including the plateau during continuation.")
    evaluation = load(args.evaluation / "report.json")
    if not evaluation["completed"]:
        raise ValueError("Simulation evaluation not complete")
    rollouts = evaluation["rollouts"]
    fig = page("Evaluation | Closed-loop VLA simulation", "Best saved checkpoint by validation-probe loss | original collector scene and collision guards")
    ax = fig.add_axes([.06, .35, .88, .40]); ax.axis("off")
    table = ax.table(cellText=[[r["source_episode"], r["steps"], "Yes" if r["grasp_established"] else "No",
                               "Yes" if r["task_success"] else "No", r["outcome"]] for r in rollouts],
                     colLabels=["Held-out reset", "Commands", "Grasp", "Insertion success", "Outcome"],
                     cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(13)
    fig.text(.07, .24, f"Observed insertion successes: {sum(r['task_success'] for r in rollouts)} / {len(rollouts)} trials", fontsize=21, weight="bold")
    fig.text(.07, .16, "Training executes and loss improves; closed-loop task competence must be measured separately.", fontsize=14)
    footer(fig, "Finite-horizon validation-reset trials, policy seed 1000, 5-command execution prefix. Simulation pauses during inference; no hardware qualification.")
    save(fig, "06_simulation_results", "All evaluated resets and outcomes, including unsuccessful attempts.")
    pdf.close()
    for name, events, fields in [("training_history.csv", updates, ["step", "stage", "loss", "grad_norm", "lr", "update_seconds"]),
                                  ("validation_history.csv", [{"step": e["step"], "stage": e["stage"], "validation_loss": e["scores"]["validation"]["mean"]} for e in evaluations], ["step", "stage", "validation_loss"])]:
        with (args.output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(events)
    video_paths = []
    for folder in sorted(args.evaluation.iterdir()):
        if folder.is_dir() and (folder / "rollout.mp4").exists():
            name = folder.name + ".mp4"
            shutil.copy2(folder / "rollout.mp4", args.output / name)
            shutil.copy2(folder / "rollout.png", args.output / (folder.name + "_poster.png"))
            video_paths.append(name)
    evidence = {"assets": assets, "videos": video_paths, "training_updates": len(updates),
                "splits": split_counts, "validation_scope": "64 fixed windows, not full validation",
                "phase_command_counts": phase_counts, "storyboard_episode": example["episode"],
                "storyboard_frames": selected, "storyboard_source_sha256": example["data_sha256"],
                "sources": {str(path): sha256(path) for path in [receipt_path, *[r / "events.jsonl" for r in runs], args.evaluation / "report.json"]},
                "rollouts": rollouts}
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    (args.output / "README.md").write_text("# OpenArm SmolVLA training evidence\n\n"
        "Six 16:9 PNG figures and training_evidence.pdf can be imported into slides.\n"
        "MP4s are actual closed-loop VLA rollouts with both cameras, at 1x simulated time; inference pauses are omitted.\n"
        "Dataset storyboard images show expert demonstrations, not learned-policy successes.\n"
        "Loss plots use logged 64-window validation probes, not the full validation dataset.\n"
        "All evaluated outcomes, including failures, appear in figure 06 and evidence.json.\n"
        "CSV files contain the chart data; evidence.json records provenance and model hashes.\n", encoding="utf-8")
    archive = args.output.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    shutil.make_archive(str(args.output), "zip", root_dir=args.output)
    print(json.dumps({"output": str(args.output), "assets": len(assets), "videos": video_paths,
                      "updates": len(updates), "archive": str(archive)}, indent=2))


if __name__ == "__main__":
    main()
