"""Dataset visualizations: real recordings and saved SmolVLA test results.

No training, downloads, synthesized camera images or dataset edits. Requires
NumPy, Matplotlib and Pillow. Refuses to overwrite an existing output directory.
"""
import argparse
from html import escape
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.smolvla_data import sha256

INK, MUTED, BLUE, GREEN, ORANGE = "#14263c", "#52657b", "#2463ce", "#16806a", "#cf662a"
PHASES = ["survey", "transit_pick", "approach", "descend", "close", "lift",
          "transit_place", "arrive", "align", "insert", "hold"]
PHASE_COLORS = dict(zip(PHASES, plt.get_cmap("tab20").colors[:len(PHASES)]))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "text.color": INK, "axes.labelcolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.facecolor": "white",
                     "savefig.facecolor": "white", "pdf.fonttype": 42})


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def page(title, subtitle):
    fig = plt.figure(figsize=(16, 9))
    fig.text(.045, .943, title, fontsize=25, weight="bold")
    fig.text(.045, .9, subtitle, fontsize=12, color=MUTED)
    return fig


def footer(fig, text):
    fig.text(.045, .037, text, fontsize=9, color=MUTED)


def box(ax, x, y, w, h, title, detail, color=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                              facecolor="#f3f6fb", edgecolor=color, linewidth=1.5))
    ax.text(x + .018, y + h - .045, title, fontsize=14, weight="bold", va="top", color=color)
    ax.text(x + .018, y + h - .11, detail, fontsize=11, va="top", linespacing=1.6)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=19,
                               color=MUTED, linewidth=1.7))


def phase_segments(phases, timestamps):
    starts = np.r_[0, np.flatnonzero(phases[1:] != phases[:-1]) + 1]
    ends = np.r_[starts[1:], len(phases)]
    return [(str(phases[a]), float(timestamps[a]), float(timestamps[b])) for a, b in zip(starts, ends)]


def first_phase_frame(phases, phase, fraction=.75):
    """Pick from the first contiguous occurrence, not a later repeated approach."""
    indices = np.flatnonzero(phases == phase)
    if not len(indices):
        raise ValueError(f"No recorded {phase} phase")
    gaps = np.flatnonzero(np.diff(indices) > 1)
    if len(gaps):
        indices = indices[:gaps[0] + 1]
    return int(indices[int(fraction * (len(indices) - 1))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "test_collection_new_env")
    parser.add_argument("--selection", type=Path, default=ROOT / "configs/smolvla_collection_smoke.json")
    parser.add_argument("--qualification", type=Path, default=ROOT / "results/smolvla_training/qualification_v2.json")
    parser.add_argument("--rollout", type=Path, default=ROOT / "results/smolvla_rollout/heldout_v2/report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/dataset_visualization_v3")
    args = parser.parse_args()
    selection, training, rollout = load_json(args.selection), load_json(args.qualification), load_json(args.rollout)
    episodes = []
    for split in ("train", "validation"):
        for name in selection["splits"][split]:
            path = args.dataset / name
            meta = load_json(path / "metadata.json")
            if sha256(path / "episode.npz") != meta["data_sha256"]:
                raise ValueError(f"Recording checksum mismatch: {name}")
            with np.load(path / "episode.npz", allow_pickle=False) as raw:
                record = {"name": name, "split": split, "metadata": meta,
                          "phase": raw["phase"], "timestamps": raw["timestamp_s"],
                          "state": raw["observation_state"], "actions": raw["action_applied"],
                          "durations": raw["executed_dt_s"]}
            if len(record["actions"]) != meta["steps"] or len(record["state"]) != meta["steps"] + 1:
                raise ValueError("Unexpected observation/action alignment")
            episodes.append(record)
    if not training["passed"] or len(training["training_events"]) != training["unique_training_updates"]:
        raise ValueError("Expected a successful report with unique update events")
    args.output.mkdir(parents=True, exist_ok=False)
    assets, provenance = [], {}
    pdf_path = args.output / "dataset_training_report.pdf"
    pdf = PdfPages(pdf_path, metadata={"Title": "OpenArm: dataset and SmolVLA training evidence",
                                      "Author": "OpenArm Capstone team"})

    def save(fig, stem, caption):
        fig.savefig(args.output / (stem + ".png"), dpi=150)
        pdf.savefig(fig)
        assets.append({"image": stem + ".png", "caption": caption})
        plt.close(fig)

    counts = {split: sum(len(e["actions"]) for e in episodes if e["split"] == split)
              for split in ("train", "validation")}
    total = sum(counts.values())
    # 1. Real dataset contents and observation/action contract.
    fig = page("01  |  What our dataset contains", "Automated successful expert demonstrations, not learned-policy successes")
    ax = fig.add_axes([.045, .13, .91, .7]); ax.axis("off")
    table_rows = [[e["name"], e["split"], f'{len(e["actions"]):,}',
                   f'{sum(e["durations"]):.2f} s', e["metadata"]["outcome"]] for e in episodes]
    table = ax.table(cellText=table_rows, colLabels=["Episode", "Use", "Commands", "Simulated time", "Expert outcome"],
                     bbox=[0, .55, 1, .38], cellLoc="left", colLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(13)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#e7edf7" if row == 0 else ("#eef8f4" if table_rows[row-1][1] == "train" else "#fff3e9"))
        if row == 0: cell.set_text_props(weight="bold")
    box(ax, .01, .19, .30, .26, "Observation at time t",
        "Fixed RGB + wrist RGB: 320 x 240\n16 robot joint/gripper values\nInstruction: insert the plug", BLUE)
    box(ax, .35, .19, .30, .26, "Expert command at time t",
        "7 right-arm joint targets [rad]\n1 per-finger travel target [m]\nRequested + applied values saved", GREEN)
    box(ax, .69, .19, .30, .26, "Timing and provenance",
        "50 Hz command boundaries\nT commands, T+1 observations\nSeeds, phases, hashes, outcomes", ORANGE)
    ax.text(.01, .08, f"{total:,} total transitions   |   {counts['train']:,} train / {counts['validation']:,} validation",
            fontsize=17, weight="bold")
    footer(fig, "Scope: four locally available episodes; explicit split override. Missing train_00005 is excluded, not fabricated. Source: raw NPZ + metadata.")
    save(fig, "01_dataset_overview", "Four successful full-task expert recordings; validation is a separate complete episode.")

    # 2. Two-camera storyboard: exact pixels at selected command boundaries.
    example = next(e for e in episodes if e["split"] == "train")
    phases_to_show = [("survey", "Table start"), ("descend", "Reach / descend"), ("close", "Grasp / close"),
                      ("lift", "Lift"), ("transit_place", "Transport"), ("insert", "Insert")]
    selected = []
    for phase, label in phases_to_show:
        index = first_phase_frame(example["phase"], phase)
        selected.append((index, label, phase))
    if any(a[0] >= b[0] for a, b in zip(selected, selected[1:])):
        raise ValueError("Storyboard phases are not chronological in this recording")
    images = {}
    with np.load(args.dataset / example["name"] / "episode.npz", allow_pickle=False) as raw:
        for camera in ("image_scene_rgb", "image_wrist_rgb"):
            frames = raw[camera]
            images[camera] = frames[[s[0] for s in selected]].copy()
            del frames
    provenance["storyboard"] = {"episode": example["name"], "frames": [
        {"step": idx, "phase": phase, "timestamp_s": float(example["timestamps"][idx])} for idx, _, phase in selected]}
    fig = page("02  |  What the model sees", f"Actual recorded RGB frames from {example['name']} (training episode)")
    grid = fig.add_gridspec(2, 6, left=.06, right=.98, top=.79, bottom=.18, hspace=.26, wspace=.05)
    for row, camera in enumerate(("image_scene_rgb", "image_wrist_rgb")):
        for col, (index, label, _) in enumerate(selected):
            ax = fig.add_subplot(grid[row, col]); ax.imshow(images[camera][col]); ax.axis("off")
            if row == 0: ax.set_title(label, fontsize=12, pad=9)
            ax.text(.5, -.06, f"t={example['timestamps'][index]:.2f}s | frame {index}",
                    transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
    fig.text(.014, .65, "Fixed camera", rotation=90, fontsize=12)
    fig.text(.014, .31, "Wrist camera", rotation=90, fontsize=12)
    fig.text(.06, .11, "Frames show expert behavior. Phase labels are presentation annotations, never actor inputs.", fontsize=12, color=MUTED)
    footer(fig, "Lossless source pixels; resized for the slide only. The instruction and robot state accompany both images.")
    save(fig, "02_camera_storyboard", "Synchronized fixed and wrist views show table start through expert insertion.")
    del images

    # 3. Actual phase timelines plus the sparse windows used in qualification.
    fig = page("03  |  Full episodes vs. the tiny training test", "Colored segments: recorded expert phases. Black ticks: selected qualification window starts.")
    ax = fig.add_axes([.16, .34, .79, .47])
    for row, e in enumerate(episodes):
        for phase, start, end in phase_segments(e["phase"], e["timestamps"]):
            ax.broken_barh([(start, end-start)], (row-.30, .6), facecolors=PHASE_COLORS[phase])
        samples = training["sample_plan"][e["split"]]
        xs = [e["timestamps"][s["step"]] for s in samples if s["episode"] == e["name"]]
        ax.scatter(xs, np.full(len(xs), row), marker="|", s=240, linewidths=2.5, color=INK, zorder=5)
    ax.set_yticks(range(len(episodes)), [e["name"] + "\n" + e["split"] for e in episodes], fontsize=11)
    ax.invert_yaxis(); ax.set_xlabel("Recorded simulated time [seconds]")
    ax.set_xlim(0, max(e["timestamps"][-1] for e in episodes) + 1)
    ax.legend(handles=[Patch(color=PHASE_COLORS[p], label=p.replace("_", " ")) for p in PHASES],
              loc="upper center", bbox_to_anchor=(.48, -.20), ncol=6, frameon=False, fontsize=10)
    fig.text(.07, .14, "The export contains all frames; the 24-update qualification used only 12 train + 4 validation windows.", fontsize=13)
    fig.text(.07, .10, "Next: train across the complete training episodes, including the initial survey/table-start behavior.", fontsize=12, color=MUTED)
    footer(fig, "Source: phase and timestamp arrays + qualification_v2.json sample_plan. Validation ticks are evaluation only, not training updates.")
    save(fig, "03_phase_coverage", "Recorded full-task phase coverage and the small subset used by the diagnostic training run.")

    # 4. One real physical-unit action chunk and terminal masking.
    anchor = selected[2][0]
    horizon = 50
    end = min(anchor + horizon, len(example["actions"]))
    chunk = example["actions"][anchor:end]
    times = example["timestamps"][anchor:end] - example["timestamps"][anchor]
    fig = page("04  |  One training sample", "Observation at t + instruction -> future applied expert targets (50 commands at nominal 50 Hz)")
    ax = fig.add_axes([.065, .47, .60, .32])
    for j in range(7): ax.plot(times, chunk[:, j], label=f"Right J{j+1}", linewidth=1.7)
    ax.set_ylabel("Absolute joint target [rad]"); ax.set_xlabel("Time relative to observation [s]")
    ax.legend(ncol=4, fontsize=9, loc="lower left"); ax.grid(alpha=.15)
    ax2 = fig.add_axes([.065, .16, .60, .22])
    ax2.plot(times, 1000 * chunk[:, 7], color=GREEN, linewidth=2)
    ax2.set_ylabel("Per-finger travel [mm]"); ax2.set_xlabel("Time relative to observation [s]"); ax2.grid(alpha=.15)
    ax3 = fig.add_axes([.71, .16, .255, .65]); ax3.axis("off")
    right = example["state"][anchor, 8:]
    ax3.text(0, .97, "Actual robot state at t", fontsize=15, weight="bold", va="top")
    ax3.text(0, .86, "Left: 7 joints + mean finger travel\nRight: 7 joints + mean finger travel\n16 values total; both arms observed", fontsize=11, va="top", linespacing=1.5)
    ax3.text(0, .66, "Right joints [rad]:\n" + "\n".join(f"J{j+1}: {right[j]: .4f}" for j in range(7)) +
             f"\nFinger: {1000*right[7]:.2f} mm", fontsize=11, va="top", linespacing=1.35)
    ax3.text(0, .24, "Near episode end:\nreal actions contribute to loss;\npadded future actions do not.\nLast command: 1 valid + 49 padded.",
             fontsize=11, color=ORANGE, va="top", linespacing=1.4)
    provenance["action_example"] = {"episode": example["name"], "step": anchor, "real_actions": len(chunk)}
    footer(fig, f"Source: {example['name']} step {anchor}. Gripper values shown in mm for readability; stored/trained in metres. Labels are targets, not next-state positions.")
    save(fig, "04_training_sample", "A real 50x8 action target chunk, robot-state mapping and terminal padding convention.")

    # 5. Implemented training/inference pipeline, with no privileged actor input.
    fig = page("05  |  How we fine-tune and use SmolVLA", "Implemented pipeline using pretrained SmolVLA and LeRobot 0.4.4 model/processors")
    ax = fig.add_axes([.05, .13, .9, .69]); ax.axis("off")
    ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02)
    box(ax, .01, .60, .27, .32, "1. Validate + export", "Check hashes, units and alignment\nSplit by whole episode\nNormalize with TRAIN stats only", BLUE)
    box(ax, .36, .60, .27, .32, "2. Form training batches", "Two RGB views + state + text\nFuture 50 x 8 expert targets\nBatch 1 + terminal padding mask", BLUE)
    box(ax, .71, .60, .27, .32, "3. Fine-tune model", "Stochastic flow-matching loss\nAdamW, LR 1e-5, BF16\nFrozen backbone; train action expert\nGradient clipping: norm 10", GREEN)
    arrow(ax, (.285, .76), (.35, .76)); arrow(ax, (.635, .76), (.70, .76))
    box(ax, .71, .08, .27, .32, "4. Verify + checkpoint", "Held-out fixed-noise loss check\nSave weights + processors + state\nOptimizer / scheduler / RNG resume", GREEN)
    box(ax, .36, .08, .27, .32, "5. Decode policy actions", "New images + state + instruction\nPredict 50 targets, physical units\nNo expert action labels at inference", BLUE)
    box(ax, .01, .08, .27, .32, "6. Run guarded simulation", "Execute 5 commands, observe again\nExisting bounds + collision guard\nRecord contacts, timing and outcome", ORANGE)
    arrow(ax, (.845, .59), (.845, .42)); arrow(ax, (.70, .24), (.64, .24)); arrow(ax, (.35, .24), (.29, .24))
    fig.text(.06, .10, "Privileged plug/socket poses and phase/contact diagnostics support the expert/evaluation only; never the VLA actor.", fontsize=11)
    footer(fig, "This diagram describes the implemented diagnostic scripts, not qualification of a long-run trainer, online RL, hardware deployment or successful learned insertion.")
    save(fig, "05_training_pipeline", "Actual data-to-training-to-guarded-simulation workflow, with separate expert labels and actor inputs.")

    # 6. Measured loss, reproducibility, memory and candid deployment boundary.
    fig = page("06  |  Evidence that training executes", "Actual saved test results: short-run numerical health and reproducibility, not autonomous task success")
    ax = fig.add_axes([.065, .42, .43, .36])
    events = training["training_events"]
    ax.plot([e["step"] for e in events], [e["loss"] for e in events], marker="o", markersize=4, color=BLUE)
    ax.axvline(12, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Unique optimizer update"); ax.set_ylabel("Stochastic training loss")
    ax.set_title("Raw loss varies with sample + noise", fontsize=13); ax.grid(alpha=.15)
    ax.text(.04, .93, "Checkpoint at 12; 3 resumed updates match exactly", transform=ax.transAxes, fontsize=9, va="top")
    ax2 = fig.add_axes([.565, .42, .37, .36])
    x = np.arange(2)
    before = [training["evaluation_before"][s]["mean"] for s in ("train", "validation")]
    after = [training["evaluation_after"][s]["mean"] for s in ("train", "validation")]
    b1 = ax2.bar(x-.18, before, .34, label="Before", color="#adc4ed")
    b2 = ax2.bar(x+.18, after, .34, label="After 24 updates", color=GREEN)
    ax2.bar_label(b1, fmt="%.4f", fontsize=10); ax2.bar_label(b2, fmt="%.4f", fontsize=10)
    ax2.set_xticks(x, ["Train: 12 windows", "Validation: 4 windows"], fontsize=10)
    ax2.set_ylim(0, max(before)*1.32); ax2.set_ylabel("Fixed-noise mean flow loss")
    ax2.set_title("Repeatable before / after comparison", fontsize=13); ax2.legend(frameon=False, fontsize=9)
    fig.text(.065, .30, f"Measured: RTX 4060 Ti | batch 1 / BF16 | peak allocated {training['peak_allocated_gb']:.2f} GB | {training['seconds']:.1f}s test interval",
             fontsize=12, weight="bold")
    fig.text(.065, .24, "Finite losses/gradients; exact saved-state resume. Earlier smoke test verified exact model/processor reload.", fontsize=12)
    fig.text(.065, .16, f"Boundary: learned policy stopped after {rollout['steps']} commands, before grasp ({rollout['outcome']}). No task-success claim.",
             fontsize=12, color=ORANGE, weight="bold")
    fig.text(.065, .10, "Next: all-frame supervised training, longer stability tests, more demonstrations, then reserved-reset evaluation. RL comes later.", fontsize=11, color=MUTED)
    footer(fig, "Sources: qualification_v2.json, smoke_v3.json, heldout_v2/report.json. Memory excludes other GPU usage; test timing includes checks, not imports or downloads.")
    save(fig, "06_training_evidence", "Measured stochastic loss, held-out loss improvement, memory and reproducibility, with the unsuccessful rollout disclosed.")
    pdf.close()
    manifest = {"title": "OpenArm dataset and SmolVLA training evidence", "assets": assets,
                "episodes": [{"episode": e["name"], "split": e["split"], "steps": len(e["actions"]),
                              "simulated_s": float(sum(e["durations"])), "data_sha256": e["metadata"]["data_sha256"]} for e in episodes],
                "total_transitions": total, "counts": counts, "examples": provenance,
                "sources": {str(p.resolve()): sha256(p) for p in (args.selection, args.qualification, args.rollout)},
                "limits": "Expert camera frames are not policy successes. Tiny qualification only; no bug-free or long-run stability claim."}
    (args.output / "evidence.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sections = "\n".join(f'<section><img src="{escape(a["image"])}" alt="{escape(a["caption"])}"><p>{escape(a["caption"])}</p></section>' for a in assets)
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenArm training evidence</title>
<style>body{font:16px system-ui,sans-serif;color:#14263c;background:#edf2f8;margin:0}
main{max-width:1280px;margin:32px auto;padding:0 20px}section{background:white;border-radius:14px;padding:12px;margin:24px 0}
img{width:100%;height:auto}p{padding:0 20px;line-height:1.6}a{color:#2463ce}</style></head><body><main>
<h1>OpenArm: dataset and SmolVLA training evidence</h1>
<p>Real recordings and saved test metrics. Expert demonstrations succeed; the tiny learned policy does not yet.</p>
<p><a href="dataset_training_report.pdf">Download the six-page PDF</a> · <a href="evidence.json">View provenance</a></p>
''' + sections + '</main></body></html>'
    (args.output / "index.html").write_text(html, encoding="utf-8")
    notes = """# Dataset and SmolVLA training visualizations

Open `dataset_training_report.pdf` (six landscape pages) or `index.html`
(offline gallery). PNGs can be copied directly into PowerPoint or Google Slides.
Every chart uses actual recordings or saved test reports; no new training was run.

## Visualization overview

1. **Dataset:** Four successful automated expert recordings, 7,251 commands.
   Three episodes train; one is validation. Two cameras, bimanual state and
   instruction map to eight right-arm targets. These are simulation demonstrations.
2. **Camera views:** The same timestamps appear in both views. The images are
   actual lossless-source expert frames, not an autonomous learned-policy run.
3. **Coverage:** Full recordings cover pickup through insertion. The initial
   qualification used only four windows per episode, so it is a pipeline check,
   not full-dataset training. Phase labels never enter the actor.
4. **Sample:** One observation predicts 50 future commands. Joint targets are
   absolute radians; gripper travel is metres (shown in mm). Terminal padding
   contributes no loss. Labels are applied commands, not next observed positions.
5. **Training:** Pretrained SmolVLA, train-only normalization, masked stochastic
   flow-matching loss, batch one/BF16/AdamW. Save processors with model weights;
   decode predicted targets through the same guarded simulator controller.
6. **Evidence:** The saved 24-update test has finite losses/gradients, improved
   fixed-noise training/validation loss and an exact three-update resume check
   on the same runtime. Raw stochastic loss is not expected to be monotonic.
   The learned policy still collides before grasp: we have proved execution of
   the training/inference pipeline, not task competence or bug-free long runs.

## Sources and reproducibility

- Raw data: `test_collection_new_env/train_00000` through `train_00003`.
- Explicit split: `configs/smolvla_collection_smoke.json`.
- Training: `results/smolvla_training/qualification_v2.json` and `smoke_v3.json`.
- Policy rollout: `results/smolvla_rollout/heldout_v2/report.json`.
- Environment: automateData revision b7297ba82701fe7e7814c32ea617a85f832b74a4,
  matched to recorded source/workspace/robot XML and runtime scene contracts.
- `evidence.json`: episode checksums, exact storyboard/sample frame indices and
  hashes of the selected input reports/config.

Regenerate with a fresh directory:

```powershell
python scripts/visualize_smolvla_dataset.py --output results/dataset_visualization_repeat
```

Requires NumPy, Matplotlib and Pillow in the plotting environment. The command
only reads the original recordings/reports and refuses existing output paths.
"""
    (args.output / "README.md").write_text(notes, encoding="utf-8")
    print(json.dumps({"pdf": str(pdf_path), "gallery": str(args.output / "index.html"),
                      "pages": len(assets), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
