"""Summarize independent state-only interventions and training coverage evidence."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import verified_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnosis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = {"diagnosis": args.diagnosis / "report.json", "sensitivity": args.diagnosis / "state_sensitivity/report.json",
        "alignment": args.diagnosis / "alignment.json", "coverage": args.diagnosis / "train_state_coverage.json"}
    data = {k: json.loads(v.read_text()) for k,v in paths.items()}
    if not data["sensitivity"]["completed"] or data["sensitivity"]["checkpoint_sha256"] != data["diagnosis"]["checkpoint_weights_sha256"]:
        raise ValueError("Sensitivity and diagnostic checkpoint binding differs")
    verified_source("scripts/probe_fp32_state_sensitivity.py", data["sensitivity"]["implementation"]["sha256"])
    for row in data["alignment"]["rows"]:
        path = args.diagnosis / row["episode"] / "recorded_observations_trajectory.npz"
        if sha256(path) != row["trajectory_sha256"]:
            raise ValueError("Alignment trajectory changed")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    labels = ["Recorded\nstate", "Only finger\nstate replaced", "All numeric\nstate replaced"]
    modes = ("recorded", "only_live_right_finger", "all_live_state")
    for axis, name in zip(axes, ("train_20002", "train_20004")):
        rows = [r for r in data["sensitivity"]["rows"] if r["episode"] == name and r["index"] == 300]
        if {r["seed"] for r in rows} != {1000, 1001}:
            raise ValueError("Both predeclared paired-noise samples required")
        means = [np.mean([r["variants"][mode]["first_finger_mm"] for r in rows]) for mode in modes]
        errors = [np.std([r["variants"][mode]["first_finger_mm"] for r in rows]) for mode in modes]
        bars = axis.bar(np.arange(3), means, yerr=errors, color=["#24885c", "#c46135", "#925735"])
        axis.bar_label(bars, fmt="%.1f", padding=5)
        axis.axhline(35, color="black", linestyle="--", label="Expert target: 35 mm")
        axis.set_xticks(np.arange(3), labels)
        axis.set_ylim(0, 42)
        axis.set_ylabel("Predicted first finger target (mm)")
        axis.set_title(name + f" | input finger 35.0 → {rows[0]['live_finger_state_mm']:.1f} mm")
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("Same images, instruction and latent noise; change numeric state only\n"
        "Command 300: model closes further instead of reopening toward the expert target", fontsize=11)
    fig.tight_layout(rect=(0, .10, 1, .88))
    coverage = data["coverage"]
    fig.text(.05, .035, f"Training coverage: {coverage['parents']} parents / {coverage['preclose_frames']:,} pre-close frames; "
        f"{coverage['below34mm']} below 34 mm.\nTwo fixed noise samples; counterfactual sensitivity, not autonomous success.", fontsize=9)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "state_sensitivity.png", dpi=160)
    fig.savefig(args.output / "state_sensitivity.pdf")
    plt.close(fig)
    (args.output / "diagnosis_summary.json").write_text(json.dumps({**data,
        "evidence_hashes": {k:sha256(v) for k,v in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
