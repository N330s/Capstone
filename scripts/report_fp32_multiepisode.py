"""Plot logged training and fixed validation probes, without GPU or model loading."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.campaign / "training/report.json").read_text())
    events = list(map(json.loads, (args.campaign / "training/events.jsonl").read_text().splitlines()))
    updates = [e for e in events if e["type"] == "update"]
    checks = [e for e in events if e["type"] == "evaluation"]
    if not report["completed"] or len(updates) != report["updates"] or not checks:
        raise ValueError("Complete, consistent training logs required")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not all(np.isfinite(e[k]) for e in updates for k in ("loss", "grad_norm", "lr")):
        raise ValueError("Nonfinite training events")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    steps = [e["step"] for e in updates]
    losses = np.array([e["loss"] for e in updates])
    axes[0, 0].plot(steps, losses, alpha=.15, color="#27699a", label="Stochastic batch loss")
    width = min(100, len(losses))
    axes[0, 0].plot(steps[width-1:], np.convolve(losses, np.ones(width)/width, mode="valid"), label=f"{width}-update mean")
    axes[0, 0].set_ylabel("Masked flow-matching loss")
    for split in ("train", "validation"):
        axes[0, 1].plot([e["step"] for e in checks], [e["scores"][split]["mean"] for e in checks], "o-", label=split)
        axes[1, 0].plot([e["step"] for e in checks], [e["scores"][split]["gripper_mae_mm"] for e in checks], "o-", label=split)
    axes[0, 1].set_ylabel("Fixed timeline probe loss (not full validation)")
    axes[1, 0].set_ylabel("Decoded gripper MAE, 8 anchors/split (mm)")
    axes[1, 1].plot(steps, [e["lr"] for e in updates], label="Learning rate")
    axes[1, 1].set_ylabel("Learning rate")
    for axis in axes.flat:
        axis.set_xlabel("Optimizer updates")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("FP32 multi-episode SmolVLA SFT | 81 training / 20 validation parents\n"
        "Two RGB views + 16 positions; task success scored separately in live simulation")
    fig.tight_layout(rect=(0, 0, 1, .93))
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "training_validation.png", dpi=160)
    fig.savefig(args.output / "training_validation.pdf")
    plt.close(fig)
    counts = {}
    for event in updates:
        for sample in event["samples"]:
            counts[sample["episode"]] = counts.get(sample["episode"], 0) + 1
    (args.output / "training_summary.json").write_text(json.dumps({"updates": len(updates),
        "unique_sampled_training_parents": len(counts), "sample_counts": counts,
        "initial_validation_probe_loss": checks[0]["scores"]["validation"]["mean"],
        "final_validation_probe_loss": checks[-1]["scores"]["validation"]["mean"],
        "stop_reason": report["stop_reason"], "limits": report["limits"]}, indent=2))


if __name__ == "__main__":
    main()
