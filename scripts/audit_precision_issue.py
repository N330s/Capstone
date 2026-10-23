"""Read-only checkpoint update audit; archive old training source before repair."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from safetensors import safe_open
    from learning.vla.runtime_io import write_json
    from learning.vla.source_archive import archive_source
    from scripts.train_gripper_feedback import BASE
    torch.set_num_threads(4)
    overlay = ROOT / "outputs/gripper_control_fix_20261019/paired_feedback/nominal/expert/expert.safetensors"
    counts = defaultdict(lambda: {"total": 0, "changed": 0})
    with safe_open(str(BASE / "model.safetensors"), framework="pt", device="cpu") as old, safe_open(str(overlay), framework="pt", device="cpu") as new:
        for key in new.keys():
            a, b = old.get_tensor(key), new.get_tensor(key)
            row = counts[str(b.dtype)]
            row["total"] += b.numel()
            row["changed"] += int((a != b).sum())
    for row in counts.values():
        row["changed_fraction"] = row["changed"] / row["total"]
    sources = [archive_source(path) for path in (
        "scripts/finetune_smolvla.py", "scripts/finetune_recovery_smolvla.py",
        "scripts/overfit_smolvla_episode.py", "scripts/continue_smolvla_until.py",
        "scripts/qualify_smolvla_training.py", "scripts/smoke_smolvla_training.py")]
    qualification = json.loads((ROOT / "outputs/gripper_control_fix_20261019/fp32_qualification/report.json").read_text())
    write_json(args.output, {"old_800_update_control": dict(counts),
        "real_gradient_gpu_audit": qualification["precision_update_audit"],
        "fp32_qualification_updates": qualification["updates"], "archived_sources": sources,
        "interrupted_augmented_updates": 94,
        "interpretation": "Direct BF16 parameter/moment storage loses small AdamW updates. Finite losses/gradients alone did not qualify effective learning.",
        "limits": "This identifies a training precision defect, not proof it is the only cause of task failure."})
    print(json.dumps({"counts": dict(counts), "sources_archived": len(sources)}), flush=True)


if __name__ == "__main__":
    main()
