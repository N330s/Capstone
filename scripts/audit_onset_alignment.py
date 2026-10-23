"""CPU-only audit of reset/first-active/terminal raw-to-export alignment."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", str(ROOT / "data/hf_cache"))
from learning.vla.runtime_io import write_json
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source
from learning.vla.onset_sampling import verify_index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous audit; a new output path is required")
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    index = json.loads(args.index.read_text())
    verify_index(index, args.collection, args.export)
    receipt_path = args.export / "openarm_export.json"
    receipt = json.loads(receipt_path.read_text())
    ds = LeRobotDataset(receipt["datasets"]["train"]["repo_id"], root=args.export / "train",
        delta_timestamps={"action": [i/50 for i in range(50)]}, video_backend="pyav")
    rows = []
    for parent in index["parents"]:
        name, n, offset = parent["episode"], parent["steps"], parent["global_start"]
        meta = json.loads((args.collection / name / "metadata.json").read_text())
        with np.load(args.collection / name / "episode.npz", allow_pickle=False) as raw:
            for step in (0, parent["excluded_initial_commands"], n-1):
                sample, valid = ds[offset+step], min(50, n-step)
                if (not np.array_equal(sample["observation.state"].numpy(), raw["observation_state"][step].astype(np.float32))
                        or not np.array_equal(sample["action"].numpy()[:valid], raw["action_applied"][step:step+valid].astype(np.float32))
                        or not np.array_equal(sample["action_is_pad"].numpy(), np.arange(50) >= valid)
                        or sample["task"] != meta["instruction"]):
                    raise ValueError(f"State/action/terminal/prompt mismatch at {name}:{step}")
                rows.append({"episode": name, "step": step, "global_index": offset+step, "valid_commands": valid})
    result = {"passed": True, "parents": len(index["parents"]), "anchors_checked": len(rows), "rows": rows,
        "index_sha256": sha256(args.index), "export_receipt_sha256": sha256(receipt_path),
        "implementation": archive_source("scripts/audit_onset_alignment.py"),
        "checks": ["state at original index", "future applied targets at original index", "terminal mask", "instruction"],
        "limits": "CPU boundary check, not bit-exact RGB encoding or autonomous pickup evidence."}
    write_json(args.output, result)
    print(json.dumps({k:result[k] for k in ("passed", "parents", "anchors_checked")}))


if __name__ == "__main__":
    main()
