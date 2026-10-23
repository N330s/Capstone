"""Verify real recovery-segment LeRobot boundaries, provenance and actor fields."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", str(ROOT / "data/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data/hf_cache/datasets"))
sys.path.insert(0, str(ROOT))
from learning.vla.recovery_data import RecoveryVLASource, RECOVERY_CONTRACT
from scripts.verify_smolvla_export import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    receipt = json.loads((args.export / "openarm_export.json").read_text())
    if receipt["contract"] != RECOVERY_CONTRACT:
        raise ValueError("Recovery-segment contract required")
    source = RecoveryVLASource(args.view)
    report = verify(source, args.export)
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from learning.vla.smolvla_data import CAMERAS
    segments, eligible = [], 0
    for split in ("train", "validation"):
        info = json.loads((args.export / split / "meta/info.json").read_text())
        actor_features = {key for key in info["features"] if key.startswith("observation.")}
        if actor_features != {"observation.state", *CAMERAS}:
            raise ValueError("Unexpected actor feature: depth/torque/labels must remain auxiliary")
        ds = LeRobotDataset(receipt["datasets"][split]["repo_id"], root=args.export / split,
                            delta_timestamps={"action": [i / 50 for i in range(50)]}, video_backend="pyav")
        offset = 0
        for record in [r for r in source.records if r["split"] == split]:
            with np.load(source.root / record["parent_episode"] / "episode.npz", allow_pickle=False) as arrays:
                labels = arrays["step_label"][record["parent_start"]:record["parent_stop_exclusive"]]
            if len(labels) != record["steps"] or np.any(labels == "fault"):
                raise ValueError("Fault target or shifted timeline inside exported segment")
            actual = ds[offset + record["steps"] - 1]
            if int((~actual["action_is_pad"]).sum()) != 1:
                raise ValueError("Chunk crosses a segment boundary")
            segments.append({**record, "last_anchor_valid_targets": 1, "fault_targets": 0})
            eligible += record["steps"]
            offset += record["steps"]
    report.update(contract=RECOVERY_CONTRACT, actor_inputs="two RGB cameras + 16 positions only",
                  eligible_commands=eligible, excluded_fault_commands=receipt["excluded_fault_commands"],
                  segment_provenance=segments, partial_release=receipt["partial_release"],
                  limits="Export readback and fault-boundary checks; not autonomous simulation success")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "segments": len(segments), "eligible_commands": eligible,
                      "excluded_fault_commands": receipt["excluded_fault_commands"]}), flush=True)


if __name__ == "__main__":
    main()
