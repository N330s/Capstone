"""Audit train-only pre-close finger-state support; never constructs new labels."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.smolvla_data import sha256
from learning.vla.runtime_io import write_json
from learning.vla.source_archive import archive_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous coverage audit")
    receipt_path = ROOT / "data/drive_v2c30_20261012/export_all_v1/openarm_export.json"
    receipt = json.loads(receipt_path.read_text())
    root = ROOT / "data/drive_v2c30_20261012/merged_all_v1"
    rows = []
    started = time.monotonic()
    for record in receipt["datasets"]["train"]["episodes"]:
        source = root / record["episode"] / "episode.npz"
        if time.monotonic()-started > 120:
            raise TimeoutError("Two-minute bounded coverage audit exhausted")
        if sha256(source) != record["data_sha256"]:
            raise ValueError("Training parent differs from immutable export receipt")
        with np.load(source, allow_pickle=False) as raw:
            close = np.flatnonzero(raw["phase"] == "close")
            if not len(close):
                raise ValueError("Nominal successful parent lacks a close phase")
            n = int(close[0])
            finger = raw["observation_state"][:n, 15]*1000
            rows.append({"episode": record["episode"], "source_sha256": record["data_sha256"],
                "preclose_frames": n, "below34mm": int((finger<34).sum()), "below30mm": int((finger<30).sum()),
                "minimum_mm": float(finger.min()), "maximum_mm": float(finger.max())})
    write_json(args.output, {"completed": True, "parents": len(rows),
        "preclose_frames": sum(r["preclose_frames"] for r in rows),
        "below34mm": sum(r["below34mm"] for r in rows), "below30mm": sum(r["below30mm"] for r in rows),
        "minimum_mm": min(r["minimum_mm"] for r in rows), "maximum_mm": max(r["maximum_mm"] for r in rows),
        "rows": rows, "receipt_sha256": sha256(receipt_path), "elapsed_s": time.monotonic()-started,
        "implementation": archive_source("scripts/audit_pickup_state_coverage.py"),
        "limits": "Original 81 nominal training parents before their first close command only. This does not audit unexported recovery collections or establish that all model failures share one cause."})


if __name__ == "__main__":
    main()
