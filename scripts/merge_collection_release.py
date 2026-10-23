"""Create a new hard-linked collection view and explicit episode-level holdout.

No source manifests are edited. Scene/controller compatibility is checked by the
existing strict adapter. Failed-preflight scenes and the evaluation bank are not
behavior-cloning examples. Evaluation seed namespace is rejected.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.vla.smolvla_data import SmolVLASource, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < .5:
        raise ValueError("Validation fraction must be between zero and .5")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("New merged collection destination required")
    records, origins, names, instructions = [], [], set(), set()
    for root in args.sources:
        root = root.resolve()
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Source and destination must be disjoint")
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema") != "openarm-pilot-v1" or not manifest.get("complete"):
            raise ValueError("Expected complete collector manifest")
        instructions.add(manifest["instruction"])
        origins.append({"root": str(root), "manifest_sha256": sha256(root / "manifest.json")})
        for record in manifest["episodes"]:
            name = record["episode"]
            path = (root / name).resolve()
            if path.parent != root or name in names:
                raise ValueError("Unsafe or duplicate episode name")
            if record.get("seed", 0) >= 9000000 or record["split"] not in ("train", "validation"):
                raise ValueError("Do not train on evaluation conditions")
            names.add(name)
            records.append((root, record))
    if len(instructions) != 1 or len(records) < 5:
        raise ValueError("One instruction and >=5 episodes required")
    supplied_validation = [r["episode"] for _, r in records if r["split"] == "validation"]
    if supplied_validation:
        validation = set(supplied_validation)
        rule = "Preserve all source episode splits"
    else:
        # Stable hash ranking avoids making 'last episode' the validation criterion.
        ranked = sorted(names, key=lambda n: hashlib.sha256(("openarm-holdout-v1:" + n).encode()).hexdigest())
        count = max(1, round(len(records) * args.validation_fraction))
        validation = set(ranked[:count])
        rule = "All source episodes labeled train; explicit deterministic SHA256-ranked 20% episode holdout"
    output.mkdir(parents=True)
    merged = []
    for root, record in records:
        episode_out = output / record["episode"]
        episode_out.mkdir()
        for name in ("episode.npz", "metadata.json", "diagnostics.json"):
            source = root / record["episode"] / name
            if source.is_file():
                os.link(source, episode_out / name)
        merged.append({**record, "split": "validation" if record["episode"] in validation else "train"})
    manifest = {"schema": "openarm-pilot-v1", "instruction": instructions.pop(), "complete": True,
                "episodes": merged, "origins": origins, "split_rule": rule,
                "source_files": "Hard links to immutable recordings; do not edit either view"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    source = SmolVLASource(output)
    report = {**source.report(), "origins": origins, "split_rule": rule}
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
