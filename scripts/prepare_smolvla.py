"""Validate raw episodes; optionally export local, split-separated LeRobot datasets."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.vla.smolvla_data import SmolVLASource, export_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--export", type=Path, help="New output directory; omitted means validation only")
    parser.add_argument("--repo-prefix", default="local/openarm_smolvla")
    parser.add_argument("--selection", type=Path, help="Explicit episode split/subset JSON; raw manifest stays unchanged")
    args = parser.parse_args()
    source = SmolVLASource(args.dataset, selection=args.selection)
    report = export_source(source, args.export, args.repo_prefix) if args.export else source.report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
