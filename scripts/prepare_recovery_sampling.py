"""Build a source-bound correction sampling index; never modifies raw recordings."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.vla.recovery_sampling import build_index
from learning.vla.runtime_io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("view", "export", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index = build_index(args.view, args.export)
    write_json(args.output, index)
    print(json.dumps({"output": str(args.output), "parent_splits": index["parent_splits"],
        "counts": {s: {k: len(v) for k, v in p.items()} for s, p in index["partitions"].items()}}), flush=True)


if __name__ == "__main__":
    main()
