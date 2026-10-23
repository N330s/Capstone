"""Plan parent-level holdouts and prepare an explicit recovery-aware data view."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "data/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(__file__).resolve().parents[1] / "data/hf_cache/datasets"))
from learning.vla.recovery_data import make_split_plan, prepare_view, RecoveryVLASource
from learning.vla.smolvla_data import export_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+")
    parser.add_argument("--view", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.manifests:
        if args.plan.exists():
            raise FileExistsError("Existing split plans are immutable")
        plan = make_split_plan(args.manifests)
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    else:
        plan = json.loads(args.plan.read_text())
    print(json.dumps({"planned_parents": len(plan["episodes"]),
                      "splits": {s: [r["episode"] for r in plan["episodes"] if r["split"] == s]
                                 for s in ("train", "validation")}}), flush=True)
    if args.view:
        if not args.roots:
            raise ValueError("Collection roots required")
        prepare_view(plan, args.roots, args.view, allow_partial=args.allow_partial)
        source = RecoveryVLASource(args.view)
        (args.view / "recovery_validation.json").write_text(json.dumps(source.report(), indent=2), encoding="utf-8")
        print(json.dumps(source.report()), flush=True)
        if args.export:
            export_source(source, args.export, "local/openarm_recovery")
            print(json.dumps({"export_completed": str(args.export)}), flush=True)
    elif args.export:
        raise ValueError("Export requires a newly prepared view")


if __name__ == "__main__":
    main()
