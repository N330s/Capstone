"""Signed one-axis sweep, with optional Cartesian combinations."""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from connector.simulation import CONFIG_PATH, ROOT, VERSION, Pose, run_trial


def numbers(value):
    try:
        result = [float(v) for v in value.split(",")]
        if not result:
            raise ValueError
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers") from exc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("offset-y-mm", "offset-z-mm"):
        p.add_argument("--" + name, type=numbers, default=[-3, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 3])
    for name in ("roll-deg", "pitch-deg", "yaw-deg"):
        p.add_argument("--" + name, type=numbers, default=[-10, -5, -2, -1, 0, 1, 2, 5, 10])
    p.add_argument("--mode", choices=("axis", "grid"), default="axis")
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument("--output", type=Path, default=ROOT / "results" / VERSION / "sweep")
    args = p.parse_args()
    keys = list(Pose.__dataclass_fields__)
    values = [getattr(args, key) for key in keys]
    if args.mode == "grid":
        cases = list(itertools.product(*values))
    else:
        cases = {(0,) * len(keys)}
        for index, dimension in enumerate(values):
            for value in dimension:
                case = [0] * len(keys)
                case[index] = value
                cases.add(tuple(case))
        cases = sorted(cases)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = []
    for index, case in enumerate(cases):
        result = run_trial(Pose(**dict(zip(keys, case))), config=config,
                           output=args.output / f"trial_{index:03d}")
        first = result["first_contact"] or {}
        row = {key: result[key] for key in keys}
        row.update({key: result[key] for key in (
            "success", "outcome", "max_insertion_depth_m", "max_contact_force_n", "max_penetration_m")})
        row.update({f"first_contact_{key}": first.get(key) for key in
                    ("offset_y_m", "offset_z_m", "orientation_error_deg")})
        rows.append(row)
        print(f"{index + 1}/{len(cases)} {case}: {result['outcome']}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "alignment_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} trials to {args.output}")


if __name__ == "__main__":
    main()
