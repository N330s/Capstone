"""Run repeatable plug-alignment sweeps and write a CSV result table."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from test_insert import ROOT, run_trial


DEFAULT_OUTPUT = ROOT / "results" / "alignment_sweep.csv"


def comma_separated_floats(value: str) -> list[float]:
    try:
        return [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset-y-mm", type=comma_separated_floats,
                        default=[0.0, 0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--offset-z-mm", type=comma_separated_floats,
                        default=[0.0, 0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--yaw-deg", type=comma_separated_floats,
                        default=[0.0, 2.0, 5.0, 10.0])
    parser.add_argument("--pitch-deg", type=comma_separated_floats,
                        default=[0.0, 2.0, 5.0, 10.0])
    parser.add_argument("--push-force-n", type=comma_separated_floats,
                        default=[0.15])
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--mode", choices=("axis", "grid"), default="axis",
                        help="axis varies one parameter at a time; grid uses the Cartesian product")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def axis_cases(args: argparse.Namespace) -> list[tuple[float, float, float, float, float]]:
    baseline = (0.0, 0.0, 0.0, 0.0, args.push_force_n[0])
    cases = {baseline}
    cases.update((v, 0.0, 0.0, 0.0, args.push_force_n[0]) for v in args.offset_y_mm)
    cases.update((0.0, v, 0.0, 0.0, args.push_force_n[0]) for v in args.offset_z_mm)
    cases.update((0.0, 0.0, v, 0.0, args.push_force_n[0]) for v in args.yaw_deg)
    cases.update((0.0, 0.0, 0.0, v, args.push_force_n[0]) for v in args.pitch_deg)
    cases.update((0.0, 0.0, 0.0, 0.0, v) for v in args.push_force_n)
    return sorted(cases)


def main() -> None:
    args = parse_args()
    if args.mode == "grid":
        cases = list(itertools.product(
            args.offset_y_mm,
            args.offset_z_mm,
            args.yaw_deg,
            args.pitch_deg,
            args.push_force_n,
        ))
    else:
        cases = axis_cases(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "offset_y_mm",
        "offset_z_mm",
        "yaw_deg",
        "pitch_deg",
        "push_force_N",
        "success",
        "max_insertion_depth",
        "max_contact_force",
        "final_orientation_deg",
        "final_lateral_error_m",
        "stable",
    ]

    rows = []
    for index, (offset_y, offset_z, yaw, pitch, force) in enumerate(cases, start=1):
        result = run_trial(
            offset_y_mm=offset_y,
            offset_z_mm=offset_z,
            yaw_deg=yaw,
            pitch_deg=pitch,
            push_force_n=force,
            duration_s=args.duration_s,
        )
        rows.append({
            "offset_y_mm": offset_y,
            "offset_z_mm": offset_z,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "push_force_N": force,
            "success": result.success,
            "max_insertion_depth": result.max_insertion_depth_m,
            "max_contact_force": result.max_contact_force_n,
            "final_orientation_deg": result.final_orientation_error_deg,
            "final_lateral_error_m": result.final_lateral_error_m,
            "stable": result.stable,
        })
        print(
            f"[{index:>3}/{len(cases)}] y={offset_y:g} mm z={offset_z:g} mm "
            f"yaw={yaw:g}° pitch={pitch:g}° force={force:g} N "
            f"success={result.success} depth={result.max_insertion_depth_m * 1e3:.2f} mm"
        )

    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} trials to {args.output}")


if __name__ == "__main__":
    main()
