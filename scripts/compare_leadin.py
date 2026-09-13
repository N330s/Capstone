"""Matched straight/lead-in gate, including combined errors and repeated seating."""
import itertools
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from connector.simulation import Pose, ROOT, run_trial

def main():
    root = ROOT / "results/lead_in_comparison"
    root.mkdir(parents=True, exist_ok=True)
    cases = {Pose()}
    for field in Pose.__dataclass_fields__:
        values = (-3, -1, -.5, -.25, .25, .5, 1, 3) if "offset" in field else (-10, -5, -2, -1, 1, 2, 5, 10)
        cases.update(Pose(**{field: v}) for v in values)
    cases.update(Pose(offset_y_mm=y, offset_z_mm=z, yaw_deg=a)
                 for y, z, a in itertools.product((-.5, .5), (-.5, .5), (-1, 1)))
    rows = []
    for i, pose in enumerate(sorted(cases, key=repr)):
        a = run_trial(pose, output=root / f"case_{i:03d}/straight")
        b = run_trial(pose, leadin=True, output=root / f"case_{i:03d}/leadin")
        rows.append({"pose": vars(pose), "straight": a["outcome"], "leadin": b["outcome"],
                     "peak_penetration_m": b["max_penetration_m"], "peak_force_n": b["max_contact_force_n"]})
        print(i+1, a["outcome"], b["outcome"], flush=True)
    repeats = [run_trial(leadin=True) for _ in range(20)]
    half = []
    for pose in (Pose(), Pose(offset_y_mm=.5), Pose(offset_z_mm=-.5), Pose(yaw_deg=5)):
        a, b = run_trial(pose, leadin=True), run_trial(pose, leadin=True, timestep=.00025)
        half.append({"pose": vars(pose), "outcomes_match": a["outcome"] == b["outcome"],
                     "depth_delta_m": abs(a["final"]["insertion_depth_m"] - b["final"]["insertion_depth_m"]),
                     "force_delta_n": abs(a["max_contact_force_n"] - b["max_contact_force_n"])})
    checks = {"20_repeatable_aligned": all(r == repeats[0] and r["success"] for r in repeats),
              "no_numerical_or_contact_aborts": all(r["leadin"] in ("success", "jam", "timeout") for r in rows),
              "some_improvement": any(r["straight"] == "jam" and r["leadin"] == "success" for r in rows),
              "failure_region_remains": any(r["leadin"] == "jam" for r in rows),
              "timestep_consistency": all(r["outcomes_match"] and r["depth_delta_m"] < .0001 and r["force_delta_n"] < .1 for r in half)}
    (root / "report.json").write_text(json.dumps({"checks": checks, "cases": rows, "half_timestep": half}, indent=2))
    print(checks, flush=True)
    raise SystemExit(0 if all(checks.values()) else 1)

if __name__ == "__main__":
    main()

