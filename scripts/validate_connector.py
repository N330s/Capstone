"""Save reproducibility and timestep evidence for the straight-slot baseline."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from connector.simulation import ConnectorSimulation, Pose, ROOT, VERSION, run_trial


def main():
    aligned = [run_trial() for _ in range(20)]
    checks = {
        "20_aligned_successes": all(r["success"] for r in aligned),
        "aligned_identical": all(r == aligned[0] for r in aligned[1:]),
    }
    comparisons = []
    for pose in (Pose(), Pose(offset_y_mm=1), Pose(offset_z_mm=-1),
                 Pose(roll_deg=5), Pose(pitch_deg=5), Pose(yaw_deg=2)):
        normal = run_trial(pose)
        fine = run_trial(pose, timestep=0.00025)
        comparison = {
            "pose": {k: getattr(pose, k) for k in Pose.__dataclass_fields__},
            "normal_outcome": normal["outcome"], "half_step_outcome": fine["outcome"],
            "depth_difference_m": abs(normal["final"]["insertion_depth_m"] -
                                      fine["final"]["insertion_depth_m"]),
            "peak_force_difference_n": abs(normal["max_contact_force_n"] -
                                          fine["max_contact_force_n"]),
        }
        comparisons.append(comparison)
    checks["representative_timestep_outcomes_match"] = all(
        r["normal_outcome"] == r["half_step_outcome"] for r in comparisons)
    checks["timestep_depth_difference_below_0_1mm"] = all(
        r["depth_difference_m"] < 0.0001 for r in comparisons)
    checks["timestep_peak_force_difference_below_0_1N"] = all(
        r["peak_force_difference_n"] < 0.1 for r in comparisons)
    report = {"checks": checks, "aligned_reference": aligned[0],
              "timestep_comparisons": comparisons,
              "metadata": ConnectorSimulation().metadata()}
    path = ROOT / "results" / VERSION / "validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(path)
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()

