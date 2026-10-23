"""Read-only forward kinematics of saved trajectories; no rollout or control."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.smolvla_data import sha256
from learning.vla.runtime_io import write_json
from learning.vla.source_archive import archive_source
from scripts.rollout_smolvla import verify_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnosis", type=Path, required=True)
    args = p.parse_args()
    destination = args.diagnosis / "alignment.json"
    if destination.exists():
        raise ValueError("Preserve original alignment measurement")
    snapshot = ROOT / "third_party/capstone_automateData"
    sys.path.insert(0, str(snapshot))
    from envs.openarm_insert import OpenArmInsertEnv
    import mujoco
    rows = []
    for name in ("train_20002", "train_20004"):
        source = ROOT / "data/drive_v2c30_20261012/merged_all_v1" / name
        meta = json.loads((source / "metadata.json").read_text())
        env = OpenArmInsertEnv(images=False, workspace=verify_snapshot(snapshot, meta))
        env.reset(seed=meta["seed"], options=meta["reset_options"])
        kinematic = mujoco.MjData(env.model)
        trajectory = args.diagnosis / name / "recorded_observations_trajectory.npz"
        with np.load(source / "episode.npz", allow_pickle=False) as data:
            reference, phases = data["qpos"].copy(), data["phase"].copy()
        with np.load(trajectory, allow_pickle=False) as data:
            qpos = data["qpos"].copy(); states = data["observation_state"].copy()
        measurements = []
        for index in (100, 200, 300, len(qpos)-1):
            poses = []
            for q in (qpos[index], reference[index]):
                kinematic.qpos[:] = q
                mujoco.mj_forward(env.model, kinematic)
                poses.append(kinematic.site_xpos[env.grasp_site].copy())
            error = (poses[0]-poses[1])*1000
            measurements.append({"command": index, "assisted_minus_expert_tcp_xyz_mm": error.tolist(),
                "tcp_distance_mm": float(np.linalg.norm(error))})
        diagnostics = json.loads((args.diagnosis / name / "recorded_observations_diagnostics.json").read_text())
        rows.append({"episode": name, "trajectory_sha256": sha256(trajectory),
            "measurements": measurements, "final_expert_phase": str(phases[len(qpos)-2]),
            "final_gripper_mm": float(states[-1, 15])*1000,
            "final_support_contact_n": diagnostics[-1]["robot_support_contact_n"],
            "peak_support_contact_n": max(d.get("robot_support_contact_peak_n", d["robot_support_contact_n"]) for d in diagnostics)})
        env.close()
    write_json(destination, {"rows": rows, "implementation": archive_source("scripts/measure_pickup_alignment.py"),
        "method": "mj_forward on separate MjData using original saved qpos; no policy execution, state forcing during rollout, or contact-force recomputation"})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
