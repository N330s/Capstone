"""Run one force-driven plug insertion trial.

Offsets are relative to the perfectly aligned initial pose.  During a trial the
plug is moved only by MuJoCo dynamics and ``xfrc_applied``; qpos is written only
while resetting the initial state.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "connector" / "plug_socket.xml"
BASE_PLUG_POSITION = np.array([-0.020, 0.0, 0.004])
SUCCESS_DEPTH_M = 0.025
SUCCESS_HOLD_S = 0.25
MAX_SUCCESS_ANGLE_DEG = 5.0


@dataclass
class TrialResult:
    success: bool
    max_insertion_depth_m: float
    max_contact_force_n: float
    final_orientation_error_deg: float
    final_lateral_error_m: float
    final_insertion_depth_m: float
    simulated_time_s: float
    contact_count: int
    stable: bool


def load_simulation() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    return model, data


def _quaternion_from_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Return MuJoCo wxyz quaternion for Rz(yaw) @ Ry(pitch)."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp])


def reset_plug(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    offset_y_mm: float = 0.0,
    offset_z_mm: float = 0.0,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
) -> None:
    mujoco.mj_resetData(model, data)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plug_free")
    qpos_adr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_adr : qpos_adr + 3] = BASE_PLUG_POSITION + np.array(
        [0.0, offset_y_mm * 1e-3, offset_z_mm * 1e-3]
    )
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = _quaternion_from_yaw_pitch(
        yaw_deg, pitch_deg
    )
    mujoco.mj_forward(model, data)


def diagnostics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plug_tip")
    entry_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "socket_entry")
    plug_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plug")

    tip = data.site_xpos[tip_id]
    entry = data.site_xpos[entry_id]
    plug_axis = data.xmat[plug_id].reshape(3, 3)[:, 0]
    angle = math.degrees(math.acos(float(np.clip(plug_axis[0], -1.0, 1.0))))

    total_contact_force = 0.0
    force = np.zeros(6)
    for contact_index in range(data.ncon):
        mujoco.mj_contactForce(model, data, contact_index, force)
        total_contact_force += float(np.linalg.norm(force[:3]))

    return {
        "insertion_depth_m": float(tip[0] - entry[0]),
        "lateral_error_m": float(np.linalg.norm(tip[1:3] - entry[1:3])),
        "orientation_error_deg": angle,
        "contact_force_n": total_contact_force,
        "contact_count": int(data.ncon),
    }


def run_trial(
    *,
    offset_y_mm: float = 0.0,
    offset_z_mm: float = 0.0,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    push_force_n: float = 0.15,
    duration_s: float = 1.0,
) -> TrialResult:
    model, data = load_simulation()
    reset_plug(model, data, offset_y_mm, offset_z_mm, yaw_deg, pitch_deg)
    plug_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plug")

    max_depth = -math.inf
    max_force = 0.0
    hold_time = 0.0
    stable = True
    last = diagnostics(model, data)

    for _ in range(math.ceil(duration_s / model.opt.timestep)):
        data.xfrc_applied[plug_id, :3] = (push_force_n, 0.0, 0.0)
        mujoco.mj_step(model, data)
        last = diagnostics(model, data)
        max_depth = max(max_depth, last["insertion_depth_m"])
        max_force = max(max_force, last["contact_force_n"])

        finite = np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        if not finite:
            stable = False
            break

        valid_pose = (
            last["insertion_depth_m"] > SUCCESS_DEPTH_M
            and last["orientation_error_deg"] < MAX_SUCCESS_ANGLE_DEG
        )
        hold_time = hold_time + model.opt.timestep if valid_pose else 0.0
        if hold_time >= SUCCESS_HOLD_S:
            break

    return TrialResult(
        success=stable and hold_time >= SUCCESS_HOLD_S,
        max_insertion_depth_m=max_depth,
        max_contact_force_n=max_force,
        final_orientation_error_deg=last["orientation_error_deg"],
        final_lateral_error_m=last["lateral_error_m"],
        final_insertion_depth_m=last["insertion_depth_m"],
        simulated_time_s=float(data.time),
        contact_count=last["contact_count"],
        stable=stable,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset-y-mm", type=float, default=0.0)
    parser.add_argument("--offset-z-mm", type=float, default=0.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--push-force-n", type=float, default=0.15)
    parser.add_argument("--duration-s", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_trial(**vars(args))
    print(json.dumps(asdict(result), indent=2))
    raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()
