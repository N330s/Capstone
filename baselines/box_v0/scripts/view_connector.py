"""Interactive viewer for the standalone connector simulation.

Keys:
  Space   toggle insertion force
  R       reset with the currently selected initial pose
  Y/G     lateral offset +/−0.5 mm
  Z/X     vertical offset +/−0.5 mm
  A/D     yaw +/−1 degree
  W/S     pitch +/−1 degree
  Esc     close the viewer
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from test_insert import (
    MAX_SUCCESS_ANGLE_DEG,
    SUCCESS_DEPTH_M,
    SUCCESS_HOLD_S,
    diagnostics,
    load_simulation,
    reset_plug,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset-y-mm", type=float, default=0.0)
    parser.add_argument("--offset-z-mm", type=float, default=0.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--push-force-n", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, data = load_simulation()
    plug_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plug")
    pose = {
        "offset_y_mm": args.offset_y_mm,
        "offset_z_mm": args.offset_z_mm,
        "yaw_deg": args.yaw_deg,
        "pitch_deg": args.pitch_deg,
    }
    pushing = False
    reset_requested = True
    success_hold_s = 0.0

    def key_callback(keycode: int) -> None:
        nonlocal pushing, reset_requested
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        if key == " ":
            pushing = not pushing
        elif key == "R":
            reset_requested = True
        elif key == "Y":
            pose["offset_y_mm"] += 0.5
            reset_requested = True
        elif key == "G":
            pose["offset_y_mm"] -= 0.5
            reset_requested = True
        elif key == "Z":
            pose["offset_z_mm"] += 0.5
            reset_requested = True
        elif key == "X":
            pose["offset_z_mm"] -= 0.5
            reset_requested = True
        elif key == "A":
            pose["yaw_deg"] += 1.0
            reset_requested = True
        elif key == "D":
            pose["yaw_deg"] -= 1.0
            reset_requested = True
        elif key == "W":
            pose["pitch_deg"] += 1.0
            reset_requested = True
        elif key == "S":
            pose["pitch_deg"] -= 1.0
            reset_requested = True

    print(__doc__)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.fixedcamid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "overview"
        )
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        next_report = 0.0
        while viewer.is_running():
            frame_start = time.perf_counter()
            if reset_requested:
                reset_plug(model, data, **pose)
                pushing = False
                reset_requested = False
                success_hold_s = 0.0
                next_report = 0.0
                print(f"reset: {pose}")

            data.xfrc_applied[plug_id, :3] = (
                (args.push_force_n, 0.0, 0.0) if pushing else (0.0, 0.0, 0.0)
            )
            mujoco.mj_step(model, data)
            info = diagnostics(model, data)
            valid_pose = (
                info["insertion_depth_m"] > SUCCESS_DEPTH_M
                and info["orientation_error_deg"] < MAX_SUCCESS_ANGLE_DEG
            )
            success_hold_s = success_hold_s + model.opt.timestep if valid_pose else 0.0

            if data.time >= next_report:
                success = success_hold_s >= SUCCESS_HOLD_S
                print(
                    f"depth={info['insertion_depth_m'] * 1e3:6.2f} mm  "
                    f"angle={info['orientation_error_deg']:5.2f} deg  "
                    f"force={info['contact_force_n']:7.3f} N  "
                    f"success={success}  pushing={pushing}"
                )
                next_report = data.time + 0.1

            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
