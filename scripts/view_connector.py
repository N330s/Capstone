"""View the shared holder test. Space: advance/pause; R: reset.
Y/G: lateral +/-0.25 mm; Z/X: vertical +/-0.25 mm; A/D: yaw +/-1 deg;
W/S: pitch +/-1 deg; E/Q: roll +/-1 deg. C: toggle collision transparency.
Mouse: left-drag orbit, right-drag pan, scroll zoom. V: toggle the fixed overview camera.
Terminal diagnostics use the same success/abort rules as the headless test.
``--plug-type type_o`` (and ``--socket``, ``--leadin``) show a catalog connector instead of the
legacy two-blade pair.
"""
from __future__ import annotations
import argparse
import queue
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import mujoco.viewer
from connector import catalog
from connector.simulation import ConnectorSimulation, Pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leadin", action="store_true", help="use the 1 mm chamfered lead-in socket")
    catalog.add_cli_arguments(parser)
    args = parser.parse_args()
    sim = ConnectorSimulation(leadin=args.leadin, spec=catalog.from_cli(args))
    print("connector:", sim.metadata()["scene_version"])
    pose = {key: 0.0 for key in Pose.__dataclass_fields__}
    events = queue.SimpleQueue()
    advancing = False
    bindings = {
        "Y": ("offset_y_mm", 0.25), "G": ("offset_y_mm", -0.25),
        "Z": ("offset_z_mm", 0.25), "X": ("offset_z_mm", -0.25),
        "A": ("yaw_deg", 1), "D": ("yaw_deg", -1),
        "W": ("pitch_deg", 1), "S": ("pitch_deg", -1),
        "E": ("roll_deg", 1), "Q": ("roll_deg", -1),
    }
    print(__doc__)
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=events.put) as viewer:
        def free_camera():
            # Free camera (mouse orbit/pan/zoom) framing the socket mouth from the plug side.
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = (-0.012, 0.0, 0.10)
            viewer.cam.distance = 0.18
            viewer.cam.azimuth = -40
            viewer.cam.elevation = -20
        free_camera()
        viewer.opt.sitegroup[3] = 0
        next_report = 0.0
        while viewer.is_running():
            start = time.perf_counter()
            with viewer.lock():
                while not events.empty():
                    code = events.get()
                    key = chr(code).upper() if 0 <= code < 256 else ""
                    if key == " ":
                        advancing = not advancing
                    if key in bindings:
                        field, increment = bindings[key]
                        pose[field] += increment
                    if key == "R" or key in bindings:
                        sim.reset(Pose(**pose))
                        advancing = False
                        next_report = 0
                        print("reset:", pose)
                    if key == "C":
                        flag = mujoco.mjtVisFlag.mjVIS_TRANSPARENT
                        viewer.opt.flags[flag] = not viewer.opt.flags[flag]
                    if key == "V":
                        if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
                            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                            viewer.cam.fixedcamid = sim.model.camera("overview").id
                        else:
                            free_camera()
                # Batch physics at 60 viewer frames/s; do not sync the GUI at 2 kHz.
                for _ in range(max(1, round(1 / (60 * sim.model.opt.timestep)))):
                    sim.step(advancing)
                if sim.data.time >= next_report:
                    i = sim.info
                    depths = " ".join(f"{name}={i[name + '_depth_m']*1000:.2f}" for name in sim.element_names)
                    print(f"depth={i['insertion_depth_m']*1000:.2f} mm ({depths}) "
                          f"angle={i['orientation_error_deg']:.2f} deg "
                          f"force={i['contact_force_n']:.3f} N "
                          f"outcome={sim.outcome or 'running'}")
                    next_report = sim.data.time + 0.25
            viewer.sync()
            time.sleep(max(0, 1/60 - (time.perf_counter() - start)))


if __name__ == "__main__":
    main()
