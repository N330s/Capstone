"""View the shared holder test. Space: advance/pause; R: reset.
Y/G: lateral +/-0.25 mm; Z/X: vertical +/-0.25 mm; A/D: yaw +/-1 deg;
W/S: pitch +/-1 deg; E/Q: roll +/-1 deg. C: toggle collision transparency.
Terminal diagnostics use the same success/abort rules as the headless test.
"""
from __future__ import annotations
import queue
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import mujoco.viewer
from connector.simulation import ConnectorSimulation, Pose


def main():
    sim = ConnectorSimulation()
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
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = sim.model.camera("overview").id
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
                # Batch physics at 60 viewer frames/s; do not sync the GUI at 2 kHz.
                for _ in range(max(1, round(1 / (60 * sim.model.opt.timestep)))):
                    sim.step(advancing)
                if sim.data.time >= next_report:
                    i = sim.info
                    print(f"depth={i['insertion_depth_m']*1000:.2f} mm "
                          f"angle={i['orientation_error_deg']:.2f} deg "
                          f"force={i['contact_force_n']:.3f} N "
                          f"outcome={sim.outcome or 'running'}")
                    next_report = sim.data.time + 0.25
            viewer.sync()
            time.sleep(max(0, 1/60 - (time.perf_counter() - start)))


if __name__ == "__main__":
    main()
