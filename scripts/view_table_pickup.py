"""Downward-rest table task viewer. --play runs the physical pickup probe once.
V: toggle free camera (mouse orbit/pan/zoom) vs the fixed scene camera."""
import argparse
import json
from pathlib import Path
import queue
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import mujoco.viewer
from envs.openarm_insert import OpenArmInsertEnv
from controllers.table_pickup import PickupProbe
from scripts.preview_table_task import aim_camera


def main(play=None,workspace=None):
    if play is None:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--play',action='store_true')
        parser.add_argument('--workspace',type=Path,default=None,help='workspace spec (default configs/workspace_v1.json)')
        args=parser.parse_args()
        play,workspace=args.play,args.workspace
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/table_task_v3.json').read_text())
    env=OpenArmInsertEnv(images=False,workspace=workspace)
    probe=PickupProbe(env)
    # The v1 table-task config carries the legacy spawn/closure numbers; other workspaces use theirs.
    position=cfg['plug_mating_position_m'] if workspace is None else env.spawn_position
    closed=cfg['closed_finger_target_m'] if workspace is None else env.pickup_travel
    try:
        probe.reset(position)
        # Viewer overview includes both hanging arms and the tabletop.
        aim_camera(env.model,'scene_rgb',[.90,-1.10,.85],[.20,0,.32],55)
        keys=queue.SimpleQueue()
        with mujoco.viewer.launch_passive(env.model,env.data,key_callback=keys.put) as viewer:
            with viewer.lock():
                viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid=env.model.camera('scene_rgb').id
                viewer.opt.sitegroup[:]=0
            def handle_keys():
                while not keys.empty():
                    if keys.get() in (86,118):
                        if viewer.cam.type==mujoco.mjtCamera.mjCAMERA_FIXED:
                            viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
                            viewer.cam.lookat[:]=env.data.xpos[env.plug]
                            viewer.cam.distance=0.8;viewer.cam.azimuth=150;viewer.cam.elevation=-20
                        else:
                            viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
                            viewer.cam.fixedcamid=env.model.camera('scene_rgb').id
            viewer.sync()
            if play:
                def update(phase):
                    if not viewer.is_running(): raise RuntimeError('Viewer closed')
                    with viewer.lock(): handle_keys()
                    viewer.sync()
                    time.sleep(.1)
                print(probe.run(position=position,pitch=cfg['tool_pitch_deg'],
                    height=cfg['grasp_height_offset_m'],closed_travel=closed,frame_callback=update))
            # Idle: keep physics running at real-time pace so the cable and plug stay live.
            while viewer.is_running():
                start=time.perf_counter()
                with viewer.lock():
                    handle_keys()
                    for _ in range(env.substeps): probe.physics_step()
                viewer.sync()
                time.sleep(max(0,env.dt-(time.perf_counter()-start)))
    finally:
        env.close()


if __name__=='__main__':main()
