"""Downward-rest table task viewer. --play runs the physical pickup probe once."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import mujoco.viewer
from envs.openarm_insert import OpenArmInsertEnv
from controllers.table_pickup import PickupProbe
from scripts.preview_table_task import aim_camera


def main(play=None):
    if play is None:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--play',action='store_true')
        play=parser.parse_args().play
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/table_task_v3.json').read_text())
    env=OpenArmInsertEnv(images=False)
    probe=PickupProbe(env)
    try:
        probe.reset(cfg['plug_mating_position_m'])
        # Viewer overview includes both hanging arms and the tabletop.
        aim_camera(env.model,'scene_rgb',[.90,-1.10,.85],[.20,0,.32],55)
        with mujoco.viewer.launch_passive(env.model,env.data) as viewer:
            with viewer.lock():
                viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid=env.model.camera('scene_rgb').id
                viewer.opt.sitegroup[:]=0
            viewer.sync()
            if play:
                def update(phase):
                    if not viewer.is_running(): raise RuntimeError('Viewer closed')
                    viewer.sync()
                    time.sleep(.1)
                print(probe.run(position=cfg['plug_mating_position_m'],pitch=cfg['tool_pitch_deg'],
                    height=cfg['grasp_height_offset_m'],closed_travel=cfg['closed_finger_target_m'],frame_callback=update))
            while viewer.is_running():
                viewer.sync()
                time.sleep(.02)
    finally:
        env.close()


if __name__=='__main__':main()
