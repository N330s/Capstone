"""View OpenArm v1 expert execution. Space: run/pause; R: reset; Esc: close."""
import queue
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import mujoco.viewer
from envs.openarm_insert import OpenArmInsertEnv
from controllers.expert import InsertionExpert

def main():
    parser=argparse.ArgumentParser(description='OpenArm: default downward-rest table task; insertion benchmark remains explicit.')
    parser.add_argument('--task',choices=('table','insertion'),default='table')
    parser.add_argument('--play',action='store_true',help='Run the table pickup once')
    args=parser.parse_args()
    if args.task=='table':
        from scripts.view_table_pickup import main as table_main
        return table_main(play=args.play)
    env=OpenArmInsertEnv(images=False)
    expert=InsertionExpert(env)
    keys=queue.SimpleQueue()
    running=False
    print(__doc__)
    try:
        with mujoco.viewer.launch_passive(env.model,env.data,key_callback=keys.put) as viewer:
            with viewer.lock():
                viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid=env.model.camera("scene_rgb").id
                viewer.opt.geomgroup[3]=0
                viewer.opt.sitegroup[:]=0
            while viewer.is_running():
                start=time.perf_counter()
                with viewer.lock():
                    while not keys.empty():
                        key=keys.get()
                        if key==32: running=not running
                        if key in (82,114):
                            env.reset()
                            expert=InsertionExpert(env)
                            running=False
                    if running and not env.outcome:
                        env.step(expert.action())
                        if env.outcome: print(env.outcome)
                viewer.sync()
                time.sleep(max(0,env.dt-(time.perf_counter()-start)))
    finally:
        env.close()

if __name__=="__main__":
    main()
