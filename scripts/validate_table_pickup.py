"""Repeat physical pickup from downward rest, save contacts/poses and optional video."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
from PIL import Image
from controllers.table_pickup import PickupProbe
from envs.openarm_insert import OpenArmInsertEnv
from scripts.preview_table_task import aim_camera


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results/table_pickup_v3'))
    p.add_argument('--render',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path('configs/table_task_v3.json').read_text())
    cases=[('nominal_0',0,0,.0005),('nominal_1',0,0,.0005),('nominal_2',0,0,.0005),
           ('x_plus',.002,0,.0005),('x_minus',-.002,0,.0005),
           ('y_plus',0,.002,.0005),('y_minus',0,-.002,.0005),('half_timestep',0,0,.00025)]
    results=[]
    for name,x,y,dt in cases:
        env=OpenArmInsertEnv(images=False,timestep=dt)
        probe=PickupProbe(env)
        renderer=None; frames=[]
        try:
            callback=None
            if args.render and name=='nominal_0':
                for c in cfg['cameras'].values():
                    aim_camera(env.model,c['model_camera'],c['position_m'],c['target_m'],c['fovy_deg'])
                renderer=mujoco.Renderer(env.model,height=480,width=640)
                def callback(phase):
                    renderer.update_scene(env.data,camera='scene_rgb')
                    frame=Image.fromarray(renderer.render())
                    frames.append(frame.resize((480,360)))
                    frame.save(args.output/f'{phase}.png')
            result=probe.run(position=(.381+x,-.22+y,.330),frame_callback=callback,
                             pitch=cfg['tool_pitch_deg'],height=cfg['grasp_height_offset_m'],
                             closed_travel=cfg['closed_finger_target_m'])
        except RuntimeError as error:
            result={'passed':False,'error':str(error)}
        finally:
            if renderer: renderer.close()
            env.close()
        result.update(case=name,offset_x_m=x,offset_y_m=y,timestep_s=dt)
        results.append(result)
        (args.output/f'{name}_trace.json').write_text(json.dumps(probe.trace,indent=2))
        if frames: frames[0].save(args.output/'pickup.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
        (args.output/'report.json').write_text(json.dumps({'config':cfg,'cases':results,
            'complete':len(results)==len(cases),'passed':len(results)==len(cases) and all(r['passed'] for r in results),
            'source_hashes':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                [Path(__file__),Path('controllers/table_pickup.py'),Path('envs/openarm_insert.py'),Path('configs/table_task_v3.json')]}},indent=2))
        print(name,result,flush=True)
    raise SystemExit(0 if all(r['passed'] for r in results) else 1)


if __name__=='__main__':main()
