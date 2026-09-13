"""Physical reach-close-lift probe. Not an end-to-end task or BC collector."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image
from envs.openarm_insert import OpenArmInsertEnv
from scripts.preview_table_task import aim_camera, inspect_ik, unexpected_contacts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("results/table_pickup_v2"))
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path("configs/table_task_v2.json").read_text())
    env=OpenArmInsertEnv(images=False)
    renderer=None
    rows=[]
    frames=[]
    outcome="incomplete"
    measurements={}
    try:
        m,d=env.model,env.data
        a=env.plug_qadr
        d.qpos[a:a+3]=config["plug_mating_position_m"]
        d.qpos[a+3:a+7]=config["plug_quaternion_wxyz"]
        d.qvel[:]=0
        d.qpos[env.fqa["right"]]=config["open_finger_travel_m"]
        env.target[7]=config["open_finger_travel_m"]
        mujoco.mj_forward(m,d)
        for _ in range(2000): env._physics_step()
        mujoco.mj_forward(m,d)
        start_height=float(d.xpos[env.plug,2])
        grasp=d.site_xpos[m.site("plug_grasp_frame").id].copy()+[0,0,config["grasp_height_offset_m"]]
        angle=np.radians(config["tool_pitch_deg"])
        rot=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        q0=d.qpos[env.qa["right"]].copy()
        retreat_position=d.site_xpos[env.grasp_site].copy()+[-.08,0,0]
        retreat,retreat_check=inspect_ik(env,retreat_position,d.site_xmat[env.grasp_site].reshape(3,3).copy(),q0)
        above,check=inspect_ik(env,grasp+[0,0,.07],rot,retreat)
        down,check2=inspect_ik(env,grasp,rot,above)
        if not retreat_check["ik_converged"] or not check["ik_converged"] or not check2["ik_converged"]:
            raise RuntimeError("IK failed")
        for c in config["cameras"].values():
            aim_camera(m,c["model_camera"],c["position_m"],c["target_m"],c["fovy_deg"])
        renderer=mujoco.Renderer(m,height=480,width=640)
        close_reference=None
        for phase,goal,finger,seconds in (("retreat",retreat,.035,3.),("reach",above,.035,6.),("descend",down,.035,3.),
                                          ("close",down,.008,2.),("lift",above,.008,3.),("hold",above,.008,1.)):
            for i in range(round(seconds/env.dt)):
                env.target[:7]+=np.clip(goal-env.target[:7],-.5*env.dt,.5*env.dt)
                env.target[7]+=np.clip(finger-env.target[7],-.02*env.dt,.02*env.dt)
                for _ in range(env.substeps): env._physics_step()
                mujoco.mj_forward(m,d)
                info=env._info()
                rows.append({"phase":phase,"time_s":float(d.time),"plug_height_m":float(d.xpos[env.plug,2]),
                             "robot_contact_n":info["robot_unwanted_contact_n"],
                             "joint_error_rad":float(np.max(np.abs(goal-d.qpos[env.qa["right"]])))})
                if info["robot_unwanted_contact_n"]>5 or any(w.number for w in d.warning):
                    raise RuntimeError("Contact or numerical safety abort: "+str(unexpected_contacts(env,d)))
                if i%5==0:
                    renderer.update_scene(d,camera="scene_rgb")
                    frames.append(Image.fromarray(renderer.render()).resize((480,360)))
            renderer.update_scene(d,camera="scene_rgb")
            Image.fromarray(renderer.render()).save(args.output/f"{phase}.png")
            if phase=="close":
                close_reference=d.site_xmat[env.grasp_site].reshape(3,3).T @ (d.xpos[env.plug]-d.site_xpos[env.grasp_site])
        relative=d.site_xmat[env.grasp_site].reshape(3,3).T @ (d.xpos[env.plug]-d.site_xpos[env.grasp_site])
        slip=float(np.linalg.norm(relative-close_reference))
        height_gain=float(d.xpos[env.plug,2]-start_height)
        outcome="lift_success" if height_gain>.04 and slip<.003 else "grasp_or_lift_failed"
        measurements={"height_gain_m":height_gain,"grasp_drift_m":slip}
        print({"outcome":outcome,"height_gain_m":height_gain,"grasp_drift_m":slip})
    except RuntimeError as error:
        outcome=str(error)
        print(outcome)
    finally:
        if frames: frames[0].save(args.output/"pickup.gif",save_all=True,append_images=frames[1:],duration=100,loop=0)
        (args.output/"report.json").write_text(json.dumps({"outcome":outcome,"trace":rows,
            "config":config,"measurements":measurements,
            "probe_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "end_to_end_validated":False,"demonstration_exported":False},indent=2))
        if renderer: renderer.close()
        env.close()
    raise SystemExit(0 if outcome=="lift_success" else 1)


if __name__=="__main__": main()
