"""Preview a table-resting task and camera candidates without changing the baseline.

Robot qpos writes in this tool are reset or scratch kinematic inspection only.
No kinematic pose is exported as a demonstration or claimed as a physical grasp.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image
from envs.openarm_insert import OpenArmInsertEnv

ROOT = Path(__file__).resolve().parents[1]


def aim_camera(model, name, position, target, fovy):
    z = np.asarray(position) - np.asarray(target)
    z = z / np.linalg.norm(z)
    x = np.cross([0.,0.,1.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z,x)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.column_stack((x,y,z)).ravel())
    camera = model.camera(name).id
    model.cam_pos[camera] = position
    model.cam_quat[camera] = quat
    model.cam_fovy[camera] = fovy


def unexpected_contacts(env, data):
    contacts = []
    for c in data.contact:
        bodies = {int(env.model.geom_bodyid[g]) for g in (c.geom1,c.geom2)}
        if not bodies & env.robot_bodies or c.dist >= 0:
            continue
        if env.plug in bodies and bodies & set(env.fingers):
            continue
        contacts.append({"geoms":[env.model.geom(g).name for g in (c.geom1,c.geom2)],
                         "penetration_m":float(-c.dist)})
    return contacts


def inspect_ik(env, position, orientation, initial):
    q = initial.copy()
    converged = False
    # The base solver's 100-iteration limit can stop early near a wrist limit.
    # Continue from its scratch result, never from a live teleported robot pose.
    for _ in range(5):
        try:
            q = env.solve_ik(position, orientation, initial=q)
            converged = True
            break
        except ValueError:
            q = env.ik_data.qpos[env.qa["right"]].copy()
    scratch = env.ik_data
    mujoco.mj_forward(env.model,scratch)
    return q, {"ik_converged":converged,
        "position_error_m":float(np.linalg.norm(scratch.site_xpos[env.grasp_site]-position)),
        "minimum_joint_limit_margin_rad":float(np.min(np.r_[q-env.lower,env.upper-q])),
        "unexpected_contacts":unexpected_contacts(env,scratch),
        "joint_targets_rad":q.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"configs/table_task_v3.json")
    p.add_argument("--output",type=Path,default=ROOT/"results/table_task_v3")
    args = p.parse_args()
    args.config = args.config.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    config = json.loads(args.config.read_text())
    env = OpenArmInsertEnv(images=False)
    renderer = None
    try:
        m,d = env.model,env.data
        # A separate reset: put the free plug just above the table, release it,
        # open the fingers, then settle using physics. No attachment or plug force.
        a = env.plug_qadr
        d.qpos[a:a+3] = config["plug_mating_position_m"]
        d.qpos[a+3:a+7] = config["plug_quaternion_wxyz"]
        d.qvel[:] = 0
        if 'rest_arm_rad' in config:
            for side in ('left','right'):
                d.qpos[env.qa[side]]=config['rest_arm_rad']
            env.target[:7]=config['rest_arm_rad']
            env.park_target=np.array(config['rest_arm_rad'],dtype=float)
        d.qpos[env.fqa["right"]] = config["open_finger_travel_m"]
        env.target[7] = config["open_finger_travel_m"]
        mujoco.mj_forward(m,d)
        for _ in range(round(config["settle_s"]/m.opt.timestep)):
            env._physics_step()
        mujoco.mj_forward(m,d)
        table_contacts = [c for c in d.contact if {m.geom(c.geom1).name,m.geom(c.geom2).name}=={"housing","work_table"}]
        settled = {"plug_position_m":d.xpos[env.plug].tolist(),
            "plug_speed_norm":float(np.linalg.norm(d.qvel[m.jnt_dofadr[m.joint("plug_free").id]:])),
            "housing_table_contacts":len(table_contacts),
            "max_table_penetration_m":max([float(-c.dist) for c in table_contacts]+[0.]),
            "unexpected_robot_contacts":unexpected_contacts(env,d),
            "warnings":sum(w.number for w in d.warning)}
        for camera in config["cameras"].values():
            aim_camera(m,camera["model_camera"],camera["position_m"],camera["target_m"],camera["fovy_deg"])
        mujoco.mj_forward(m,d)
        renderer = mujoco.Renderer(m,height=480,width=640)
        def render(prefix):
            for alias,camera in config["cameras"].items():
                renderer.update_scene(d,camera=camera["model_camera"])
                Image.fromarray(renderer.render()).save(args.output/f"{prefix}_{alias}.png")
        render("table_start")
        offset = config.get("grasp_height_offset_m",0.)
        grasp = d.site_xpos[m.site("plug_grasp_frame").id].copy()+np.array([0,0,offset])
        angle = np.radians(config.get("tool_pitch_deg",90.))
        topdown = np.array([[np.cos(angle),0.,np.sin(angle)],[0.,1.,0.],[-np.sin(angle),0.,np.cos(angle)]])
        targets = {"pregrasp":grasp+np.array([0,0,config["pregrasp_clearance_m"]]),
                   "grasp":grasp,
                   "seated_same_topdown_grasp":d.site_xpos[m.site("socket_entry").id]+np.array([-.016,0,offset])}
        ik = {}
        q = d.qpos[env.qa["right"]].copy()
        for name,pos in targets.items():
            q,ik[name] = inspect_ik(env,pos,topdown,q)
        # Render the new cameras against an existing recorded seated state as a
        # visual-only probe; this is NOT a table-to-insert rollout.
        with np.load(ROOT/"data/openarm_v1_varied10/varied_0000/episode.npz",allow_pickle=False) as recorded:
            d.qpos[:] = recorded["qpos"][-1]
            d.qvel[:] = recorded["qvel"][-1]
        mujoco.mj_forward(m,d)
        render("recorded_seating_camera_probe")
        report = {"config":config,"settled_table_reset":settled,"kinematic_probes":ik,
            "physical_pickup_validated":False,"end_to_end_success":False,
            "camera_schema":"candidate 640x480 task+socket fixed views; original wrist retained in model",
            "source_hashes":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__),ROOT/"envs/openarm_insert.py",ROOT/"envs/scene.py",args.config)},
            "baseline_manifest":env.manifest()}
        (args.output/"report.json").write_text(json.dumps(report,indent=2))
        print(json.dumps({"settled":settled,"ik":ik},indent=2))
    finally:
        if renderer: renderer.close()
        env.close()


if __name__=="__main__":
    main()
