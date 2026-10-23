"""Offline contact/vertical-escape geometry for the preserved failed teacher.

Saved state is loaded only into PRIVATE MjData, never a running environment.
Kinematic clearance is not a prediction of dynamic recovery success.
"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from learning.vla.runtime_io import write_json,read_json
from learning.vla.source_archive import archive_source
from scripts.rollout_smolvla import verify_snapshot


def main():
    output=ROOT/'outputs/teacher_release_geometry_20261020'; output.mkdir(exist_ok=False)
    source=ROOT/'outputs/teacher_takeovers_20261020/train_22045_at_350'
    meta=read_json(ROOT/'data/drive_v2c30_20261012/merged_all_v1/train_22045/metadata.json')
    with np.load(source/'trajectory.npz') as z: qpos=z['qpos'][-1].copy(); qvel=z['qvel'][-1].copy()
    sys.path.insert(0,str(ROOT/'third_party/capstone_automateData'))
    import mujoco
    from envs.openarm_insert import OpenArmInsertEnv
    from controllers.pick_insert_expert import GraspPlanner
    from controllers.place_insert import plug_geometry
    env=OpenArmInsertEnv(images=False,workspace=verify_snapshot(ROOT/'third_party/capstone_automateData',meta))
    try:
        env.reset(seed=meta['seed'],options=meta['reset_options'])
        scratch=mujoco.MjData(env.model); scratch.qpos[:]=qpos; scratch.qvel[:]=qvel
        mujoco.mj_forward(env.model,scratch)
        contacts=[]
        for c in scratch.contact:
            bodies=[int(env.model.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
            if env.plug in bodies and any(b in env.fingers for b in bodies):
                contacts.append({'geoms':[env.model.geom(g).name for g in (c.geom1,c.geom2)],
                    'distance_m':float(c.dist),'world_position_m':c.pos.tolist(),'normal_world':c.frame[:3].tolist()})
        planner=GraspPlanner(env.model,float(env.reset_options.get('table_height_m',0.)),geom=plug_geometry(env.model,env.workspace))
        planner.set_state(qpos); q=qpos[env.qa['right']].copy(); p,rotation=planner.fk(q)
        path=planner.line(q,p,p+np.array([0.,0.,.08]),rotation,step=.002)
        samples=[]
        if path:
            for k,joints in enumerate(path,1):
                samples.append({'vertical_motion_m':k*.08/len(path),
                    'planned_contacts_at_open_command':planner.contacts(joints,.044),
                    'planned_table_clearance_m':planner.distal_clearance(joints,.044)})
        diagnostics=read_json(source/'diagnostics.json')
        report={'offline_private_data_only':True,'source':str(source),'initial_contacts':contacts,
            'measured_final_contact_force_n':diagnostics[-1]['grasp_contact_force_n'],
            'measured_final_finger_mm':diagnostics[-1]['grip_travel_m']*1000,
            'vertical_escape_ik_available':path is not None,'samples':samples,
            'implementation':archive_source('scripts/diagnose_teacher_release.py'),
            'limits':'Private-data geometry only. Does not demonstrate a safe physical recovery, relax a guard, or change a recorded trial.'}
        write_json(output/'report.json',report); print(json.dumps(report),flush=True)
    finally: env.close()


if __name__=='__main__': main()
