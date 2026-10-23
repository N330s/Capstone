"""Bounded read-only FK/contact and training-coverage audit of the completed five-parent run."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from learning.vla.smolvla_data import sha256
from learning.vla.runtime_io import write_json
from learning.vla.source_archive import archive_source
from learning.vla.collision_diagnosis import contact_chronology, orientation_error_deg, nearest_path
from scripts.rollout_smolvla import verify_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    args.output=args.output.resolve()
    if args.output.parent!=ROOT/'outputs': raise ValueError('New direct outputs directory required')
    args.output.mkdir(exist_ok=False); started=time.monotonic()
    campaign=ROOT/'outputs/five_episode_openhold_20261020'
    plan=json.loads((campaign/'plan.json').read_text())
    completed=json.loads((campaign/'completion.json').read_text())
    selection=json.loads((campaign/'selection.json').read_text())
    trained=json.loads((campaign/'training/report.json').read_text())
    overlay=campaign/'training/expert/expert.safetensors'
    if not completed['completed'] or sha256(overlay)!=trained['overlay']['weights_sha256']:
        raise ValueError('Completed frozen candidate required')
    snapshot=ROOT/'third_party/capstone_automateData'
    sources={f:archive_source(f) for f in ('scripts/diagnose_five_episode_collisions.py','learning/vla/collision_diagnosis.py')}
    write_json(args.output/'plan.json',{'campaign':str(campaign),'candidate_sha256':sha256(overlay),
        'campaign_completion_sha256':sha256(campaign/'completion.json'),'sources':sources,
        'limits':'Read-only saved-trajectory analysis; offline FK in separate MjData, no physical execution or contact-force recomputation.'})
    sys.path.insert(0,str(snapshot))
    from envs.openarm_insert import OpenArmInsertEnv
    import mujoco
    updates=list(map(json.loads,(campaign/'training/updates.jsonl').read_text().splitlines()))
    rows=[]
    for parent in selection['parents']:
        if time.monotonic()-started>280: raise TimeoutError('Five-minute diagnostic allowance exhausted')
        name=parent['episode']; folder=args.output/name; folder.mkdir()
        source=ROOT/'data/drive_v2c30_20261012/merged_all_v1'/name
        meta=json.loads((source/'metadata.json').read_text())
        if sha256(source/'episode.npz')!=parent['data_sha256']: raise ValueError('Original recording changed')
        with np.load(source/'episode.npz',allow_pickle=False) as data:
            reference_qpos=data['qpos'].copy(); phases=data['phase'].copy()
            ref_states=data['observation_state'].copy(); ref_actions=data['action_applied'].copy()
        live=campaign/('training_policy_'+name)
        report=json.loads((live/'report.json').read_text()); diagnostics=json.loads((live/'diagnostics.json').read_text())
        expert=campaign/('training_expert_'+name)
        expert_report=json.loads((expert/'report.json').read_text())
        if (report['expert_overlay_weights_sha256']!=trained['overlay']['weights_sha256'] or report['recorded_input_assistance']
                or report['simulator_state_forced_to_recording'] or report['reset_state_max_error']!=0
                or max(expert_report['replay_errors'].values())!=0): raise ValueError('Original evaluation binding differs')
        with np.load(live/'trajectory.npz',allow_pickle=False) as data:
            actual={k:data[k].copy() for k in data.files}
        env=OpenArmInsertEnv(images=False,workspace=verify_snapshot(snapshot,meta))
        try:
            env.reset(seed=meta['seed'],options=meta['reset_options']); independent=mujoco.MjData(env.model)
            plug_site=env.model.site('plug_grasp_frame').id
            paths=[]
            for sequence in (actual['qpos'],reference_qpos):
                tcp=[]; rotations=[]; plug=[]; plug_rot=[]
                for q in sequence:
                    independent.qpos[:]=q; mujoco.mj_forward(env.model,independent)
                    tcp.append(independent.site_xpos[env.grasp_site].copy())
                    rotations.append(independent.site_xmat[env.grasp_site].reshape(3,3).copy())
                    plug.append(independent.site_xpos[plug_site].copy())
                    plug_rot.append(independent.xmat[env.plug].reshape(3,3).copy())
                paths.append((np.asarray(tcp),np.asarray(rotations),np.asarray(plug),np.asarray(plug_rot)))
            tcp,rot,plug,plug_rot=paths[0]; ref_tcp,ref_rot,ref_plug,ref_plug_rot=paths[1]
            close=int(np.flatnonzero(phases=='close')[0])
            nearest,nearest_index=nearest_path(tcp,ref_tcp[:close+1])
            local_error=np.einsum('tji,tj->ti',rot,plug-tcp)
            alignment=np.linalg.norm(plug-tcp,axis=1)
            angle=orientation_error_deg(rot,plug_rot)
            # Before latching, saved slip is raw site separation. After latching
            # it is displacement relative to the grasp reference, not separation.
            first_grasp=next((i for i,d in enumerate(diagnostics) if d['grasp_established']),len(diagnostics))
            if np.max(np.abs(alignment[:first_grasp]-np.array([d['grasp_slip_m'] for d in diagnostics[:first_grasp]])))>1e-7:
                raise ValueError('Offline FK differs from saved hand/plug displacement')
            z=actual['qpos'][:,report['pickup_evaluation']['plug_qpos_address']+2]
            chronology=contact_chronology(diagnostics,actual['observation_state'],z)
            selected=sorted({0,100,200,250,275,300,350,400,450,500,len(tcp)-1,
                *[x for x in [chronology['first_one_or_two_contacts'],chronology['first_below_33mm'],chronology['first_below_30mm']] if x is not None]})
            measurements=[]
            for i in selected:
                if i>=len(tcp): continue
                measurements.append({'command':i,'finger_mm':float(actual['observation_state'][i,15]*1000),
                    'expert_finger_mm':float(ref_states[i,15]*1000),'contacts':diagnostics[i]['plug_finger_contacts'],
                    'hand_to_actual_plug_grasp_mm':float(alignment[i]*1000),'hand_to_initial_plug_grasp_mm':float(np.linalg.norm(tcp[i]-plug[0])*1000),
                    'hand_frame_plug_offset_xyz_mm':(local_error[i]*1000).tolist(),'hand_plug_angle_deg':float(angle[i]),
                    'nearest_expert_preclose_path_mm':float(nearest[i]*1000),'nearest_expert_command':int(nearest_index[i]),
                    'orientation_vs_nearest_expert_deg':float(orientation_error_deg(rot[i],ref_rot[nearest_index[i]])),
                    'expert_phase_at_same_index':str(phases[min(i,len(phases)-1)])})
            augmented=[]
            for update in updates:
                if update['episode']==name:
                    for index,offset in zip(update['indices'][:4],update['gripper_sensor_offsets_m']):
                        augmented.append(float(ref_states[index-parent['start'],15]+offset))
            coverage={'sampled_augmented_state_min_mm':min(augmented)*1000,
                'sampled_augmented_state_max_mm':max(augmented)*1000,'augmented_samples':len(augmented),
                'source_preclose_below_33mm':int(np.sum(ref_states[:close,15]<.033)),
                'source_preclose_below_30mm':int(np.sum(ref_states[:close,15]<.030))}
            row={'episode':name,'outcome':report['outcome'],'pickup_success':report['autonomous_pickup_success'],
                'steps':report['steps'],'expert_close_anchor':close,'chronology':chronology,'measurements':measurements,
                'coverage':coverage,'terminal_robot_contacts':report['final_robot_contacts'],
                'terminal_unwanted_contact_n':diagnostics[-1]['robot_unwanted_contact_n'],
                'unchanged_robot_abort_n':env.config['robot_contact_abort_n'],
                'source_data_sha256':parent['data_sha256'],'trajectory_sha256':sha256(live/'trajectory.npz')}
            np.savez_compressed(folder/'kinematics.npz',tcp=tcp,rotation=rot,plug_grasp=plug,
                hand_frame_plug_offset=local_error,nearest_path_m=nearest,nearest_path_index=nearest_index,
                expert_tcp=ref_tcp,expert_rotation=ref_rot,expert_plug_grasp=ref_plug,
                finger_mm=actual['observation_state'][:,15]*1000,expert_finger_mm=ref_states[:,15]*1000,
                contacts=np.array([d['plug_finger_contacts'] for d in diagnostics]),
                plug_lift_mm=(z-z[0])*1000,hand_plug_angle_deg=angle)
            write_json(folder/'report.json',row); rows.append(row)
            print(json.dumps({'episode':name,'outcome':row['outcome'],'chronology':chronology,'coverage':coverage}),flush=True)
        finally: env.close()
    write_json(args.output/'report.json',{'completed':True,'rows':rows,'elapsed_s':time.monotonic()-started,
        'limits':'Five existing training-reset rollouts, one policy seed. Same-index differences include timing; nearest-path distances ignore orientation. Contact forces come only from original saved execution.'})


if __name__=='__main__': main()
