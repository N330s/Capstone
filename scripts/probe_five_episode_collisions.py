"""Offline same-image/same-noise state-channel interventions; never runs a policy rollout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']=os.environ['HF_DATASETS_OFFLINE']='1'
os.environ.setdefault('HF_HOME',str(ROOT/'data/hf_cache'))
from learning.vla.smolvla_data import observation,sha256
from learning.vla.runtime_io import write_json
from learning.vla.source_archive import archive_source
from scripts.rollout_smolvla import verify_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from learning.vla.smolvla_policy import SmolVLAController
    from scripts.train_gripper_feedback import BASE,BACKBONE
    campaign=ROOT/'outputs/five_episode_openhold_20261020'
    trained=json.loads((campaign/'training/report.json').read_text()); overlay=campaign/'training/expert'
    if sha256(overlay/'expert.safetensors')!=trained['overlay']['weights_sha256']: raise ValueError('Frozen candidate changed')
    write_json(args.output/'plan.json',{'candidate_sha256':trained['overlay']['weights_sha256'],
        'implementation':archive_source('scripts/probe_five_episode_collisions.py'),
        'cases':['train_20003','train_22045','train_20042'],'seeds':[1000,1001],
        'limits':'Offline FK rendering from saved qpos/qvel, not state forcing during execution. Counterfactual numeric-state changes reuse exact live-render RGB and cloned noise; not feasible recoveries or new successes.'})
    controller=SmolVLAController(BASE,BACKBONE,1000,expert_overlay=overlay)
    sys.path.insert(0,str(ROOT/'third_party/capstone_automateData'))
    from envs.openarm_insert import OpenArmInsertEnv
    import mujoco
    rows=[]; outputs={}
    for name in ('train_20003','train_22045','train_20042'):
        source=ROOT/'data/drive_v2c30_20261012/merged_all_v1'/name
        meta=json.loads((source/'metadata.json').read_text())
        if sha256(source/'episode.npz')!=meta['data_sha256']: raise ValueError('Source changed')
        with np.load(source/'episode.npz',allow_pickle=False) as raw:
            data={k:raw[k].copy() for k in ('observation_state','action_applied','phase','image_scene_rgb','image_wrist_rgb')}
        with np.load(campaign/('training_policy_'+name)/'trajectory.npz',allow_pickle=False) as raw:
            live={k:raw[k].copy() for k in ('qpos','qvel','observation_state')}
        close=int(np.flatnonzero(data['phase']=='close')[0])
        anchors=sorted({200,275,300,350,400,450,min(500,close-5,len(live['qpos'])-1)})
        env=OpenArmInsertEnv(images=True,workspace=verify_snapshot(ROOT/'third_party/capstone_automateData',meta))
        try:
            env.reset(seed=meta['seed'],options=meta['reset_options'])
            for anchor in anchors:
                if time.monotonic()-started>270: raise TimeoutError('Five-minute offline probe allowance exhausted')
                env.data.qpos[:]=live['qpos'][anchor]; env.data.qvel[:]=live['qvel'][anchor]
                mujoco.mj_forward(env.model,env.data)
                actual=env.observe()
                if not np.array_equal(actual['state'],live['observation_state'][anchor]):
                    raise ValueError('Offline render state differs from saved live state')
                image_hashes={cam:hashlib.sha256(actual['images'][cam].tobytes()).hexdigest() for cam in ('scene_rgb','wrist_rgb')}
                for seed in (1000,1001):
                    noise=torch.randn((1,50,controller.model.config.max_action_dim),device='cuda',
                        generator=torch.Generator(device='cuda').manual_seed(seed+anchor))
                    variants={}
                    for variant in ('recorded','live','live_only_finger_recorded','live_only_right_joints_recorded'):
                        state=(data['observation_state'][anchor] if variant=='recorded' else actual['state']).copy()
                        images=({cam:data['image_'+cam][anchor] for cam in ('scene_rgb','wrist_rgb')} if variant=='recorded' else actual['images'])
                        if variant=='live_only_finger_recorded': state[15]=data['observation_state'][anchor,15]
                        if variant=='live_only_right_joints_recorded': state[8:15]=data['observation_state'][anchor,8:15]
                        batch=controller.pre(default_collate([observation({'state':state,'instruction':meta['instruction'],'images':images})]))
                        controller.model.reset()
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                            result=controller.post(controller.model.predict_action_chunk(batch,noise=noise.clone()))
                        chunk=result[0].float().cpu().numpy()
                        if chunk.shape!=(50,8) or not np.isfinite(chunk).all(): raise ValueError('Invalid decoded chunk')
                        outputs[f'{name}_{anchor}_{seed}_{variant}']=chunk
                        variants[variant]={'input_finger_mm':float(state[15]*1000),'first_target_mm':float(chunk[0,7]*1000),
                            'first5_target_mm':float(chunk[:5,7].mean()*1000),
                            'first25_target_mm':float(chunk[:25,7].mean()*1000),
                            'first_target_minus_input_mm':float((chunk[0,7]-state[15])*1000),
                            'first7_joint_targets_rad':chunk[0,:7].tolist()}
                    rows.append({'episode':name,'anchor':anchor,'seed':seed,'recorded_phase':str(data['phase'][anchor]),
                        'recorded_target_mm':float(data['action_applied'][anchor,7]*1000),'live_rgb_hashes':image_hashes,'variants':variants})
                print(json.dumps({'episode':name,'anchor':anchor}),flush=True)
        finally: env.close()
        del data,live
    np.savez_compressed(args.output/'predictions.npz',**outputs)
    write_json(args.output/'report.json',{'completed':True,'rows':rows,'elapsed_s':time.monotonic()-started,
        'candidate_sha256':trained['overlay']['weights_sha256'],
        'limits':'Same-image/noise comparisons isolate state-channel effects, not optimal on-policy targets. Recorded-time expert action is not ground truth after phase/pose/object drift. No training, guard changes, robot commands or success claims.'})


if __name__=='__main__': main()
