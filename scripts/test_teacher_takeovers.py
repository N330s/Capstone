"""Four physical prefix replays + privileged pickup recoveries, never VLA successes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Pin ROOT modules before the scene snapshot shadows its own learning package.
from learning.vla.teacher_takeover import check_prefix_state, command_owner, recovery_teacher
from learning.vla.physical_grasp import install_physical_grasp_metrics
from learning.vla.pickup_evaluation import PickupGate
from learning.vla.runtime_io import write_json as atomic_write_json, read_json
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source
from scripts.rollout_smolvla import verify_snapshot, serial

CAMPAIGN = ROOT/'outputs/five_episode_openhold_20261020'
CANDIDATE_SHA = 'f1518a2141cf382cafa5e893e12fe698470d0b5c2d62fb33500f8774d7546ac3'
RESERVE = 25*1024**3


def write_json(path, value):
    atomic_write_json(path, json.loads(json.dumps(value, default=serial, allow_nan=False)))


def disk_guard(extra=0):
    if shutil.disk_usage(ROOT).free < RESERVE+extra:
        raise RuntimeError('25 GiB free-space reserve would be crossed')


def write_video(path, scene, wrist, owners, phases, episode, outcome):
    import av
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 22)
    with av.open(str(path), mode='w', options={'movflags':'+faststart'}) as container:
        stream = container.add_stream('libx264', rate=50)
        stream.width, stream.height, stream.pix_fmt = 1280, 624, 'yuv420p'
        stream.options = {'crf':'20', 'preset':'fast'}
        for i, (a,b) in enumerate(zip(scene,wrist)):
            canvas=Image.new('RGB',(1280,624),'#111e30')
            canvas.paste(Image.fromarray(a).resize((640,480)),(0,88))
            canvas.paste(Image.fromarray(b).resize((640,480)),(640,88))
            d=ImageDraw.Draw(canvas)
            d.text((20,8),'DIAGNOSTIC: saved VLA prefix -> privileged expert takeover',font=font,fill='white')
            d.text((20,44),f'{episode} | {owners[i]} | {phases[i]}',font=font,fill='#adc4ed')
            d.text((20,583),f'Simulation t={i/50:.2f}s | NOT autonomous VLA success',font=font,fill='white')
            d.text((920,583),outcome,font=font,fill='#ffca8a')
            frame=av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)


def worker(args):
    import numpy as np
    started=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'data/drive_v2c30_20261012/merged_all_v1'/args.episode
    meta=read_json(source/'metadata.json')
    if sha256(source/'episode.npz') != meta['data_sha256']: raise ValueError('Source changed')
    if sha256(CAMPAIGN/'training/expert/expert.safetensors') != CANDIDATE_SHA: raise ValueError('Candidate changed')
    original=CAMPAIGN/('training_policy_'+args.episode)
    original_hashes={f:sha256(original/f) for f in ('trajectory.npz','diagnostics.json','report.json')}
    with np.load(original/'trajectory.npz',allow_pickle=False) as z:
        saved={k:z[k].copy() for k in ('qpos','qvel','action_requested','action_applied','executed_dt_s')}
    sys.path.insert(0,str(ROOT/'third_party/capstone_automateData'))
    from envs.openarm_insert import OpenArmInsertEnv
    env=OpenArmInsertEnv(images=True,workspace=verify_snapshot(ROOT/'third_party/capstone_automateData',meta))
    try:
        metrics=install_physical_grasp_metrics(env)
        obs,info=env.reset(seed=meta['seed'],options=meta['reset_options'])
        write_json(args.output/'manifest.json', {'environment':env.manifest(), 'grasp_metrics':metrics,
            'original_hashes':original_hashes, 'source_data_sha256':meta['data_sha256'],
            'frozen_candidate_sha256':CANDIDATE_SHA,'anchor_command_count':args.anchor,
            'mode':'privileged_expert_takeover','actor_weights_loaded':False,
            'live_state_forcing':False,'guard_overrides':False,'teacher_regrasps':0,
            'prefix_tolerance':1e-8,'recovery_sim_limit_s':55.,'recovery_step_limit':2750,
            'collection_status':'diagnostic candidate only, NOT exported to training'})
        errors=check_prefix_state(env.data.qpos,env.data.qvel,saved['qpos'][0],saved['qvel'][0])
        maximum_errors=errors.copy()
        gate=PickupGate(float(env.data.qpos[env.plug_qadr+2]))
        states=[obs['state'].copy()]; qpos=[env.data.qpos.copy()]; qvel=[env.data.qvel.copy()]
        diagnostics=[info]; scene=[obs['images']['scene_rgb'].copy()]; wrist=[obs['images']['wrist_rgb'].copy()]
        owners=[]; phases=[]; commands=[]; applied=[]; durations=[]; teacher_records=[]
        teacher=None; outcome='recovery_budget_exhausted'; prefix_passed=False
        prefix_applied_error=0.; prefix_dt_error=0.
        for i in range(args.anchor+2750):
            if time.monotonic()-started > 290: raise TimeoutError('Worker computation allowance exhausted')
            if i%50 == 0: disk_guard(128*1024**2)
            if i == args.anchor:
                prefix_passed=True
                write_json(args.output/'prefix_audit.json', {'passed':True,'commands':args.anchor,
                    'maximum_state_errors':maximum_errors, 'maximum_applied_error':prefix_applied_error,
                    'maximum_dt_error':prefix_dt_error,'state_forcing':False,
                    'takeover_plug_pos_m':env.data.xpos[env.plug].tolist(), 'takeover_info':info})
                teacher=recovery_teacher(env)
            owner=command_owner(i,args.anchor)
            if teacher is None:
                action=saved['action_requested'][i].copy(); phase='saved_failed_vla_prefix'
            else:
                action=np.asarray(teacher.action(),dtype=float)
                teacher_records.append(teacher.diagnostics())
                if teacher.failure:
                    outcome='teacher_failure:'+teacher.failure
                    break
                phase=teacher.diagnostics()['phase']
            if action.shape != (8,) or not np.isfinite(action).all(): raise ValueError('Invalid teacher command')
            obs,_,term,trunc,info=env.step(action)
            owners.append(owner); phases.append(phase); commands.append(action.copy()); applied.append(env.target.copy())
            dt=float(info['executed_dt_s'])
            durations.append(dt)
            states.append(obs['state'].copy()); qpos.append(env.data.qpos.copy()); qvel.append(env.data.qvel.copy())
            diagnostics.append(info); scene.append(obs['images']['scene_rgb'].copy()); wrist.append(obs['images']['wrist_rgb'].copy())
            if owner == 'prefix_replay':
                errors=check_prefix_state(env.data.qpos,env.data.qvel,saved['qpos'][i+1],saved['qvel'][i+1])
                maximum_errors={k:max(maximum_errors[k],errors[k]) for k in errors}
                prefix_applied_error=max(prefix_applied_error,float(np.max(np.abs(env.target-saved['action_applied'][i]))))
                prefix_dt_error=max(prefix_dt_error,abs(dt-float(saved['executed_dt_s'][i])))
                if prefix_applied_error > 1e-8 or prefix_dt_error > 1e-10: raise ValueError('Prefix commands/timing differ')
            passed=gate.update(plug_z_m=float(env.data.qpos[env.plug_qadr+2]),grasp=info['grasp_established'],
                contacts=info['plug_finger_contacts'],drops=info['drops'],safe=info['outcome'] in ('running','success'),dt=dt)
            if i%100 == 0:
                print(json.dumps({'episode':args.episode,'anchor':args.anchor,'command':i+1,'phase':phase,
                    'finger_mm':float(obs['state'][15]*1000),'contacts':info['plug_finger_contacts'],
                    'grasp':info['grasp_established'],'lift_mm':gate.maximum_lift_m*1000}),flush=True)
            if passed:
                outcome='teacher_pickup_success' if prefix_passed else 'unexpected_prefix_pickup'
                break
            if term or trunc:
                outcome=info['outcome']; break
        count=len(commands)
        # Independent replay of the physical success criterion from saved samples.
        audit=PickupGate(float(qpos[0][env.plug_qadr+2]))
        for j, d in enumerate(diagnostics[1:]):
            audit.update(plug_z_m=float(qpos[j+1][env.plug_qadr+2]),grasp=d['grasp_established'],
                contacts=d['plug_finger_contacts'],drops=d['drops'],safe=d['outcome'] in ('running','success'),dt=durations[j])
        if audit.passed != (outcome == 'teacher_pickup_success'): raise ValueError('Saved physical pickup audit failed')
        # Prefix targets explicitly excluded. Failed teacher attempts are not positive examples.
        qualified=prefix_passed and audit.passed
        trace={'qpos':np.asarray(qpos),'qvel':np.asarray(qvel),'observation_state':np.asarray(states),
            'action_requested':np.asarray(commands),'action_applied':np.asarray(applied),
            'executed_dt_s':np.asarray(durations),'command_owner':np.asarray(owners), 'phase':np.asarray(phases),
            'teacher_positive_candidate':np.asarray([qualified and x=='teacher_recovery' for x in owners])}
        disk_guard(128*1024**2)
        np.savez_compressed(args.output/'trajectory.npz',**trace)
        if count > args.anchor:
            np.savez_compressed(args.output/'recovery_observations.npz',
                observation_state=np.asarray(states[args.anchor:]), image_scene_rgb=np.asarray(scene[args.anchor:]),
                image_wrist_rgb=np.asarray(wrist[args.anchor:]),action_requested=np.asarray(commands[args.anchor:]),
                executed_dt_s=np.asarray(durations[args.anchor:]),phase=np.asarray(phases[args.anchor:]),
                time_s=np.r_[0.,np.cumsum(durations)][args.anchor:],
                instruction=np.asarray(meta['instruction']),qualified_positive_candidate=np.asarray(qualified))
        write_json(args.output/'diagnostics.json',diagnostics)
        write_json(args.output/'teacher.json',{'records':teacher_records,
            'final':teacher.diagnostics() if teacher else None})
        frameowners=owners+[owners[-1]]; framephases=phases+[phases[-1]]
        # Observation at anchor is BEFORE the first teacher command.
        if prefix_passed:
            frameowners[args.anchor]='teacher_recovery'; framephases[args.anchor]='takeover_boundary'
        write_video(args.output/'rollout.mp4',scene,wrist,frameowners,framephases,args.episode,outcome)
        if original_hashes != {f:sha256(original/f) for f in original_hashes}: raise ValueError('Original rollout changed')
        if sha256(CAMPAIGN/'training/expert/expert.safetensors') != CANDIDATE_SHA: raise ValueError('Candidate changed')
        report={'completed':True,'episode':args.episode,'anchor':args.anchor,'outcome':outcome,
            'teacher_pickup_success':qualified,'prefix_audit_passed':prefix_passed,
            'steps':count,'recovery_commands':max(0,count-args.anchor),'maximum_state_errors':maximum_errors,
            'pickup_evaluation':{**gate.definition(),'passed':audit.passed,'maximum_lift_m':audit.maximum_lift_m,
                'maximum_consecutive_carry_s':audit.maximum_s},'wall_s':time.monotonic()-started,
            'peak_robot_contact_n':float(env.peak_robot_contact),'peak_support_contact_n':float(env.peak_support_contact),
            'drops':int(info['drops']),'video':'rollout.mp4', 'training_exported':False,
            'limits':'Privileged teacher recovery on two TRAINING resets, not autonomous VLA success, heldout generalization, insertion or sim2real. Prefix VLA targets are excluded from teacher candidates. No live state forcing, guard changes, training or actor weight edits.'}
        write_json(args.output/'report.json',report)
        print(json.dumps(report),flush=True)
    finally: env.close()


def supervise(args):
    args.output.mkdir(parents=True,exist_ok=False)
    disk_guard(512*1024**2)
    cases=[{'episode':e,'anchor':a} for e in ('train_20003','train_22045') for a in (275,350)]
    write_json(args.output/'plan.json',{'cases':cases,'candidate_sha256':CANDIDATE_SHA,
        'implementation':[archive_source(x) for x in ('scripts/test_teacher_takeovers.py','learning/vla/teacher_takeover.py')],
        'per_case_hard_timeout_s':360,'campaign_hard_limit_s':1500,'disk_reserve_gib':25,
        'prefix_tolerance':1e-8,'no_training':True,'no_validation_resets':True,
            'recovery':'Hold measured arm joints, open; if one light pad contact remains, plan 6 mm horizontal separation away from contact normals with a 0.5 N teacher force ceiling, then retreat vertically 80 mm, settle and re-plan actual-pose pickup. No regrasp retries or insertion.',
        'supersedes':'Initial four tests remain at outputs/teacher_takeovers_20261020 (3/4); repaired teacher is separately requalified on all four cases.'})
    started=time.monotonic(); rows=[]
    for case in cases:
        disk_guard(128*1024**2)
        name=f"{case['episode']}_at_{case['anchor']}"; destination=args.output/name
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--output',str(destination),
            '--episode',case['episode'],'--anchor',str(case['anchor'])]
        with (args.output/(name+'.log')).open('w',encoding='utf-8') as log:
            try:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    timeout=min(360,1500-(time.monotonic()-started)),check=False)
                status={'exit_code':result.returncode}
            except subprocess.TimeoutExpired:
                status={'hard_timeout':True}
        report=read_json(destination/'report.json') if (destination/'report.json').exists() else None
        rows.append({**case,**status,'report':report})
        write_json(args.output/'status.json',{'elapsed_s':time.monotonic()-started,'cases':rows})
        print(json.dumps({'case':name,**status,'outcome':report['outcome'] if report else 'process_failure'}),flush=True)
    write_json(args.output/'completion.json',{'completed':True,'elapsed_s':time.monotonic()-started,
        'teacher_successes':sum(bool(r['report'] and r['report']['teacher_pickup_success']) for r in rows),
        'case_count':len(rows),'all_reports_completed':all(r['report'] for r in rows),
        'cases':rows,'limits':'Four privileged recovery feasibility tests, not VLA task success or training.'})


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--worker',action='store_true')
    p.add_argument('--episode',choices=('train_20003','train_22045')); p.add_argument('--anchor',type=int,choices=(275,350))
    arguments=p.parse_args()
    if arguments.worker: worker(arguments)
    else: supervise(arguments)
