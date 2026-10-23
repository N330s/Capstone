"""Bounded, frozen-model 1/5/25 execution-prefix comparison on training resets."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from learning.vla.runtime_io import read_json,write_json,write_status
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source,verified_source
from learning.vla.inference_cadence import audit_cadence_chunks,resource_violations,continuation_state
from learning.vla.pickup_evaluation import audit_pickup_trace
from scripts.run_smolvla_job import arm_deadline,kill_owned_process
from scripts.train_gripper_feedback import BASE,BACKBONE

CAMPAIGN=ROOT/'outputs/five_episode_openhold_20261020'
OVERLAY=CAMPAIGN/'training/expert'
SNAPSHOT=ROOT/'third_party/capstone_automateData'
COLLECTION=ROOT/'data/drive_v2c30_20261012/merged_all_v1'
CANDIDATE='f1518a2141cf382cafa5e893e12fe698470d0b5c2d62fb33500f8774d7546ac3'
FILES=('scripts/evaluate_inference_cadence.py','scripts/rollout_smolvla.py',
    'learning/vla/smolvla_policy.py','learning/vla/inference_cadence.py',
    'learning/vla/smolvla_data.py','learning/vla/physical_grasp.py',
    'learning/vla/pickup_evaluation.py','learning/vla/expert_overlay.py',
    'learning/vla/training_precision.py','learning/vla/rollout_video.py',
    'learning/vla/inference_inputs.py')


def audit_condition(folder,case,plan):
    import numpy as np
    import av
    report=read_json(folder/'report.json'); manifest=read_json(folder/'manifest.json')
    if (report['mode']!=case['mode'] or report['execution_prefix']!=case['prefix'] or report['policy_seed']!=1000
            or report['inference_noise_schedule']!=case['noise_schedule'] or report['recorded_input_assistance']
            or report['simulator_state_forced_to_recording'] or report['reset_state_max_error']!=0
            or report['source_data_sha256']!=plan['source_data_sha256'][case['episode']]
            or report['checkpoint_weights_sha256']!=plan['base_sha256']
            or report['expert_overlay_weights_sha256']!=(CANDIDATE if case['mode']=='policy' else None)):
        raise ValueError('Evaluation binding differs from fixed plan')
    original=read_json(CAMPAIGN/('training_policy_'+case['episode'])/'manifest.json')
    if manifest!=original: raise ValueError('Environment, guards or grasp metrics changed')
    with np.load(folder/'trajectory.npz',allow_pickle=False) as trace:
        metrics=audit_pickup_trace(report,trace,read_json(folder/'diagnostics.json'))
        chunks=read_json(folder/'chunks.json')
        input_audit=report.get('inference_input_audit')
        if (case['mode']=='policy' and 'learning/vla/inference_inputs.py' in plan['sources']
                and input_audit is None):
            raise ValueError('Current protocol requires raw actor-input evidence')
        if case['mode']=='policy' and input_audit is not None:
            from learning.vla.inference_inputs import inference_input_fingerprint
            if (input_audit['observations_modified'] or not input_audit['per_chunk_fingerprints']
                    or input_audit['first_input_file']!='first_actor_input.npz'
                    or sha256(folder/'first_actor_input.npz')!=input_audit['first_input_sha256']
                    or input_audit['helper_sha256']!=plan['sources']['learning/vla/inference_inputs.py']['sha256']):
                raise ValueError('Raw actor-input audit binding differs')
            with np.load(folder/'first_actor_input.npz',allow_pickle=False) as raw:
                fingerprint=inference_input_fingerprint({'state':raw['state'],
                    'instruction':str(raw['instruction'].item()),
                    'images':{k:raw[k] for k in ('scene_rgb','wrist_rgb')}})
            if chunks[0]['input_fingerprint']!=fingerprint or any('input_fingerprint' not in c for c in chunks):
                raise ValueError('Initial or per-inference actor-input audit missing/different')
        cadence=audit_cadence_chunks(chunks,prefix=case['prefix'],steps=report['steps'],seed=1000,schedule=case['noise_schedule'])
        predictions=np.concatenate([np.asarray(c['prediction'])[:c['executed_steps']] for c in chunks])
        if not np.array_equal(predictions,trace['action_requested']): raise ValueError('Executed predictions differ from chunk prefix')
    if case['mode']=='replay' and (not metrics['pickup_success'] or max(report['replay_errors'].values())!=0):
        raise ValueError('Fresh exact expert control failed; stop before model conditions')
    with av.open(str(folder/'rollout.mp4')) as video:
        if video.streams.video[0].average_rate!=50: raise ValueError('Wrong video clock')
        frames=sum(1 for _ in video.decode(video=0))
    if frames!=report['steps']+1: raise ValueError('Video omits observations')
    return {**case,**metrics,'outcome':report['outcome'],'steps':report['steps'],
        'directory':str(folder),'cadence_audit':cadence,'safety_abort':report['safety_abort'],
        'latency_mean_s':report['latency_mean_s'],'latency_warm_mean_s':report['latency_warm_mean_s'],
        'wall_s':report['wall_s'],'video_frames':frames,
        'trajectory_sha256':sha256(folder/'trajectory.npz'),'video_sha256':sha256(folder/'rollout.mp4')}


def main():
    begun=time.monotonic()
    import numpy as np
    import psutil
    import av
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-from',type=Path,help='Continue the preserved initial resource stop into a NEW output folder')
    args=p.parse_args(); out=args.output.resolve()
    if out.parent!=ROOT/'outputs' or out.exists(): raise ValueError('New dedicated outputs folder required')
    if sha256(OVERLAY/'expert.safetensors')!=CANDIDATE: raise ValueError('Frozen actor changed')
    for proc in psutil.process_iter(['pid','cmdline']):
        try:
            if proc.pid not in [os.getpid(),*[a.pid for a in psutil.Process().parents()]] and any(
                    Path(a).name in ('rollout_smolvla.py','train_gripper_feedback.py','finetune_smolvla.py',
                                     'evaluate_inference_cadence.py') for a in (proc.info['cmdline'] or [])):
                raise RuntimeError('Another model job is already running')
        except (psutil.AccessDenied,psutil.NoSuchProcess): pass
    if shutil.disk_usage(ROOT).free<26*2**30: raise RuntimeError('Insufficient disk headroom, no automatic deletion')
    out.mkdir()
    parents=['train_20003','train_22045','train_20002']
    cases=[{'episode':e,'prefix':25,'mode':'replay','noise_schedule':'sequential'} for e in parents]
    cases += [{'episode':e,'prefix':pfx,'mode':'policy','noise_schedule':'step-keyed'}
              for pfx in (25,1,5) for e in parents]
    rows=[]; prior_s=0; resume=None; preserved={}
    if args.resume_from:
        import ast
        old=args.resume_from.resolve()
        if old!=ROOT/'outputs/inference_cadence_20261020': raise ValueError('Only the documented resource interruption is supported')
        old_plan=read_json(old/'plan.json'); failure=read_json(old/'failure.json')
        conditions=read_json(old/'conditions.json')
        resume=continuation_state(old_plan,failure,conditions)
        if old_plan['cases']!=cases or old_plan['candidate_sha256']!=CANDIDATE:
            raise ValueError('Original actor or case bank changed')
        for f,source in old_plan['sources'].items():
            original=verified_source(f,source['sha256'])
            if f not in ('scripts/evaluate_inference_cadence.py','learning/vla/inference_cadence.py') and sha256(ROOT/f)!=source['sha256']:
                raise ValueError('Actor/scene/observation implementation changed: '+f)
            if f=='learning/vla/inference_cadence.py':
                before=ast.parse(original.read_text()); after=ast.parse((ROOT/f).read_text())
                for name in ('anchor_noise_seed','audit_cadence_chunks'):
                    functions=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
                    if ast.dump(functions(before))!=ast.dump(functions(after)):
                        raise ValueError('Original noise or prefix-audit algorithm changed')
        for row in conditions:
            folder=Path(row['directory']).resolve()
            if folder.parent!=old or audit_condition(folder,{k:row[k] for k in ('episode','prefix','mode','noise_schedule')},old_plan)!=row:
                raise ValueError('Completed original condition no longer reproduces its audit')
        preserved={str(f.relative_to(old)):sha256(f) for f in old.rglob('*') if f.is_file()}
        rows=list(conditions); prior_s=resume['charged_prior_s']
        resume.update(original_directory=str(old),preserved_files=preserved,
            original_elapsed_s=failure['elapsed_s'],original_plan_sha256=sha256(old/'plan.json'),
            note='Completed control reused after full physical/video audit; interrupted next control restarted once in a new directory, not scored as failure. No policy trials existed in the original attempt.')
        write_json(out/'resume.json',resume)
    plan={'protocol':'fixed-actor-cadence-v1','parents':parents,'cases':cases,'policy_seed':1000,
        'candidate_sha256':CANDIDATE,'base_sha256':sha256(BASE/'model.safetensors'),
        'original_campaign_completion_sha256':sha256(CAMPAIGN/'completion.json'),
        'max_steps':1200,'total_limit_s':3600,'per_case_caps_s':{'replay':240,'prefix25':300,'prefix1':900,'prefix5':480},
        'noise':'Initial Gaussian flow noise uses an independent CUDA generator seeded by 1000+executed command index. Matching anchors share latent noise across prefixes; this does NOT reproduce old sequential-noise policy rollouts.',
        'control':'Three fresh expert controls must pass before all nine model conditions; one seed, training-only resets, all outcomes retained.',
        'changes':'Only executed prefix and the explicit common-anchor noise protocol; frozen 50-action model, same actual observations, action targets, controller, guards and pickup gate. No teacher, training, sensor perturbation or observation replacement.',
        'limits':'Paper-aligned per-action feedback in a custom OpenArm scene, NOT replication of LIBERO/MetaWorld results. Paused-time synchronous simulation, not feasible 50 Hz wall-clock inference or heldout generalization.',
        'sources':{f:archive_source(f) for f in FILES},
        'source_data_sha256':{e:sha256(COLLECTION/e/'episode.npz') for e in parents},
        'original_rollout_hashes':{e:sha256(CAMPAIGN/('training_policy_'+e)/'trajectory.npz') for e in parents}}
    if resume:
        plan['resume']={'original_directory':resume['original_directory'],'prior_charged_s':prior_s,
            'preparation_s':time.monotonic()-begun,'reused_conditions':len(rows),
            'resume_receipt_sha256':sha256(out/'resume.json')}
    write_json(out/'plan.json',plan)
    started=begun-prior_s; deadline=started+3600; stages=[]
    write_json(out/'conditions.json',rows)
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    keeper=subprocess.Popen([sys.executable,str(ROOT/'scripts/keep_training_awake.py'),'--pid',str(os.getpid()),
        '--max-seconds',str(max(1,int(deadline-time.monotonic())))],cwd=ROOT,creationflags=flags) if os.name=='nt' else None
    try:
        for case in cases[len(rows):]:
            remaining=deadline-time.monotonic()
            if remaining<180: raise TimeoutError('One-hour campaign reserve reached; remaining cases not run')
            for f,source in plan['sources'].items():
                if sha256(ROOT/f)!=source['sha256']: raise ValueError('Frozen source changed: '+f)
            if sha256(OVERLAY/'expert.safetensors')!=CANDIDATE: raise ValueError('Frozen actor changed')
            name=f"{case['mode']}_{case['episode']}_prefix{case['prefix']}"; folder=out/name
            cap=min(remaining-30,240 if case['mode']=='replay' else {25:300,1:900,5:480}[case['prefix']])
            cmd=[sys.executable,'-u',str(ROOT/'scripts/rollout_smolvla.py'),'--snapshot',str(SNAPSHOT),
                '--episode',str(COLLECTION/case['episode']),'--checkpoint',str(BASE),'--backbone',str(BACKBONE),
                '--output',str(folder),'--mode',case['mode'],'--max-steps','1200','--prefix',str(case['prefix']),
                '--seed','1000','--physical-grasp-metrics','--pickup-only','--noise-schedule',case['noise_schedule']]
            if case['mode']=='policy': cmd+=['--expert-overlay',str(OVERLAY)]
            tick=time.monotonic(); event={'name':name,'command':cmd,'cap_s':cap,'started_unix':time.time()}
            with (out/(name+'.log')).open('x',encoding='utf-8') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
                fired=threading.Event(); timer=arm_deadline(child,cap,fired)
                try:
                    while child.poll() is None:
                        gpu=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,memory.free','--format=csv,noheader,nounits'],
                            capture_output=True,text=True,timeout=15,check=True,creationflags=flags)
                        temp,free=map(int,gpu.stdout.strip().splitlines()[0].split(','))
                        resources={'temperature_c':temp,'gpu_free_mib':free,
                            'ram_available_bytes':psutil.virtual_memory().available,'disk_free_bytes':shutil.disk_usage(ROOT).free}
                        violations=resource_violations(**resources)
                        if violations:
                            write_json(out/'safety_stop.json',{'stage':name,'elapsed_s':time.monotonic()-started,
                                'resources':resources,'violations':violations,'child_pid':child.pid})
                            raise RuntimeError('; '.join(violations))
                        write_status(out/'health.json',{'running':True,'stage':name,'child_pid':child.pid,
                            'elapsed_s':time.monotonic()-started,'case_elapsed_s':time.monotonic()-tick,
                            'remaining_s':deadline-time.monotonic(),'temperature_c':temp,'free_gpu_mib':free,
                            'disk_free_gib':resources['disk_free_bytes']/2**30,
                            'ram_available_gib':resources['ram_available_bytes']/2**30,'completed_cases':len(rows)})
                        try: child.wait(timeout=10)
                        except subprocess.TimeoutExpired: pass
                    code=child.wait()
                finally:
                    timer.cancel()
                    if child.poll() is None: kill_owned_process(child)
                    child.wait()
            event.update(exit_code=code,hard_timeout=fired.is_set(),elapsed_s=time.monotonic()-tick)
            stages.append(event); write_json(out/'stages.json',stages)
            if code or fired.is_set(): raise RuntimeError('Case process failed: '+name+'; no automatic retry')
            row=audit_condition(folder,case,plan)
            rows.append(row); write_json(out/'conditions.json',rows)
            print(json.dumps({'completed_case':name,'outcome':row['outcome'],'steps':row['steps'],'elapsed_s':time.monotonic()-started}),flush=True)
        if sha256(CAMPAIGN/'completion.json')!=plan['original_campaign_completion_sha256']: raise ValueError('Original outcomes changed')
        for e,h in plan['original_rollout_hashes'].items():
            if sha256(CAMPAIGN/('training_policy_'+e)/'trajectory.npz')!=h: raise ValueError('Original trajectory changed')
        if resume:
            for relative,expected in preserved.items():
                if sha256(Path(resume['original_directory'])/relative)!=expected: raise ValueError('Preserved interrupted experiment changed')
        write_json(out/'completion.json',{'completed':True,'elapsed_s':time.monotonic()-started,'conditions':rows,
            'training_performed':False,'candidate_sha256':CANDIDATE,'limits':plan['limits']})
    except BaseException as exc:
        write_json(out/'failure.json',{'completed':False,'error':repr(exc),'elapsed_s':time.monotonic()-started,
            'conditions':rows,'remaining_cases_not_run':cases[len(rows):],'no_automatic_retry':True})
        raise
    finally:
        if keeper and keeper.poll() is None: keeper.terminate(); keeper.wait(timeout=10)
        write_status(out/'health.json',{'running':False,'completed':(out/'completion.json').exists(),
            'elapsed_s':time.monotonic()-started,'completed_cases':len(rows)})


if __name__=='__main__': main()
