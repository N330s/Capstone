"""Bounded scaling of the successful open-hold recipe to five training parents."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from learning.vla.runtime_io import read_json, write_json, write_status
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source
from learning.vla.five_episode import verify_selection, audit_updates
from learning.vla.pickup_evaluation import select_validation_parents, audit_pickup_trace
from learning.vla.job_monitoring import heartbeat_age_seconds
from scripts.run_smolvla_job import arm_deadline, kill_owned_process
from scripts.train_gripper_feedback import BASE, EXPORT, BACKBONE

COLLECTION = ROOT / 'data/drive_v2c30_20261012/merged_all_v1'
SNAPSHOT = ROOT / 'third_party/capstone_automateData'
REFERENCE = ROOT / 'outputs/gripper_control_fix_20261019/precision_repair/training/expert'
FILES = ('scripts/run_five_episode_openhold.py','scripts/train_gripper_feedback.py',
    'learning/vla/five_episode.py','learning/vla/gripper_feedback.py','learning/vla/expert_overlay.py',
    'learning/vla/training_precision.py','learning/vla/smolvla_policy.py','learning/vla/smolvla_compat.py',
    'scripts/rollout_smolvla.py','learning/vla/physical_grasp.py','learning/vla/pickup_evaluation.py',
    'scripts/report_five_episode.py')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot-python',type=Path,required=True)
    args=p.parse_args(); out=args.output.resolve()
    if out.parent != ROOT/'outputs' or not out.name.startswith('five_episode_openhold_') or (out/'plan.json').exists():
        raise ValueError('Prepared, never-launched dedicated campaign directory required')
    selection=read_json(out/'selection.json'); verify_selection(selection,EXPORT)
    if shutil.disk_usage(ROOT).free < 28*2**30:
        raise RuntimeError('28 GiB initial headroom required; no automatic cleanup')
    import psutil
    for proc in psutil.process_iter(['pid','cmdline']):
        try:
            if proc.pid not in [os.getpid(),*[a.pid for a in psutil.Process().parents()]] and any(
                Path(a).name in ('train_gripper_feedback.py','finetune_smolvla.py','rollout_smolvla.py','run_five_episode_openhold.py') for a in (proc.info['cmdline'] or [])):
                raise RuntimeError(f'Other training/rollout exists: {proc.pid}')
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    reference=read_json(REFERENCE/'overlay.json')
    from learning.vla.expert_overlay import verify_binding
    verify_binding(reference,BASE)
    if sha256(REFERENCE/'expert.safetensors') != 'ad9214130d4e3fbc4ba7f029e5e3f9bdfcbd2196c7edc1495410391f64cb46c6':
        raise ValueError('Known successful reference changed')
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    started=time.monotonic(); deadline=started+10800; stages=[]; rows=[]
    receipt=read_json(EXPORT/'openarm_export.json')
    heldout=select_validation_parents(receipt)
    plan={'protocol':'five-parent-openhold-scale-v1','supervisor_pid':os.getpid(),'started_unix':time.time(),
        'total_limit_s':10800,'training_limit_s':6000,'training_steps':4000,'updates_per_parent':800,
        'seed':1000,'batch_size':8,'learning_rate':1e-5,'training_initialization':'Original base, not successful reference or last 81-parent model',
        'base_weights_sha256':sha256(BASE/'model.safetensors'),'reference_overlay_sha256':reference['weights_sha256'],
        'selection_sha256':sha256(out/'selection.json'),'export_sha256':sha256(EXPORT/'openarm_export.json'),
        'training_parents':[r['episode'] for r in selection['parents']], 'validation_parents':[r['episode'] for r in heldout],
        'sampling':'One parent per batch, round-robin; four open-hold and four uniform anchors, +/-2mm sensor perturbations on first four only',
        'scheduler':'Original normalized warmup/cosine recipe stretched from 800 to 4000 optimizer steps; not the identical absolute-step schedule',
        'live_prefix':25,'policy_seed':1000,'max_commands':1200,'selection':'Final step 4000 only; no validation checkpoint choice or prefix sweep',
        'progression':'Fresh expert replays for all five, saved single-parent control, GPU preflight; then train/reload/test all five; heldout only if all five pass and time remains',
        'storage':'One FP32 expert-only inference overlay; no optimizer copy, no deletions, no optimizer-resume claim; 25 GiB reserve',
        'limits':'Recipe scaling and same-scene validation pilot, not causal isolation of augmentation versus sampling, blind testing, real-time, insertion or sim2real proof.',
        'sources':{name:archive_source(name) for name in FILES}}
    write_json(out/'plan.json',plan)
    keeper=subprocess.Popen([sys.executable,ROOT/'scripts/keep_training_awake.py','--pid',str(os.getpid()),'--max-seconds','10800'],
        cwd=ROOT,creationflags=flags) if os.name=='nt' else None

    def stage(name,cmd,cap,heartbeat=None):
        for name_,source in plan['sources'].items():
            if sha256(ROOT/name_) != source['sha256']:
                raise ValueError('Frozen source changed: '+name_)
        cap=min(cap,deadline-time.monotonic())
        if cap<60: raise RuntimeError('Three-hour cumulative deadline reached')
        event={'name':name,'command':list(map(str,cmd)),'started_unix':time.time(),'cap_s':cap}; tick=time.monotonic()
        with (out/(name+'.log')).open('x',encoding='utf-8') as log:
            child=subprocess.Popen(event['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
            fired=threading.Event(); timer=arm_deadline(child,cap,fired)
            try:
                while child.poll() is None:
                    if shutil.disk_usage(ROOT).free < 25*2**30 or psutil.virtual_memory().available < 1024**3:
                        raise RuntimeError('Disk/RAM reserve guard')
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,memory.free','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,check=True,timeout=15,creationflags=flags)
                    temp,free=map(int,gpu.stdout.strip().splitlines()[0].split(','))
                    if temp>=85: raise RuntimeError('GPU thermal guard')
                    paths=[out/(name+'.log')]
                    if heartbeat: paths.extend([heartbeat/'updates.jsonl',heartbeat/'binding.json'])
                    age=heartbeat_age_seconds(paths,stage_elapsed_s=time.monotonic()-tick,now_unix=time.time())
                    if age>600: raise RuntimeError('Stage heartbeat stale for ten minutes')
                    update=None
                    if heartbeat and (heartbeat/'updates.jsonl').exists():
                        lines=(heartbeat/'updates.jsonl').read_text().splitlines()
                        try: update=json.loads(lines[-1]) if lines else None
                        except json.JSONDecodeError: pass
                    write_status(out/'health.json',{'running':True,'stage':name,'supervisor_pid':os.getpid(),'child_pid':child.pid,
                        'elapsed_s':time.monotonic()-started,'remaining_s':deadline-time.monotonic(),'temperature_c':temp,
                        'free_gpu_mib':free,'disk_free_gib':shutil.disk_usage(ROOT).free/2**30,'heartbeat_age_s':age,'latest_update':update})
                    if name=='preflight' and update and update['step']>=1 and free<1024:
                        raise RuntimeError('Less than 1 GiB GPU headroom under gradients')
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: pass
                code=child.wait()
            finally:
                timer.cancel()
                if child.poll() is None: kill_owned_process(child)
                child.wait()
        event.update(returncode=code,timed_out=fired.is_set(),elapsed_s=time.monotonic()-tick)
        stages.append(event); write_json(out/'stages.json',stages)
        if code or fired.is_set(): raise RuntimeError('Stage failed: '+name+'; no automatic retry')

    def command(script,*extra): return [sys.executable,'-u',ROOT/'scripts'/script,*extra]

    def rollout(record,group,overlay=None):
        mode='replay' if group.endswith('expert') else 'policy'
        name=group+'_'+record['episode']; folder=out/name
        cmd=command('rollout_smolvla.py','--snapshot',SNAPSHOT,'--episode',COLLECTION/record['episode'],
            '--checkpoint',BASE,'--backbone',BACKBONE,'--output',folder,'--mode',mode,'--max-steps','1200',
            '--prefix','25','--seed','1000','--physical-grasp-metrics','--pickup-only')
        if overlay: cmd.extend(['--expert-overlay',overlay])
        stage(name,cmd,420)
        r=read_json(folder/'report.json')
        expected_overlay=read_json(overlay/'overlay.json')['weights_sha256'] if overlay else None
        if (not r['pipeline_passed'] or r['recorded_input_assistance'] or r['simulator_state_forced_to_recording']
                or r['reset_state_max_error'] != 0 or r['mode'] != mode or r['execution_prefix'] != 25 or r['policy_seed'] != 1000
                or r['source_data_sha256'] != record['data_sha256'] or r['checkpoint_weights_sha256'] != plan['base_weights_sha256']
                or r['expert_overlay_weights_sha256'] != expected_overlay
                or r['actuation_filter'] != {'gripper_resolution_mm':0.,'reference_or_phase_inputs':False,'arm_targets_changed':False}
                or r['grasp_metrics']['extension_sha256'] != plan['sources']['learning/vla/physical_grasp.py']['sha256']
                or r['pickup_evaluation']['helper_sha256'] != plan['sources']['learning/vla/pickup_evaluation.py']['sha256']):
            raise ValueError('Live actor/source/reset/metric binding failed')
        import numpy as np
        import av
        with np.load(folder/'trajectory.npz',allow_pickle=False) as trace:
            metrics=audit_pickup_trace(r,trace,read_json(folder/'diagnostics.json'))
        if mode=='replay' and max(r['replay_errors'].values()) != 0: raise ValueError('Expert replay diverged')
        with av.open(str(folder/'rollout.mp4')) as video: frames=sum(1 for _ in video.decode(video=0))
        if frames != r['steps']+1: raise ValueError('Incomplete rollout video')
        row={'episode':record['episode'],'group':group,'mode':mode,'directory':str(folder),'steps':r['steps'],
             'outcome':r['outcome'],'safety_abort':r['safety_abort'],**metrics}
        rows.append(row); write_json(out/'conditions.json',rows)
        return row

    try:
        for record in selection['parents']:
            if not rollout(record,'training_expert')['pickup_success']:
                raise RuntimeError('Selected expert failed; do not train on unqualified cohort')
        if not rollout(selection['parents'][0],'single_parent_reference',REFERENCE)['pickup_success']:
            raise RuntimeError('Known successful live control did not reproduce; stop before training')
        common=['--mode','augmented','--selection',out/'selection.json']
        stage('preflight',command('train_gripper_feedback.py','--output',out/'preflight',*common,
            '--steps','12','--max-seconds','480','--qualify-only'),540,out/'preflight')
        pre=read_json(out/'preflight/report.json')
        pre_rows=[json.loads(line) for line in (out/'preflight/updates.jsonl').read_text().splitlines()]
        if not pre['completed'] or not pre['qualification_only'] or pre['overlay'] is not None: raise ValueError('Preflight completion differs')
        write_json(out/'qualification.json',{'passed':True,'sampling_audit':audit_updates(pre_rows,selection['parents'],12),
            'note':'Fresh preflight uses a 12-step schedule; GPU/contract qualification, not an exact trajectory/resume test of the 4000-step schedule.'})
        stage('training',command('train_gripper_feedback.py','--output',out/'training',*common,
            '--steps','4000','--max-seconds','5900'),6000,out/'training')
        trained=read_json(out/'training/report.json')
        updates=[json.loads(line) for line in (out/'training/updates.jsonl').read_text().splitlines()]
        audit=audit_updates(updates,selection['parents'],4000)
        if (not trained['completed'] or trained['stop_reason']!='step_budget'
                or set(audit['updates_per_parent'].values())!={800} or trained['binding']['base_sha256']!=plan['base_weights_sha256']):
            raise ValueError('Fixed five-parent training budget/initialization differs')
        write_json(out/'training_audit.json',audit)
        stage('reload_verification',command('train_gripper_feedback.py','--output',out/'training',*common,
            '--steps','4000','--max-seconds','5900','--verify'),300)
        candidate=out/'training/expert'
        train_policy=[rollout(r,'training_policy',candidate) for r in selection['parents']]
        heldout_state='not_run_training_gate_failed'
        if all(r['pickup_success'] for r in train_policy):
            heldout_state='not_run_remaining_budget' if deadline-time.monotonic()<2820 else 'completed'
            if heldout_state=='completed':
                for record in heldout:
                    if not rollout(record,'heldout_expert')['pickup_success']: raise RuntimeError('Heldout expert failed')
                    rollout(record,'heldout_policy',candidate)
        for name,source in plan['sources'].items():
            if sha256(ROOT/name)!=source['sha256']: raise ValueError('Frozen implementation changed')
        write_json(out/'completion.json',{'completed':True,'updates':4000,'training_successes':sum(r['pickup_success'] for r in train_policy),
            'training_trials':5,'heldout_state':heldout_state,'conditions':rows,'elapsed_s':time.monotonic()-started,'limits':plan['limits']})
        stage('report',[args.plot_python,ROOT/'scripts/report_five_episode.py','--campaign',out,'--output',ROOT/'results'/out.name],180)
        write_json(out/'report_generation.json',{'completed':True,'completion_sha256':sha256(out/'completion.json')})
    except BaseException as error:
        write_json(out/'failure.json',{'completed':False,'error':repr(error),'conditions':rows,'stages':stages,
            'elapsed_s':time.monotonic()-started,'no_automatic_retry':True})
        raise
    finally:
        write_status(out/'health.json',{'running':False,'supervisor_pid':os.getpid(),'elapsed_s':time.monotonic()-started,
            'completed':(out/'completion.json').exists() and not (out/'failure.json').exists()})
        if keeper and keeper.poll() is None:
            keeper.terminate(); keeper.wait(timeout=10)


if __name__=='__main__': main()
