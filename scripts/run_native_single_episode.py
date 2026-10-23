"""Bounded native one-episode fine-tune, then fresh live pickup; no sweeps/deletion."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from learning.vla.runtime_io import read_json,write_json,write_status
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source
from learning.vla.native_single_episode import episode_selection
from learning.vla.native_evaluation import audit_native_trial
from scripts.diagnose_cadence_repeatability import resources
from scripts.run_smolvla_job import arm_deadline,kill_owned_process


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prior-campaign',type=Path,help='Continue only a pre-update failed preflight; preserve elapsed budget/control')
    a=p.parse_args(); output=a.output.resolve(); name='train_20002'
    if output.parent!=ROOT/'outputs' or output.exists(): raise ValueError('New dedicated output required')
    resources()
    if shutil.disk_usage(ROOT).free<27*2**30: raise RuntimeError('Need final-checkpoint room above 25 GiB reserve')
    charged=0.; prior=None
    if a.prior_campaign:
        prior=a.prior_campaign.resolve(); failure=read_json(prior/'failure.json'); previous=read_json(prior/'plan.json')
        stages=read_json(prior/'stages.json')
        if (prior.parent!=ROOT/'outputs' or previous['protocol']!='native-single-episode-v1'
                or failure['completed'] or not stages or stages[-1]['stage']!='preflight'
                or any(s['stage']=='training' for s in stages)):
            raise ValueError('Only a stopped pre-update preflight may continue')
        events=prior/'preflight/native_events.jsonl'
        if events.exists() and any(json.loads(line)['type']=='update' for line in events.read_text().splitlines()):
            raise ValueError('Do not retry a preflight that already updated weights')
        charged=failure['elapsed_s']+max(0,time.time()-(prior/'failure.json').stat().st_mtime)
    begun=time.monotonic()-charged; deadline=begun+2700
    if deadline-time.monotonic()<900: raise TimeoutError('Original allowance insufficient after repair')
    old=read_json(ROOT/'outputs/native_pickup_eval_20261022/plan.json')
    initializer=Path(old['checkpoint']); export=ROOT/'data/drive_v2c30_20261012/export_all_v1'
    receipt=read_json(export/'openarm_export.json'); index,record=episode_selection(receipt,name)
    if sha256(initializer/'model.safetensors')!=old['weights_sha256']: raise ValueError('Initializer changed')
    source_names=list(old['sources'])+['scripts/run_native_single_episode.py','scripts/train_native_smolvla.py',
        'learning/vla/native_single_episode.py','learning/vla/native_smolvla.py','scripts/qualify_native_smolvla.py',
        'scripts/diagnose_native_actions.py']
    sources={n:archive_source(n) for n in dict.fromkeys(source_names)}
    bindings={str(f.relative_to(ROOT)):sha256(f) for f in list(initializer.iterdir()) if f.is_file()}
    for f in (export/'openarm_export.json',export/'train/meta/stats.json'):
        bindings[str(f.relative_to(ROOT))]=sha256(f)
    plan={'protocol':'native-single-episode-v1','output':str(output),'checkpoint':str(initializer),
        'weights_sha256':old['weights_sha256'],'parents':{name:old['parents'][name]},'sources':sources,
        'bindings':bindings,'episode':name,'native_episode_index':index,'frames':record['steps'],
        'total_limit_s':2700,'training_limit_s':1800,'training_step_ceiling':2000,'batch_size':8,
        'normalization':'unchanged full-training-split saved statistics',
        'initialization':'warm-start native step12243 weights; new native optimizer/scheduler, not optimizer resume',
        'recipe':'native uniform shuffle within one complete recorded episode; native policy/loss/optimizer/scheduler',
        'evaluation':'fresh exact expert control, then final checkpoint, live observations, prefix5/horizon50, seed1000 step-keyed controlled numerics',
        'limits':'One fixed training reset memorization test, not generalization/insertion/RLT/sim2real. No controller assistance, data changes, parameter sweep, checkpoint deletion or automatic retry.'}
    if prior:
        if bindings!=previous['bindings']: raise ValueError('Initializer/data changed since failed preflight')
        plan['continuation']={'prior':str(prior),'charged_prior_and_repair_s':charged,
            'reason':'Audit accepts native n_obs_steps=1 history axis; no training input/model change',
            'prior_failure_sha256':sha256(prior/'failure.json')}
    output.mkdir(); write_json(output/'plan.json',plan)
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    keeper=subprocess.Popen([sys.executable,str(ROOT/'scripts/keep_training_awake.py'),'--pid',str(os.getpid()),
        '--max-seconds','2700'],creationflags=flags) if os.name=='nt' else None
    stages=[]; rows=[]
    def frozen():
        for file,binding in sources.items():
            if sha256(ROOT/file)!=binding['sha256']: raise RuntimeError('Source changed: '+file)
    def run(stage,command,cap):
        frozen(); resources(); remaining=deadline-time.monotonic()
        if remaining<45: raise TimeoutError('Campaign budget exhausted')
        started=time.monotonic()
        with (output/(stage+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
            fired=threading.Event(); timer=arm_deadline(child,min(cap,remaining-10),fired)
            try:
                while child.poll() is None:
                    frozen(); state=resources()
                    write_status(output/'health.json',{'running':True,'stage':stage,'pid':child.pid,
                        'elapsed_s':time.monotonic()-begun,'remaining_s':deadline-time.monotonic(),**state})
                    time.sleep(3)
                code=child.wait()
            finally:
                if child.poll() is None: kill_owned_process(child)
                child.wait(); timer.cancel()
        stages.append({'stage':stage,'command':command,'elapsed_s':time.monotonic()-started,
                       'returncode':code,'timed_out':fired.is_set()})
        write_json(output/'stages.json',stages)
        if code or fired.is_set(): raise RuntimeError('Stage failed: '+stage)
    def rollout(mode,checkpoint,folder):
        case={'group':'training','episode':name,'mode':mode,'prefix':5 if mode=='policy' else 25,
              'noise_schedule':'step-keyed' if mode=='policy' else 'sequential',
              'numerics':'controlled' if mode=='policy' else 'default'}
        command=[sys.executable,'-u',str(ROOT/'scripts/rollout_smolvla.py'),'--snapshot',str(ROOT/'third_party/capstone_automateData'),
            '--episode',str(ROOT/'data/drive_v2c30_20261012/merged_all_v1'/name),'--checkpoint',str(checkpoint),
            '--backbone',str(ROOT/'data/models/smolvlm_backbone'),'--output',str(folder),'--mode',mode,
            '--max-steps','1200','--prefix',str(case['prefix']),'--seed','1000','--noise-schedule',case['noise_schedule'],
            '--numerics',case['numerics'],'--physical-grasp-metrics','--pickup-only']
        if mode=='policy': command+=['--verify-first-prediction']
        run(folder.name,command,360)
        audit_plan={**plan,'checkpoint':str(checkpoint),'weights_sha256':sha256(checkpoint/'model.safetensors')}
        row=audit_native_trial(folder,case,audit_plan); rows.append(row)
        write_json(output/'evaluation_progress.json',{'rows':rows}); return row
    def train_command(destination,seconds,steps,save):
        return [sys.executable,'-u',str(ROOT/'scripts/train_native_smolvla.py'),'--native-max-seconds',str(seconds),
            '--native-single-episode',name,'--native-no-early-checkpoint',f'--policy.path={initializer}',
            '--policy.input_features={}',f'--policy.vlm_model_name={ROOT / "data/models/smolvlm_backbone"}',
            '--policy.load_vlm_weights=false','--policy.device=cuda','--policy.push_to_hub=false',
            '--policy.freeze_vision_encoder=true','--policy.train_expert_only=true','--policy.train_state_proj=true',
            '--policy.adapt_to_pi_aloha=false','--policy.use_delta_joint_actions_aloha=false','--policy.empty_cameras=0',
            f'--dataset.repo_id={receipt["datasets"]["train"]["repo_id"]}',f'--dataset.root={export / "train"}',
            f'--dataset.episodes=[{index}]','--dataset.video_backend=pyav','--dataset.image_transforms.enable=false',
            f'--output_dir={destination}',f'--steps={steps}','--batch_size=8','--num_workers=0','--seed=1000',
            '--eval_freq=0','--log_freq=50','--save_freq=5000',f'--save_checkpoint={str(save).lower()}','--wandb.enable=false']
    try:
        if prior:
            original=prior/('training_'+name+'_replay'); copied=output/original.name
            for key in old['sources']:
                if key!='scripts/evaluate_native_smolvla.py' and sha256(ROOT/key)!=previous['sources'][key]['sha256']:
                    raise ValueError('Expert-control implementation changed: '+key)
            case={'group':'training','episode':name,'mode':'replay','prefix':25,'noise_schedule':'sequential','numerics':'default'}
            audit_native_trial(original,case,previous)
            shutil.copytree(original,copied)
            expert=audit_native_trial(copied,case,plan); rows.append(expert)
            write_json(output/'reused_control.json',{'original':str(original),'row':expert,'hash_audited':True})
            write_json(output/'evaluation_progress.json',{'rows':rows})
        else: expert=rollout('replay',initializer,output/('training_'+name+'_replay'))
        if not expert['pickup_success']: raise ValueError('Fresh expert control must pass')
        run('preflight',train_command(output/'preflight',600,8,False),300)
        status=read_json(output/'preflight/native_status.json')
        if status['type']!='completed' or status['last_saved_step'] is not None: raise ValueError('Eight-update no-copy preflight incomplete')
        events=[json.loads(line) for line in (output/'preflight/native_events.jsonl').read_text().splitlines()]
        if len([e for e in events if e['type']=='update'])!=8: raise ValueError('Wrong preflight update count')
        seconds=min(1800,deadline-time.monotonic()-650)
        if seconds<300: raise TimeoutError('Insufficient training budget after checks')
        run('training',train_command(output/'training',seconds,2000,True),seconds+120)
        final=read_json(output/'training/native_status.json')
        if final['type']!='completed' or not final['last_saved_step']: raise ValueError('No final native checkpoint')
        saved=[f for f in (output/'training/checkpoints').iterdir() if f.is_dir()]
        if len(saved)!=1: raise ValueError('Expected one retained final checkpoint')
        checkpoint=saved[0]/'pretrained_model'; complete=read_json(saved[0]/'openarm_native_complete.json')
        if (not complete['complete'] or sha256(checkpoint/'model.safetensors')!=complete['weights_sha256']
                or not read_json(saved[0]/'single_episode_normalization.json')['unchanged']):
            raise ValueError('Final model/normalization receipt incomplete')
        write_json(output/'selected_checkpoint.json',{'checkpoint':str(checkpoint),**complete,'selection':'only final checkpoint; no sweep'})
        run('reload',[sys.executable,'-u',str(ROOT/'scripts/qualify_native_smolvla.py'),'--mode','reload',
            '--checkpoint',str(checkpoint),'--batch-size','1','--output',str(output/'reload')],240)
        if not read_json(output/'reload/report.json')['all_saved_tensors_reload_exact']: raise ValueError('Fresh reload failed')
        result=rollout('policy',checkpoint,output/('training_'+name+'_policy'))
        for file,binding in bindings.items():
            if sha256(ROOT/file)!=binding: raise ValueError('Initializer/data changed: '+file)
        write_json(output/'report.json',{'completed':True,'pickup_success':result['pickup_success'],
            'rows':rows,'training':final,'selected_checkpoint':str(checkpoint),'weights_sha256':complete['weights_sha256'],
            'elapsed_s':time.monotonic()-begun,'limits':plan['limits']})
        write_status(output/'health.json',{'running':False,'completed':True,'pickup_success':result['pickup_success'],
            'elapsed_s':time.monotonic()-begun})
    except BaseException as error:
        result={'completed':False,'error':repr(error),'rows':rows,'elapsed_s':time.monotonic()-begun,
                'automatic_retry':False,'incomplete_trials_not_task_failures':True}
        write_json(output/'failure.json',result); write_status(output/'health.json',{'running':False,**result}); raise
    finally:
        if keeper and keeper.poll() is None: keeper.terminate(); keeper.wait(timeout=10)


if __name__=='__main__': main()
