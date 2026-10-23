"""Bounded selected-checkpoint pickup evaluation; no training or recipe sweep."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from learning.vla.runtime_io import read_json,write_json,write_status
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source
from learning.vla.pickup_evaluation import select_validation_parents
from learning.vla.native_evaluation import TRAIN_PARENTS,training_gate,audit_native_trial
from scripts.rollout_smolvla import verify_snapshot
from scripts.run_smolvla_job import arm_deadline,kill_owned_process
from scripts.diagnose_cadence_repeatability import resources


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); output=args.output.resolve()
    if output.parent!=ROOT/'outputs' or output.exists(): raise ValueError('New dedicated evaluation directory required')
    if shutil.disk_usage(ROOT).free<26*2**30: raise RuntimeError('Insufficient artifact headroom above 25 GiB reserve')
    started=time.monotonic(); resources()
    summary_path=ROOT/'results/native_smolvla_20261021_v2/summary.json'
    summary=read_json(summary_path); selection=summary['selected_checkpoint']
    if not summary['campaign_terminal']['completed']: raise ValueError('Training did not complete')
    candidate=(ROOT/selection['checkpoint']).resolve()
    if candidate!=ROOT/'outputs/native_smolvla_20261021_v2/training/checkpoints/012243/pretrained_model':
        raise ValueError('Declared selected checkpoint differs')
    complete=read_json(candidate.parent/'openarm_native_complete.json')
    if (not complete['complete'] or complete['step']!=12243 or not complete['fp32_trainable_storage']
            or selection['weights_sha256']!=complete['weights_sha256']
            or sha256(candidate/'model.safetensors')!=selection['weights_sha256']):
        raise ValueError('Selected native checkpoint changed')
    export=ROOT/'data/drive_v2c30_20261012/export_all_v1'
    collection=ROOT/'data/drive_v2c30_20261012/merged_all_v1'
    snapshot=ROOT/'third_party/capstone_automateData'
    receipt=read_json(export/'openarm_export.json')
    heldout=select_validation_parents(receipt)
    records=[('training',next(r for r in receipt['datasets']['train']['episodes'] if r['episode']==name))
             for name in TRAIN_PARENTS]+[('validation',r) for r in heldout]
    parents={}
    for group,record in records:
        folder=collection/record['episode']; meta=read_json(folder/'metadata.json')
        if meta['outcome']!='success' or sha256(folder/'episode.npz')!=record['data_sha256'] or meta['data_sha256']!=record['data_sha256']:
            raise ValueError('Recorded parent changed')
        verify_snapshot(snapshot,meta)
        keys=('schema','scene_sha256','robot_config','contact_config','action_names','state_order','seed','reset_options')
        parents[record['episode']]={**record,'group':group,'metadata_sha256':sha256(folder/'metadata.json'),
                                   'manifest_binding':{key:meta[key] for key in keys}}
    sources={name:archive_source(name) for name in ('scripts/evaluate_native_smolvla.py','scripts/rollout_smolvla.py',
        'learning/vla/native_evaluation.py','learning/vla/smolvla_policy.py','learning/vla/smolvla_data.py',
        'learning/vla/training_precision.py','learning/vla/expert_overlay.py','learning/vla/inference_inputs.py',
        'learning/vla/inference_cadence.py','learning/vla/pickup_evaluation.py','learning/vla/physical_grasp.py',
        'learning/vla/rollout_video.py','learning/vla/runtime_io.py','scripts/diagnose_cadence_repeatability.py')}
    cases=[]
    for group in ('training','validation'):
        for mode in ('replay','policy'):
            for name,record in parents.items():
                if record['group']==group:
                    cases.append({'group':group,'episode':name,'mode':mode,'prefix':5 if mode=='policy' else 25,
                        'noise_schedule':'step-keyed' if mode=='policy' else 'sequential',
                        'numerics':'controlled' if mode=='policy' else 'default'})
    output.mkdir()
    plan={'protocol':'native-smolvla-pickup-v1','output':str(output),'checkpoint':str(candidate),
        'weights_sha256':selection['weights_sha256'],'selection_report_sha256':sha256(summary_path),
        'checkpoint_files':{f.name:sha256(f) for f in candidate.iterdir() if f.is_file()},
        'export_sha256':sha256(export/'openarm_export.json'),'parents':parents,'cases':cases,'sources':sources,
        'total_limit_s':3600,'per_trial_limit_s':360,'max_commands':1200,'policy_seed':1000,
        'training_gate':'All five exact expert controls and five autonomous policy pickups, no safety aborts',
        'validation_selection':'Five evenly spaced lexicographic validation parents; fixed before model outcomes',
        'limits':'Selected seen training resets, then conditional same-scene validation pilot. Validation influenced checkpoint selection: not a blind test. Pickup only, not insertion, real-time or sim2real. No teacher/state assistance, retraining, checkpoint/prefix/noise sweep.'}
    write_json(output/'plan.json',plan)
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    keeper=subprocess.Popen([sys.executable,str(ROOT/'scripts/keep_training_awake.py'),
        '--pid',str(os.getpid()),'--max-seconds','3600'],creationflags=flags) if os.name=='nt' else None
    rows=[]; skipped=[]
    def frozen():
        for name,source in sources.items():
            if sha256(ROOT/name)!=source['sha256']: raise RuntimeError('Evaluation source changed: '+name)
    try:
        for index,case in enumerate(cases):
            if case['group']=='validation' and not training_gate(rows):
                skipped=cases[index:]; break
            frozen(); resources()
            remaining=3600-(time.monotonic()-started)
            if remaining<45: raise TimeoutError('Evaluation allowance exhausted; unrun trials are not failures')
            name=f"{case['group']}_{case['episode']}_{case['mode']}"
            folder=output/name
            command=[sys.executable,'-u',str(ROOT/'scripts/rollout_smolvla.py'),
                '--snapshot',str(snapshot),'--episode',str(collection/case['episode']),
                '--checkpoint',str(candidate),'--backbone',str(ROOT/'data/models/smolvlm_backbone'),
                '--output',str(folder),'--mode',case['mode'],'--max-steps','1200',
                '--prefix',str(case['prefix']),'--seed','1000','--noise-schedule',case['noise_schedule'],
                '--numerics',case['numerics'],'--physical-grasp-metrics','--pickup-only']
            if case['mode']=='policy': command+=['--verify-first-prediction']
            write_json(output/(name+'_request.json'),{'command':command,'case':case})
            with (output/(name+'.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
                fired=threading.Event(); timer=arm_deadline(child,min(360,remaining-15),fired)
                try:
                    while child.poll() is None:
                        resource=resources(); frozen()
                        write_status(output/'health.json',{'running':True,'stage':name,'pid':child.pid,
                            'completed_trials':len(rows),'elapsed_s':time.monotonic()-started,
                            'remaining_s':3600-(time.monotonic()-started),**resource})
                        time.sleep(3)
                    code=child.wait()
                finally:
                    if child.poll() is None: kill_owned_process(child)
                    child.wait(); timer.cancel()
            if code or fired.is_set(): raise RuntimeError('Trial process/audit incomplete: '+name)
            row=audit_native_trial(folder,case,plan); rows.append(row)
            write_json(output/'progress.json',{'rows':rows,'completed_trials':len(rows)})
            print({'finished':name,'outcome':row['outcome'],'pickup_success':row['pickup_success']},flush=True)
        frozen()
        for name,expected in plan['checkpoint_files'].items():
            if sha256(candidate/name)!=expected: raise ValueError('Selected checkpoint changed during evaluation')
        result={'completed':True,'rows':rows,'training_gate_passed':training_gate(rows),
            'conditional_validation_not_run':skipped,'elapsed_s':time.monotonic()-started,'limits':plan['limits']}
        write_json(output/'report.json',result)
        write_status(output/'health.json',{'running':False,'completed':True,'training_gate_passed':result['training_gate_passed'],
                                          'elapsed_s':result['elapsed_s'],'completed_trials':len(rows)})
    except BaseException as error:
        result={'completed':False,'error':repr(error),'rows':rows,'elapsed_s':time.monotonic()-started,
                'incomplete_trials_not_scored_as_task_failures':True}
        write_json(output/'failure.json',result); write_status(output/'health.json',{'running':False,**result}); raise
    finally:
        if keeper and keeper.poll() is None: keeper.terminate(); keeper.wait(timeout=10)


if __name__=='__main__': main()
