"""Bounded independent evidence watcher for the native one-episode test."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.append(str(ROOT/'.venv-smolvla/Lib/site-packages'))
from learning.vla.runtime_io import read_json,write_json
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source,verified_source
from learning.vla.native_evaluation import audit_native_trial


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--watch',action='store_true'); a=p.parse_args()
    if a.output.exists(): raise FileExistsError('New evidence folder required')
    plan=read_json(a.campaign/'plan.json'); source=archive_source('scripts/report_native_single_episode.py')
    if a.watch:
        remaining=read_json(a.campaign/'health.json').get('remaining_s',2700)
        deadline=time.monotonic()+min(2760,max(60,remaining+60))
        while not any((a.campaign/f).exists() for f in ('report.json','failure.json')):
            if time.monotonic()>deadline: raise TimeoutError('Evidence watcher deadline')
            write_json(a.campaign/'report_status.json',{'running':True,'phase':'waiting_for_terminal_status',
                'remaining_wait_s':deadline-time.monotonic(),'output':str(a.output),'source':source})
            time.sleep(10)
    result=read_json(a.campaign/('report.json' if (a.campaign/'report.json').exists() else 'failure.json'))
    for file,binding in plan['sources'].items(): verified_source(file,binding['sha256'])
    for row in result['rows']:
        checkpoint=Path(result['selected_checkpoint'] if row['mode']=='policy' else plan['checkpoint'])
        audit_plan={**plan,'weights_sha256':sha256(checkpoint/'model.safetensors')}
        case={k:row[k] for k in ('group','episode','mode','prefix','noise_schedule','numerics')}
        if audit_native_trial(row['directory'],case,audit_plan)!=row: raise ValueError('Independent trial audit differs')
    file=a.campaign/'training/native_events.jsonl'
    events=[json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []
    updates=[e for e in events if e['type']=='update']; probes=[e for e in events if e['type']=='single_episode_generated_probe']
    if [e['step'] for e in updates]!=list(range(1,len(updates)+1)): raise ValueError('Skipped/repeated update')
    if any(not np.isfinite([e['loss'],e['grad_norm'],e['lr']]).all() for e in updates): raise ValueError('Nonfinite training')
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'summary.json',{'campaign':result,'update_count':len(updates),
        'initial_probe':probes[0] if probes else None,'final_probe':probes[-1] if probes else None,
        'episode':plan['episode'],'reporter':source,'plan_sha256':sha256(a.campaign/'plan.json'),
        'native_events_sha256':sha256(file) if file.exists() else None,'limits':plan['limits']})
    with (a.output/'training_metrics.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['step','elapsed_s','loss','grad_norm','lr','update_seconds'],extrasaction='ignore')
        writer.writeheader(); writer.writerows(updates)
    with (a.output/'generated_action_metrics.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['step','phase','frame','finger_first5_rmse_mm','arm_first5_rmse_rad'])
        writer.writeheader()
        for event in probes:
            for row in event['rows']:
                e=row['errors']['5']; writer.writerow({'step':event['step'],'phase':row['phase'],'frame':row['frame'],
                    'finger_first5_rmse_mm':e['finger_rmse_mm'],'arm_first5_rmse_rad':e['arm_rmse_rad']})
    if updates:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
        x=np.array([e['step'] for e in updates]); y=np.array([e['loss'] for e in updates]); window=min(50,len(y))
        axes[0].plot(x,y,alpha=.2,linewidth=.6)
        axes[0].plot(x[window-1:],np.convolve(y,np.ones(window)/window,mode='valid'),label=f'{window}-update moving mean')
        axes[0].set_ylabel('Native training flow loss'); axes[0].legend(fontsize=8)
        for phase in ('descend','close'):
            points=[(e['step'],next(r for r in e['rows'] if r['phase']==phase)) for e in probes]
            axes[1].plot([s for s,r in points],[r['errors']['5']['finger_rmse_mm'] for s,r in points],'o-',label=phase)
            axes[2].plot([s for s,r in points],[r['errors']['5']['arm_rmse_rad'] for s,r in points],'o-',label=phase)
        axes[1].set_ylabel('Finger command RMSE (mm)'); axes[1].legend()
        axes[2].set(ylabel='Arm command RMSE (rad)',xlabel='Native optimizer update'); axes[2].legend()
        for ax in axes: ax.grid(alpha=.2)
        fig.suptitle('One-episode native fine-tune | recorded-observation probes, not validation or live success')
        fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(a.output/'training_curves.png',dpi=160)
        fig.savefig(a.output/'training_curves.pdf'); plt.close(fig)
    write_json(a.campaign/'report_status.json',{'running':False,'completed':True,'output':str(a.output),
        'update_count':len(updates),'campaign_completed':result['completed'],'pickup_success':result.get('pickup_success')})
    print({'updates':len(updates),'campaign_completed':result['completed'],'pickup_success':result.get('pickup_success')},flush=True)


if __name__=='__main__':
    try: main()
    except BaseException as error:
        if '--campaign' in sys.argv:
            campaign=Path(sys.argv[sys.argv.index('--campaign')+1])
            write_json(campaign/'report_status.json',{'running':False,'completed':False,'error':repr(error)})
        raise
