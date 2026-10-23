"""Audit saved native-policy rollouts and render evidence, with no model calls."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# System Python owns matplotlib; reuse the existing Python-3.11 PyAV wheel only.
# numpy/matplotlib remain from this interpreter. No install/package mutation.
sys.path.append(str(ROOT/'.venv-smolvla/Lib/site-packages'))
from learning.vla.runtime_io import read_json,write_json
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source,verified_source
from learning.vla.native_evaluation import audit_native_trial,training_gate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--watch',action='store_true'); args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new evidence directory')
    plan=read_json(args.campaign/'plan.json'); source=archive_source('scripts/report_native_evaluation.py')
    if sha256(ROOT/'learning/vla/native_evaluation.py')!=plan['sources']['learning/vla/native_evaluation.py']['sha256']:
        raise ValueError('Load the exact trial auditor used by this campaign')
    if args.watch:
        remaining=read_json(args.campaign/'health.json').get('remaining_s',3600)
        deadline=time.monotonic()+min(3660,max(60,remaining+60))
        while not any((args.campaign/n).exists() for n in ('report.json','failure.json')):
            if time.monotonic()>deadline: raise TimeoutError('Evaluation report watcher deadline')
            write_json(args.campaign/'report_status.json',{'running':True,'phase':'waiting_for_trials',
                'remaining_wait_s':deadline-time.monotonic(),'output':str(args.output),'source':source})
            time.sleep(10)
    terminal='report.json' if (args.campaign/'report.json').exists() else 'failure.json'
    result=read_json(args.campaign/terminal); rows=result['rows']
    for name,binding in plan['sources'].items(): verified_source(name,binding['sha256'])
    for row in rows:
        if sha256(Path(row['directory'])/'trajectory.npz')!=row['trajectory_sha256'] or sha256(Path(row['directory'])/'rollout.mp4')!=row['video_sha256']:
            raise ValueError('Scored motion/video changed')
        case={k:row[k] for k in ('group','episode','mode','prefix','noise_schedule','numerics')}
        if audit_native_trial(row['directory'],case,plan)!=row: raise ValueError('Independent trial audit differs')
    if sha256(Path(plan['checkpoint'])/'model.safetensors')!=plan['weights_sha256']:
        raise ValueError('Selected checkpoint changed')
    if shutil.disk_usage(ROOT).free<25*2**30: raise RuntimeError('Preserve 25 GiB disk reserve')
    counts={group:{'pickups':sum(r['pickup_success'] for r in rows if r['group']==group and r['mode']=='policy'),
                   'scored_trials':sum(r['group']==group and r['mode']=='policy' for r in rows),'planned_trials':5}
            for group in ('training','validation')}
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'summary.json',{'completed':result['completed'],'counts':counts,'rows':rows,
        'training_gate_passed':training_gate(rows),'checkpoint':plan['checkpoint'],'weights_sha256':plan['weights_sha256'],
        'conditional_validation_not_run':result.get('conditional_validation_not_run',[]),
        'failure':result.get('error'),'elapsed_s':result['elapsed_s'],'limits':plan['limits'],
        'reporter':source,'plan_sha256':sha256(args.campaign/'plan.json')})
    if rows:
        with (args.output/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
            fields=['group','episode','mode','outcome','pickup_success','safety_abort','steps','first_grasp_command',
                'success_command','maximum_lift_mm','maximum_continuous_carry_s','drops','first_prediction_repeat_max_abs','warm_inference_seconds','directory']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    policies=[r for r in rows if r['mode']=='policy']
    if policies:
        import av
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(len(policies),1,figsize=(11,max(3,2.6*len(policies))),squeeze=False)
        sheet,panels=plt.subplots(len(policies),3,figsize=(16,max(3,2.9*len(policies))),squeeze=False)
        for index,row in enumerate(policies):
            folder=Path(row['directory']); report=read_json(folder/'report.json')
            with np.load(folder/'trajectory.npz') as z:
                t=np.r_[0.,np.cumsum(z['executed_dt_s'])]
                address=report['pickup_evaluation']['plug_qpos_address']
                lift=(z['qpos'][:,address+2]-z['qpos'][0,address+2])*1000
                ax=axes[index,0]
                ax.plot(t,z['observation_state'][:,15]*1000,label='Measured finger travel (mm)',color='#315f9e')
                ax.plot(t,lift,label='Actual plug lift (mm)',color='#248b60')
            d=read_json(folder/'diagnostics.json')
            held=np.array([v['grasp_established'] and v['plug_finger_contacts']==2 for v in d])
            ax.fill_between(t,0,1,where=held,transform=ax.get_xaxis_transform(),color='#248b60',alpha=.15)
            ax.axhline(20,color='#248b60',linestyle=':',linewidth=1)
            ax.set_title(f"{row['group']} / {row['episode']} | {row['outcome']} | continuous carry {row['maximum_continuous_carry_s']:.2f}s",fontsize=10)
            ax.set(xlim=(0,24),ylabel='mm'); ax.grid(alpha=.2)
            wanted={0:'initial',row['steps']//2:'midpoint',row['steps']:'final'}; images={}
            with av.open(str(folder/'rollout.mp4')) as video:
                for frame_index,frame in enumerate(video.decode(video=0)):
                    if frame_index in wanted: images[frame_index]=frame.to_ndarray(format='rgb24')
            for col,(frame_index,label) in enumerate(wanted.items()):
                panels[index,col].imshow(images[frame_index]); panels[index,col].axis('off')
                panels[index,col].set_title(f"{row['episode']} | {label} {frame_index/50:.2f}s\n{row['outcome']}",fontsize=9)
        axes[0,0].legend(fontsize=8); axes[-1,0].set_xlabel('Simulated seconds; inference pauses omitted')
        fig.suptitle('Native SmolVLA physical pickup evaluation | selected seen training / conditional validation resets',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(args.output/'pickup_traces.png',dpi=160); fig.savefig(args.output/'pickup_traces.pdf'); plt.close(fig)
        sheet.suptitle('Actual closed-loop simulator frames | no recorded state/image assistance',fontsize=13)
        sheet.tight_layout(rect=(0,0,1,.97)); sheet.savefig(args.output/'rollout_contact_sheet.png',dpi=140); plt.close(sheet)
    write_json(args.campaign/'report_status.json',{'running':False,'completed':True,'counts':counts,'output':str(args.output)})
    print(counts,flush=True)


if __name__=='__main__':
    try: main()
    except BaseException as error:
        if '--campaign' in sys.argv:
            campaign=Path(sys.argv[sys.argv.index('--campaign')+1])
            write_json(campaign/'report_status.json',{'running':False,'completed':False,'error':repr(error)})
        raise
