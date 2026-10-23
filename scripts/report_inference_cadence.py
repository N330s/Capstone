"""Read audited cadence conditions and plot them; never runs a model or simulator."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from learning.vla.runtime_io import read_json,write_json
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source,verified_source
from learning.vla.pickup_evaluation import audit_pickup_trace
from learning.vla.inference_cadence import audit_cadence_chunks,continuation_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-incomplete',action='store_true'); args=p.parse_args()
    plan=read_json(args.campaign/'plan.json')
    terminal=args.campaign/'completion.json'
    completed=terminal.exists() and read_json(terminal)['completed']
    if not completed and not args.allow_incomplete: raise ValueError('Campaign unfinished; opt in to an explicitly incomplete report')
    for f,s in plan['sources'].items(): verified_source(f,s['sha256'])
    if sha256(ROOT/'outputs/five_episode_openhold_20261020/training/expert/expert.safetensors')!=plan['candidate_sha256']:
        raise ValueError('Frozen actor changed')
    rows=read_json(args.campaign/'conditions.json') if (args.campaign/'conditions.json').exists() else []
    if 'resume' in plan:
        receipt=read_json(args.campaign/'resume.json'); old=Path(receipt['original_directory'])
        if sha256(args.campaign/'resume.json')!=plan['resume']['resume_receipt_sha256']:
            raise ValueError('Continuation receipt changed')
        if sha256(old/'plan.json')!=receipt['original_plan_sha256']:
            raise ValueError('Original interrupted plan changed')
        old_rows=read_json(old/'conditions.json')
        ledger=continuation_state(read_json(old/'plan.json'),read_json(old/'failure.json'),old_rows)
        if (ledger['charged_prior_s']!=plan['resume']['prior_charged_s']
                or rows[:len(old_rows)]!=old_rows):
            raise ValueError('Prior conditions or cumulative clock not preserved')
        for relative,expected in receipt['preserved_files'].items():
            if sha256(old/relative)!=expected: raise ValueError('Original interrupted artifact changed')
        if completed and not plan['resume']['prior_charged_s']+plan['resume']['preparation_s']<=read_json(terminal)['elapsed_s']<=3600:
            raise ValueError('Cumulative one-hour budget differs')
    if [(r['episode'],r['prefix'],r['mode']) for r in rows]!=[(c['episode'],c['prefix'],c['mode']) for c in plan['cases'][:len(rows)]]:
        raise ValueError('Reported conditions differ from predeclared order')
    for row in rows:
        folder=Path(row['directory']); report=read_json(folder/'report.json')
        if sha256(folder/'trajectory.npz')!=row['trajectory_sha256'] or sha256(folder/'rollout.mp4')!=row['video_sha256']:
            raise ValueError('Scored motion/video changed')
        with np.load(folder/'trajectory.npz',allow_pickle=False) as trace:
            audit=audit_pickup_trace(report,trace,read_json(folder/'diagnostics.json'))
        if any(row[k]!=v for k,v in audit.items()): raise ValueError('Saved physical metrics differ')
        audit_cadence_chunks(read_json(folder/'chunks.json'),prefix=row['prefix'],steps=row['steps'],
            seed=plan['policy_seed'],schedule=row['noise_schedule'])
    pairs=[]
    for parent in plan['parents']:
        first={r['prefix']:read_json(Path(r['directory'])/'chunks.json')[0]['prediction'] for r in rows
               if r['mode']=='policy' and r['episode']==parent}
        if first:
            reference=np.asarray(next(iter(first.values())))
            differences={str(k):float(np.max(np.abs(np.asarray(v)-reference))) for k,v in first.items()}
            if max(differences.values())>1e-6: raise ValueError('Initial same-state/same-noise predictions differ across prefixes')
            pairs.append({'episode':parent,'initial_full_chunk_max_abs_differences':differences})
    args.output.mkdir(parents=True,exist_ok=False)
    policies=[r for r in rows if r['mode']=='policy']
    counts={str(prefix):{'pickups':sum(r['pickup_success'] for r in policies if r['prefix']==prefix),
        'scored_trials':sum(r['prefix']==prefix for r in policies),'planned_trials':3} for prefix in (25,1,5)}
    summary={'completed':completed,'conditions':rows,'prefix_counts':counts,'same_initial_prediction_audit':pairs,
        'candidate_sha256':plan['candidate_sha256'],'campaign':str(args.campaign),
        'report_source':archive_source('scripts/report_inference_cadence.py'),
        'limits':plan['limits']+' Single noise protocol on three selected seen resets; not nine independent tasks. Missing conditions are pending/not run, never scored failures. Fresh step-keyed prefix-25 results do not overwrite the historical sequential-noise 3/5 result.'}
    write_json(args.output/'summary.json',summary)
    if rows:
        fields=['episode','mode','prefix','noise_schedule','outcome','pickup_success','steps','first_grasp_command',
            'maximum_lift_mm','maximum_continuous_carry_s','drops','latency_warm_mean_s','wall_s']
        with (args.output/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    if not policies:
        print('No model conditions scored; no outcome figure generated.',flush=True)
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(13,9),sharex=True,sharey=True)
    for ri,parent in enumerate(plan['parents']):
        for ci,prefix in enumerate((25,1,5)):
            ax=axes[ri,ci]
            row=next((r for r in policies if r['episode']==parent and r['prefix']==prefix),None)
            ax.set_title(f'{parent} | prefix {prefix} ({prefix*.02:.2f}s)',fontsize=10)
            if row:
                report=read_json(Path(row['directory'])/'report.json')
                with np.load(Path(row['directory'])/'trajectory.npz') as z:
                    t=np.r_[0.,np.cumsum(z['executed_dt_s'])]
                    address=report['pickup_evaluation']['plug_qpos_address']
                    lift=(z['qpos'][:,address+2]-z['qpos'][0,address+2])*1000
                    ax.plot(t,z['observation_state'][:,15]*1000,label='Measured finger travel',color='#315f9e')
                    ax.plot(t,lift,label='Actual plug lift',color='#248b60')
                d=read_json(Path(row['directory'])/'diagnostics.json')
                grasp=np.array([i['grasp_established'] and i['plug_finger_contacts']==2 for i in d])
                ax.fill_between(t,0,1,where=grasp,transform=ax.get_xaxis_transform(),alpha=.15,color='#248b60')
                ax.text(.04,.96,f"{row['outcome']}\ncarry {row['maximum_continuous_carry_s']:.2f}s",transform=ax.transAxes,
                    va='top',fontsize=8,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
            else:
                ax.text(.5,.5,'PENDING / NOT RUN',transform=ax.transAxes,ha='center',color='gray')
            ax.axhline(20,color='#248b60',linestyle=':',linewidth=1)
            ax.set(xlim=(0,24),ylim=(-3,90)); ax.grid(alpha=.2)
            if ci==0: ax.set_ylabel('mm')
            if ri==2: ax.set_xlabel('Simulated seconds (inference pauses omitted)')
    fig.suptitle('Frozen SmolVLA execution-cadence test | THREE TRAINING RESETS'+(' | INCOMPLETE' if not completed else ''),fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(args.output/'cadence_traces.png',dpi=170); fig.savefig(args.output/'cadence_traces.pdf'); plt.close(fig)
    print(counts,flush=True)


if __name__=='__main__': main()
