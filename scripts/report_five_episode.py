"""Independent log/trace audit and plots for the five-parent recipe-scaling pilot."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from learning.vla.five_episode import audit_updates
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import verified_source
from learning.vla.pickup_evaluation import audit_pickup_trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); root=args.campaign
    plan=json.loads((root/'plan.json').read_text()); selection=json.loads((root/'selection.json').read_text())
    completed=json.loads((root/'completion.json').read_text()); trained=json.loads((root/'training/report.json').read_text())
    if not completed['completed'] or trained['updates']!=4000 or sha256(root/'selection.json')!=plan['selection_sha256']:
        raise ValueError('Complete bound experiment required')
    for name,source in plan['sources'].items(): verified_source(name,source['sha256'])
    updates=list(map(json.loads,(root/'training/updates.jsonl').read_text().splitlines()))
    audit=audit_updates(updates,selection['parents'],4000)
    if sha256(root/'training/expert/expert.safetensors')!=trained['overlay']['weights_sha256']:
        raise ValueError('Candidate changed')
    for row in completed['conditions']:
        folder=Path(row['directory']); report=json.loads((folder/'report.json').read_text())
        with np.load(folder/'trajectory.npz',allow_pickle=False) as trace:
            metrics=audit_pickup_trace(report,trace,json.loads((folder/'diagnostics.json').read_text()))
        if any(row[k]!=v for k,v in metrics.items()): raise ValueError('Physical outcome differs')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    loss=np.asarray([r['loss'] for r in updates]); width=100
    axes[0].plot(np.arange(width,len(loss)+1),np.convolve(loss,np.ones(width)/width,'valid'))
    axes[0].set(xlabel='Optimizer updates',ylabel='Native loss, 100-update mean')
    checks=[trained['initial'],*trained['evaluations'][1:],trained['final']]
    for category in ('nominal','perturbed_open_hold'):
        axes[1].plot([c['step'] for c in checks],[c['metrics'][category]['gripper_mae_mm'] for c in checks],'o-',label=category)
    axes[1].set(xlabel='Optimizer updates',ylabel='Fixed training-probe finger MAE (mm)'); axes[1].legend(fontsize=8)
    for axis in axes[:2]: axis.grid(alpha=.2)
    axes[2].axis('off')
    lines=[f"Training-reset pickup: {completed['training_successes']}/5",f"Heldout: {completed['heldout_state']}",'']
    lines.extend(f"{r['episode']}: {r['outcome']}" for r in completed['conditions'] if r['group']=='training_policy')
    lines.extend(['','5 x 800 updates; 50% open-hold quota.','No insertion or sim-to-real claim.'])
    axes[2].text(.02,.95,'\n'.join(lines),va='top',fontsize=10)
    fig.suptitle('SmolVLA: scaling the successful open-hold recipe to five training episodes')
    fig.tight_layout(rect=(0,0,1,.93)); args.output.mkdir(parents=True,exist_ok=False)
    fig.savefig(args.output/'training_pickup.png',dpi=160); fig.savefig(args.output/'training_pickup.pdf'); plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps({'completed':True,'training_audit':audit,
        'outcomes':completed,'overlay_sha256':trained['overlay']['weights_sha256'],
        'completion_sha256':sha256(root/'completion.json'),'limits':plan['limits']},indent=2))


if __name__=='__main__': main()
