"""Saved native-SFT metrics/plots/checkpoint selection; no model or optimizer calls."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.runtime_io import read_json, write_json
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source, verified_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new report folder')
    plan = read_json(args.campaign/'plan.json')
    source = archive_source('scripts/report_native_smolvla.py')
    if args.watch:
        health = read_json(args.campaign/'health.json')
        budget = min(14460, max(60, health.get('remaining_s', 14400)+60))
        deadline = time.monotonic()+budget
        while not any((args.campaign/name).exists() for name in ('completion.json','failure.json')):
            if time.monotonic() > deadline: raise TimeoutError('Report watcher deadline')
            write_json(args.campaign/'report_status.json', {'running': True, 'phase': 'waiting_for_terminal_status',
                'remaining_wait_s': deadline-time.monotonic(), 'output': str(args.output), 'source': source})
            time.sleep(10)
    terminal = 'completion.json' if (args.campaign/'completion.json').exists() else 'failure.json'
    result = read_json(args.campaign/terminal)
    for path, record in plan['sources'].items(): verified_source(path, record['sha256'])
    training = args.campaign/'training'
    events_file = training/'native_events.jsonl'
    events = [json.loads(line) for line in events_file.read_text().splitlines()] if events_file.exists() else []
    updates = [e for e in events if e['type']=='update']
    validations = [e for e in events if e['type']=='validation']
    if updates and [e['step'] for e in updates] != list(range(1,len(updates)+1)):
        raise ValueError('Skipped or repeated native update log')
    if any(not np.isfinite([e['loss'],e['grad_norm'],e['lr']]).all() for e in updates):
        raise ValueError('Nonfinite training metrics')
    if any(e['windows'] != 160 or e['episodes'] != 20 or not np.isfinite(e['loss']) for e in validations):
        raise ValueError('Unexpected validation scope/nonfinite loss')
    retained = []
    for folder in sorted((training/'checkpoints').glob('[0-9]*')):
        if not folder.is_dir() or not folder.name.isdecimal() or not (folder/'openarm_native_complete.json').exists(): continue
        receipt = read_json(folder/'openarm_native_complete.json')
        if not receipt['complete'] or sha256(folder/'pretrained_model/model.safetensors') != receipt['weights_sha256']:
            raise ValueError('Retained native checkpoint differs')
        retained.append({'checkpoint':str(folder/'pretrained_model'), 'step':receipt['step'],
                         'weights_sha256':receipt['weights_sha256'], 'validation':receipt.get('validation')})
    scored = [r for r in retained if r['validation'] is not None]
    selected = min(scored,key=lambda r:(r['validation']['loss'],-r['step'])) if scored else (
        max(retained,key=lambda r:r['step']) if retained else None)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'summary.json', {'campaign_terminal': result, 'update_count': len(updates),
        'validation_events': validations, 'retained_checkpoints': retained, 'selected_checkpoint': selected,
        'selection_criterion': 'Lowest fixed-window validation loss among retained checkpoints; latest fallback',
        'sources': {'reporter': source, 'campaign_plan_sha256':sha256(args.campaign/'plan.json'),
                    'native_events_sha256':sha256(events_file) if events_file.exists() else None},
        'limits':'Offline training/160 fixed-window validation evidence, not full-validation loss, autonomous pickup, insertion or generalization.'})
    for name, rows, fields in (
        ('training_metrics.csv',updates,['step','elapsed_s','loss','grad_norm','lr','update_seconds']),
        ('validation_metrics.csv',validations,['step','elapsed_s','loss','windows','episodes','seconds','scope'])):
        with (args.output/name).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    if updates:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        steps=np.array([e['step'] for e in updates]); losses=np.array([e['loss'] for e in updates])
        fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
        axes[0].plot(steps,losses,alpha=.2,linewidth=.5,color='#3865a5',label='Stochastic training flow loss')
        window=min(200,len(updates))
        axes[0].plot(steps[window-1:],np.convolve(losses,np.ones(window)/window,mode='valid'),
                     color='#204f83',label=f'{window}-update moving average')
        axes[0].set_ylabel('Training loss'); axes[0].legend()
        if validations:
            axes[1].plot([e['step'] for e in validations],[e['loss'] for e in validations],'o-',color='#268660')
        else:
            axes[1].text(.5,.5,'No validation event recorded',transform=axes[1].transAxes,ha='center')
        axes[1].set_ylabel('Fixed-window validation loss')
        axes[1].set_title('160 windows / 20 validation episodes; fixed flow time 0.5; not all frames',fontsize=10)
        axes[2].plot(steps,[e['lr'] for e in updates],color='#996326',label='Learning rate')
        axes[2].set(xlabel='Native optimizer update',ylabel='Learning rate')
        for ax in axes: ax.grid(alpha=.2)
        fig.suptitle('Native LeRobot SmolVLA fine-tuning | offline metrics'+(' | interrupted' if not result.get('completed') else ''))
        fig.tight_layout(); fig.savefig(args.output/'training_curves.png',dpi=160); fig.savefig(args.output/'training_curves.pdf'); plt.close(fig)
    write_json(args.campaign/'report_status.json', {'running': False, 'completed': True,
                                                  'output':str(args.output), 'update_count':len(updates)})
    print(json.dumps({'updates':len(updates),'validation_events':len(validations),'selected_checkpoint':selected}),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        if '--campaign' in sys.argv:
            campaign = Path(sys.argv[sys.argv.index('--campaign')+1])
            write_json(campaign/'report_status.json', {'running':False, 'completed':False, 'error':repr(error)})
        raise
