"""Audit collision diagnostics and visualize the original motion, never reruns policy."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source,verified_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnosis',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); root=args.diagnosis
    plan=json.loads((root/'plan.json').read_text()); report=json.loads((root/'report.json').read_text())
    probe=json.loads((root/'state_probe/report.json').read_text()); probe_plan=json.loads((root/'state_probe/plan.json').read_text())
    frames=json.loads((root/'frames_manifest.json').read_text())
    if not report['completed'] or not probe['completed'] or report['elapsed_s']>300 or probe['elapsed_s']>300:
        raise ValueError('Complete bounded diagnostic inputs required')
    campaign=Path(plan['campaign'])
    if (sha256(campaign/'training/expert/expert.safetensors')!=plan['candidate_sha256']
            or probe['candidate_sha256']!=plan['candidate_sha256']
            or sha256(campaign/'completion.json')!=plan['campaign_completion_sha256']):
        raise ValueError('Original model or outcomes changed')
    for name,source in plan['sources'].items(): verified_source(name,source['sha256'])
    verified_source(probe_plan['implementation']['source'],probe_plan['implementation']['sha256'])
    with np.load(root/'state_probe/predictions.npz',allow_pickle=False) as data:
        for row in probe['rows']:
            for variant,metrics in row['variants'].items():
                chunk=data[f"{row['episode']}_{row['anchor']}_{row['seed']}_{variant}"]
                if (chunk.shape!=(50,8) or not np.isfinite(chunk).all()
                        or abs(float(chunk[:5,7].mean()*1000)-metrics['first5_target_mm'])>1e-5
                        or abs(float(chunk[0,7]*1000)-metrics['first_target_mm'])>1e-5):
                    raise ValueError('Probe summary differs from saved predictions')
    rows=[]
    for row in report['rows']:
        parent=campaign/('training_policy_'+row['episode'])
        if sha256(parent/'trajectory.npz')!=row['trajectory_sha256']: raise ValueError('Original motion changed')
        contact=row['chronology']['first_one_or_two_contacts']
        m=next(m for m in row['measurements'] if m['command']==contact)
        pre=next(m for m in row['measurements'] if m['command']==275)
        rows.append({'episode':row['episode'],'outcome':row['outcome'],'steps':row['steps'],
            'precontact_275_finger_mm':pre['finger_mm'],
            'precontact_275_nearest_path_mm':pre['nearest_expert_preclose_path_mm'],
            'first_contact_command':contact,'finger_at_first_contact_mm':m['finger_mm'],
            'site_distance_at_first_contact_mm':m['hand_to_actual_plug_grasp_mm'],
            'first_two_contacts':row['chronology']['first_two_contacts'],'first_grasp':row['chronology']['first_grasp'],
            'maximum_lift_mm':row['chronology']['maximum_plug_lift_mm'],
            'robot_abort_n':row['unchanged_robot_abort_n'],'terminal_unwanted_force_n':row['terminal_unwanted_contact_n']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    colors={'train_20003':'#bd5944','train_22045':'#315f9e','train_20042':'#248b60'}
    for name,color in colors.items():
        row=next(r for r in report['rows'] if r['episode']==name)
        with np.load(root/name/'kinematics.npz',allow_pickle=False) as data:
            limit=min(len(data['finger_mm']),601); x=np.arange(limit)
            label=name+(' (pass)' if row['pickup_success'] else ' (fail)')
            axes[0,0].plot(x,data['finger_mm'][:limit],color=color,label=label)
            touch=row['chronology']['first_one_or_two_contacts']
            path_limit=min(limit,touch+1)
            axes[0,1].plot(np.arange(path_limit),data['nearest_path_m'][:path_limit]*1000,color=color,label=label)
            axes[0,1].scatter([touch],[data['nearest_path_m'][touch]*1000],color=color,s=35)
            axes[1,0].step(x,data['contacts'][:limit],where='post',color=color,label=label)
            axes[0,0].scatter([touch],[data['finger_mm'][touch]],color=color,marker='o',s=35,zorder=4)
    axes[0,0].axhspan(33,37,color='#b8c6ca',alpha=.25,label='Approx. augmented open-hold input range')
    axes[0,0].axhline(35,color='black',linestyle=':',linewidth=1)
    axes[0,0].set(xlabel='Executed command',ylabel='Actual measured finger travel (mm)',title='One-sided contact precedes gripper collapse')
    axes[0,1].set(xlabel='Executed command (curves stop at first contact)',ylabel='Nearest expert pre-close path (mm)',title='Pre-contact drift is not just a timing difference')
    axes[1,0].set(xlabel='Executed command',ylabel='Finger bodies touching plug',title='Failed runs never reach two-finger contact',yticks=[0,1,2])
    for ax in axes.flat[:3]: ax.grid(alpha=.2); ax.legend(fontsize=7)
    selected=[r for r in probe['rows'] if r['episode']=='train_22045' and r['anchor']==500]
    for i,variant in enumerate(('live','live_only_finger_recorded')):
        values=[r['variants'][variant]['first5_target_mm'] for r in selected]
        axes[1,1].scatter([i-.04,i+.04],values,s=55,label=variant)
        axes[1,1].hlines(np.mean(values),i-.18,i+.18,linewidth=2)
    axes[1,1].set(xticks=[0,1],xticklabels=['Actual finger state\n10.97 mm','Replace only finger state\n34.99 mm'],
        ylabel='Mean first-five predicted target (mm)',title='Same actual-render RGB + matched noise, two seeds')
    axes[1,1].grid(axis='y',alpha=.2)
    fig.suptitle('Five-parent VLA: failed approach → one-finger push → empty closure/self-contact\n'
        'Read-only existing traces and offline counterfactuals; no new training, controller fix or success trial',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(args.output/'failure_chain.png',dpi=160)
    fig.savefig(args.output/'failure_chain.pdf'); plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(16,10.5))
    for row,frame_row in zip(axes,frames['rows']):
        name=frame_row['episode']; video=campaign/('training_policy_'+name)/'rollout.mp4'
        if sha256(video)!=frame_row['video_sha256']: raise ValueError('Original video changed')
        by_label={r['label']:r for r in frame_row['frames']}
        for ax,label in zip(row,('first_contact','middle','terminal')):
            record=by_label[label]; ax.imshow(plt.imread(record['image'])); ax.axis('off')
            ax.set_title(f"{name}: {label.replace('_',' ')} at command {record['command']}",fontsize=9)
    fig.suptitle('Unchanged frames from original live-policy videos\n'
        'Top/middle: failed runs; bottom: successful comparison. Video footer shows final trial outcome on every frame.',fontsize=12)
    fig.subplots_adjust(top=.86,bottom=.03,left=.02,right=.98,hspace=.25,wspace=.04)
    fig.savefig(args.output/'original_video_frames.png',dpi=150); plt.close(fig)
    with (args.output/'events.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary={'completed':True,'cases':rows,'probe':probe,'kinematics_report_sha256':sha256(root/'report.json'),
        'probe_report_sha256':sha256(root/'state_probe/report.json'),'model_sha256':plan['candidate_sha256'],
        'implementation':archive_source('scripts/report_five_episode_collisions.py'),
        'limits':'Only existing five training resets, one execution seed. Counterfactuals use two explicit latent seeds, not original rollout latent stream. Same-index action discrepancies include phase drift; nearest path is Cartesian position only. No physical recovery/intervention was tested.'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'completed':True,'cases':rows},indent=2))


if __name__=='__main__': main()
