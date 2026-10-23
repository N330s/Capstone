"""Re-audit saved takeover motion/labels/video, then plot without running simulation."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from learning.vla.pickup_evaluation import PickupGate
from learning.vla.teacher_takeover import check_prefix_state,command_owner
from learning.vla.runtime_io import write_json,read_json
from learning.vla.source_archive import archive_source,verified_source
from learning.vla.smolvla_data import sha256


def audit(root,output,initial=None):
    import av
    from PIL import Image,ImageDraw,ImageFont
    plan=read_json(root/'plan.json'); complete=read_json(root/'completion.json')
    if not complete['completed'] or not complete['all_reports_completed']: raise ValueError('Four complete test reports required')
    for source in plan['implementation']: verified_source(source['source'],source['sha256'])
    campaign=ROOT/'outputs/five_episode_openhold_20261020'
    if sha256(campaign/'training/expert/expert.safetensors') != plan['candidate_sha256']: raise ValueError('Candidate changed')
    output.mkdir(parents=True,exist_ok=False)
    rows=[]; frames=[]; comparisons=[]
    for case in plan['cases']:
        name=f"{case['episode']}_at_{case['anchor']}"; directory=root/name
        report=read_json(directory/'report.json'); manifest=read_json(directory/'manifest.json')
        prefix=read_json(directory/'prefix_audit.json'); diagnostics=read_json(directory/'diagnostics.json')
        parent=campaign/('training_policy_'+case['episode'])
        source=ROOT/'data/drive_v2c30_20261012/merged_all_v1'/case['episode']
        meta=read_json(source/'metadata.json')
        if sha256(source/'episode.npz')!=manifest['source_data_sha256'] or meta['data_sha256']!=manifest['source_data_sha256']:
            raise ValueError('Source episode changed')
        for file,expected in manifest['original_hashes'].items():
            if sha256(parent/file)!=expected: raise ValueError('Original rollout changed')
        original_manifest=read_json(parent/'manifest.json')
        if any(original_manifest[k]!=v for k,v in manifest['environment'].items()):
            raise ValueError('Scene/guards/camera/reset manifest differs from original')
        if manifest['grasp_metrics']!=original_manifest['evaluation_grasp_metrics']:
            raise ValueError('Grasp bookkeeping changed')
        with np.load(directory/'trajectory.npz',allow_pickle=False) as z:
            trace={k:z[k].copy() for k in z.files}
        with np.load(parent/'trajectory.npz',allow_pickle=False) as z:
            expected={k:z[k].copy() for k in ('qpos','qvel','action_requested','action_applied','observation_state','executed_dt_s')}
        n=report['steps']; anchor=case['anchor']; address=read_json(parent/'report.json')['pickup_evaluation']['plug_qpos_address']
        if len(diagnostics)!=n+1 or trace['qpos'].shape[0]!=n+1 or trace['action_requested'].shape!=(n,8):
            raise ValueError('Saved motion lengths differ')
        maxima={'qpos':0.,'qvel':0.}
        for j in range(anchor+1):
            error=check_prefix_state(trace['qpos'][j],trace['qvel'][j],expected['qpos'][j],expected['qvel'][j])
            maxima={k:max(maxima[k],error[k]) for k in error}
        for key in ('action_requested','action_applied','observation_state','executed_dt_s'):
            length=anchor+1 if key=='observation_state' else anchor
            if not np.array_equal(trace[key][:length],expected[key][:length]): raise ValueError('Saved prefix differs: '+key)
        if maxima!=prefix['maximum_state_errors'] or not prefix['passed']: raise ValueError('Prefix receipt differs')
        if not np.array_equal(trace['command_owner'],[command_owner(i,anchor) for i in range(n)]):
            raise ValueError('Incorrect command ownership labels')
        gate=PickupGate(float(trace['qpos'][0,address+2])); first=None
        if any(report['pickup_evaluation'].get(k)!=v for k,v in gate.definition().items()):
            raise ValueError('Pickup definition changed')
        for i,d in enumerate(diagnostics[1:]):
            if gate.update(plug_z_m=float(trace['qpos'][i+1,address+2]),grasp=d['grasp_established'],
                    contacts=d['plug_finger_contacts'],drops=d['drops'],safe=d['outcome'] in ('running','success'),
                    dt=float(trace['executed_dt_s'][i])) and first is None: first=i+1
        qualified=gate.passed
        if (qualified!=report['teacher_pickup_success'] or (qualified and first!=n)
                or abs(gate.maximum_lift_m-report['pickup_evaluation']['maximum_lift_m'])>1e-10):
            raise ValueError('Physical pickup summary differs from saved motion')
        eligibility=np.arange(n)>=anchor
        if not qualified: eligibility[:]=False
        if not np.array_equal(eligibility,trace['teacher_positive_candidate']): raise ValueError('Fault actions included in teacher labels')
        with np.load(directory/'recovery_observations.npz',allow_pickle=False) as z:
            if (not np.array_equal(z['observation_state'],trace['observation_state'][anchor:])
                    or not np.array_equal(z['action_requested'],trace['action_requested'][anchor:])
                    or not np.array_equal(z['executed_dt_s'],trace['executed_dt_s'][anchor:])
                    or not np.array_equal(z['phase'],trace['phase'][anchor:])
                    or not np.array_equal(z['time_s'],np.r_[0.,np.cumsum(trace['executed_dt_s'])][anchor:])
                    or bool(z['qualified_positive_candidate'])!=qualified
                    or str(z['instruction'])!=meta['instruction']):
                raise ValueError('Corrective observation/action alignment failed')
            for camera in ('image_scene_rgb','image_wrist_rgb'):
                if z[camera].shape!=(n-anchor+1,240,320,3) or z[camera].dtype!=np.uint8:
                    raise ValueError('Camera synchronization failed')
        picks={anchor:'takeover',n:'final'}; extracted={}; frame_count=0
        with av.open(str(directory/'rollout.mp4')) as container:
            if container.streams.video[0].average_rate!=50: raise ValueError('Wrong video clock')
            for i,frame in enumerate(container.decode(video=0)):
                frame_count+=1
                if i in picks:
                    image=frame.to_image()
                    destination=output/(name+'_'+picks[i]+'.png'); image.save(destination)
                    extracted[picks[i]]=destination.name
        if frame_count!=n+1: raise ValueError('Video omits or adds simulator observations')
        phase_names,phase_counts=np.unique(trace['phase'][anchor:],return_counts=True)
        row={**case,'outcome':report['outcome'],'teacher_pickup_success':qualified,
            'recovery_commands':n-anchor,'recovery_sim_s':float(sum(trace['executed_dt_s'][anchor:])),
            'maximum_lift_mm':gate.maximum_lift_m*1000,'continuous_carry_s':gate.maximum_s,
            'peak_robot_contact_n':report['peak_robot_contact_n'],'peak_support_contact_n':report['peak_support_contact_n'],
            'drops':report['drops'],'prefix_max_qpos_error':maxima['qpos'],'prefix_max_qvel_error':maxima['qvel'],
            'takeover_contacts':diagnostics[anchor]['plug_finger_contacts'],
            'takeover_finger_mm':float(trace['observation_state'][anchor,15]*1000),
            'takeover_lift_mm':float((trace['qpos'][anchor,address+2]-trace['qpos'][0,address+2])*1000),
            'teacher_first_grasp_command':next((i for i,d in enumerate(diagnostics) if d['grasp_established']),None),
            'recovery_phase_counts':{str(k):int(v) for k,v in zip(phase_names,phase_counts)},
            'positive_candidate_commands':int(eligibility.sum()),'video_frames':frame_count,
            'video_sha256':sha256(directory/'rollout.mp4'),'trace_sha256':sha256(directory/'trajectory.npz'),
            'observations_sha256':sha256(directory/'recovery_observations.npz'),'wall_s':report['wall_s']}
        rows.append(row); frames.append({'case':name,**extracted})
        if initial is not None:
            old=initial/name; old_report=read_json(old/'report.json')
            with np.load(old/'trajectory.npz',allow_pickle=False) as z:
                same={k:np.array_equal(z[k],trace[k]) for k in ('qpos','qvel','action_requested','action_applied')}
            if old_report['teacher_pickup_success'] and not all(same.values()):
                raise ValueError('Repair unexpectedly changed a previously successful teacher trajectory')
            teacher=read_json(directory/'teacher.json')['final']
            release=teacher.get('release_plan')
            comparison={'case':name,'initial_outcome':old_report['outcome'],'new_outcome':report['outcome'],
                'identical_full_motion':same,'initial_trace_sha256':sha256(old/'trajectory.npz'),
                'release_plan':release}
            if release is not None:
                start=next(i for i,p in enumerate(trace['phase']) if p=='takeover_retreat')
                cleared=next((i for i in range(start,len(diagnostics)) if diagnostics[i]['plug_finger_contacts']==0),None)
                if cleared is None: raise ValueError('New separation did not clear contact')
                force=max(d['grasp_contact_force_n'] for d in diagnostics[start:cleared+1])
                if force>release['force_abort_n']: raise ValueError('Observed separation force exceeds new teacher limit')
                comparison.update(first_retreat_command_index=start,first_clear_observation_index=cleared,
                    maximum_observed_release_force_n=force,
                    release_until_contact_clear_s=float(sum(trace['executed_dt_s'][start:cleared])))
            comparisons.append(comparison)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    canvas=Image.new('RGB',(1280,4*350+60),'#111e30'); draw=ImageDraw.Draw(canvas)
    draw.text((18,12),'DIAGNOSTIC TEACHER RECOVERIES | left: actual takeover | right: end | NOT VLA successes',font=font,fill='white')
    for i,item in enumerate(frames):
        y=60+i*350
        draw.text((15,y),item['case'],font=font,fill='white')
        for col,key in enumerate(('takeover','final')):
            with Image.open(output/item[key]) as im: canvas.paste(im.resize((640,312)),(640*col,y+30))
    canvas.save(output/'takeover_frames.png')
    summary={'audited':True,'experiment':str(root),'candidate_sha256':plan['candidate_sha256'],
        'teacher_successes':sum(r['teacher_pickup_success'] for r in rows),'test_count':len(rows),
        'positive_candidate_commands':sum(r['positive_candidate_commands'] for r in rows),
        'elapsed_s':complete['elapsed_s'],'cases':rows,'frames':frames,
        'initial_experiment':str(initial) if initial is not None else None,'repair_comparisons':comparisons,
        'report_source':archive_source('scripts/report_teacher_takeovers.py'),
        'limits':'Privileged teacher feasibility on two failed TRAINING resets x two anchors; correlated tests, not four independent environments. No new VLA inference, training, validation generalization, insertion or sim2real. Candidates have NOT been exported into a training dataset.'}
    write_json(output/'summary.json',summary)
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary),flush=True)


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=read_json(output/'summary.json'); root=Path(summary['experiment'])
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
    for ax,row in zip(axes.flat,summary['cases']):
        directory=root/f"{row['episode']}_at_{row['anchor']}"
        with np.load(directory/'trajectory.npz',allow_pickle=False) as z:
            state=z['observation_state']; dt=z['executed_dt_s']
            x=np.r_[0.,np.cumsum(dt)]; x-=x[row['anchor']]
            original=ROOT/'outputs/five_episode_openhold_20261020'/('training_policy_'+row['episode'])/'report.json'
            adr=read_json(original)['pickup_evaluation']['plug_qpos_address']
            lift=(z['qpos'][:,adr+2]-z['qpos'][0,adr+2])*1000
        d=read_json(directory/'diagnostics.json')
        carried=np.array([item['grasp_established'] and item['plug_finger_contacts']==2 for item in d])
        ax.plot(x,state[:,15]*1000,label='Measured finger travel',color='#315f9e')
        ax.plot(x,lift,label='Actual plug lift',color='#248b60',linewidth=2)
        ax.fill_between(x,0,1,where=carried,transform=ax.get_xaxis_transform(),color='#85c9a5',alpha=.18,label='Two contacts + grasp')
        ax.axvspan(x[0],0,color='#999999',alpha=.14,label='Replayed failed VLA prefix')
        ax.axvline(0,color='black',linestyle='--',linewidth=1,label='Teacher takeover')
        ax.axhline(20,color='#248b60',linestyle=':',linewidth=1,label='20 mm lift threshold')
        title=f"{row['episode']} | takeover after {row['anchor']} commands"
        ax.set(title=title,xlabel='Simulation seconds relative to takeover',ylabel='mm',ylim=(-3,80))
        ax.text(.98,.97,f"{row['outcome']}\ncarry {row['continuous_carry_s']:.2f}s, drops {row['drops']}",
            transform=ax.transAxes,ha='right',va='top',fontsize=8,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
        ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle(f"Teacher takeover feasibility: {summary['teacher_successes']}/4 recoveries | NOT autonomous VLA results",fontsize=15)
    fig.tight_layout(rect=(0,.09,1,.95))
    fig.savefig(output/'recovery_traces.png',dpi=180); fig.savefig(output/'recovery_traces.pdf'); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initial-experiment',type=Path)
    p.add_argument('--plot-only',action='store_true'); args=p.parse_args()
    if args.plot_only: plot(args.output)
    else: audit(args.experiment,args.output,args.initial_experiment)
