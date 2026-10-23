"""CPU-only command-pose analysis and evidence plots for the frozen action probe."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from learning.vla.runtime_io import read_json,write_json
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source
from scripts.diagnose_native_actions import PHASES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--plot',action='store_true')
    a=p.parse_args(); source=ROOT/'outputs/native_action_diagnosis_20261022'
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        summary=read_json(a.output/'summary.json'); rows=summary['rows']
        fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
        parents=list(dict.fromkeys(r['episode'] for r in rows))
        for i,parent in enumerate(parents):
            group=[next(r for r in rows if r['episode']==parent and r['phase']==phase) for phase in PHASES]
            x=np.arange(len(PHASES))+(i-2)*.07
            axes[0].scatter(x,[r['raw_errors']['5']['finger_rmse_mm'] for r in group],label=parent,s=35)
            axes[1].scatter(x,[r['command_pose_first5_rmse_mm'] for r in group],label=parent,s=35)
        axes[0].set(ylabel='Finger-command RMSE (mm)',title='Generated actions with correct recorded expert observations | first five commands')
        axes[1].set(ylabel='Command-pose discrepancy (mm)',xticks=np.arange(len(PHASES)),xticklabels=PHASES,
                    title='Static robot-grasp-site FK: generated vs expert joint targets (not simulated motion)')
        axes[0].legend(ncol=3,fontsize=8)
        for ax in axes: ax.grid(alpha=.2); ax.set_ylim(bottom=0)
        fig.text(.5,.015,'Five selected training resets; one median frame per phase; fixed noise. Diagnostic sample, not success/generalization.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(a.output/'action_errors.png',dpi=160)
        fig.savefig(a.output/'action_errors.pdf'); plt.close(fig)
        return
    if a.output.exists(): raise FileExistsError('New report output required')
    import mujoco
    from scripts.rollout_smolvla import verify_snapshot
    prior=read_json(source/'report.json'); plan=read_json(source/'plan.json')
    if not prior['completed'] or sha256(Path(prior['checkpoint'])/'model.safetensors')!=prior['weights_sha256']:
        raise ValueError('Frozen diagnostic/model binding changed')
    snapshot=ROOT/'third_party/capstone_automateData'
    collection=ROOT/'data/drive_v2c30_20261012/merged_all_v1'
    metadata=read_json(collection/next(iter(plan['parents']))/'metadata.json')
    workspace=verify_snapshot(snapshot,metadata)
    sys.path.insert(0,str(snapshot))
    from envs.openarm_insert import OpenArmInsertEnv
    env=OpenArmInsertEnv(images=False,workspace=workspace)
    rows=[]
    try:
        with np.load(source/'predictions.npz') as predictions:
            for parent in plan['parents']:
                file=collection/parent/'episode.npz'
                if sha256(file)!=plan['parents'][parent]['data_sha256']: raise ValueError('Raw source changed')
                with np.load(file) as raw: qpos=raw['qpos']
                for original in [r for r in prior['rows'] if r['episode']==parent]:
                    row=dict(original); key=parent+'_'+row['phase']; poses=[]
                    for commands in (predictions[key+'_raw'],predictions[key+'_target']):
                        points=[]
                        for command in commands[:5]:
                            env.data.qpos[:]=qpos[row['step']]
                            env.data.qpos[env.qa['right']]=command[:7]
                            env.data.qpos[env.fqa['right']]=command[7]
                            mujoco.mj_forward(env.model,env.data)
                            points.append(env.data.site_xpos[env.grasp_site].copy())
                        poses.append(np.asarray(points))
                    row['command_pose_first5_rmse_mm']=float(1000*np.sqrt(np.mean(np.sum((poses[0]-poses[1])**2,axis=1))))
                    rows.append(row)
    finally: env.close()
    close=[r for r in rows if r['phase']=='close']
    summary={**prior,'rows':rows,'diagnostic_report_sha256':sha256(source/'report.json'),
        'reporter':archive_source('scripts/report_native_action_diagnosis.py'),
        'kinematics_limits':'Static forward kinematics of commanded joint targets in an offline model copy. No reset/physics step, actuator/rate-limit/contact simulation or robot command. Not actual trajectory or a counterfactual task success.',
        'aggregate':{
            'close_finger_first5_rmse_mm':[r['raw_errors']['5']['finger_rmse_mm'] for r in close],
            'close_finger_first5_seed2_rmse_mm':[r['second_seed_errors']['5']['finger_rmse_mm'] for r in close],
            'close_command_pose_first5_rmse_mm':[r['command_pose_first5_rmse_mm'] for r in close],
            'raw_video_finger_first5_max_rmse_mm':max(r['raw_vs_video_action']['5']['finger_rmse_mm'] for r in rows),
            'mean_native_flow_loss_t05':float(np.mean([r['flow_losses'][1]['native_loss_32dim'] for r in rows])),
            'mean_active8_flow_mse_t05':float(np.mean([r['flow_losses'][1]['active8_velocity_mse'] for r in rows])),
            'mean_active8_flow_mse_t01':float(np.mean([r['flow_losses'][0]['active8_velocity_mse'] for r in rows]))}}
    stats=read_json(ROOT/'data/drive_v2c30_20261012/export_all_v1/train/meta/stats.json')
    state_min=float(stats['observation.state']['min'][15]); command_min=float(stats['action']['min'][7])
    envelope=[]
    for trial in read_json(ROOT/'outputs/native_pickup_eval_20261022/report.json')['rows']:
        if trial['mode']!='policy': continue
        folder=Path(trial['directory']); diagnostic=read_json(folder/'diagnostics.json')
        with np.load(folder/'trajectory.npz') as trace:
            measured=trace['observation_state'][:,15]; command=trace['action_requested'][:,7]
            state_cross=np.flatnonzero(measured<state_min-1e-6)
            command_cross=np.flatnonzero(command<command_min-1e-6)
            first=int(state_cross[0]) if len(state_cross) else None
            envelope.append({'episode':trial['episode'],'minimum_measured_finger_mm':float(measured.min()*1000),
                'minimum_command_mm':float(command.min()*1000),'first_state_below_train_min':first,
                'first_command_below_train_min':int(command_cross[0]) if len(command_cross) else None,
                'hand_to_grasp_point_error_at_first_state_cross_mm':float(diagnostic[first]['grasp_slip_m']*1000) if first is not None else None})
    summary['finger_training_envelope']={'measured_min_mm':state_min*1000,
        'measured_max_mm':float(stats['observation.state']['max'][15])*1000,
        'command_min_mm':command_min*1000,'command_max_mm':float(stats['action']['max'][7])*1000,
        'trials':envelope,
        'limits':'Empirical training split min/max, not physical limits. Crossing it identifies unseen scalar values, not a proof of sole causality or permission to clamp commands.'}
    a.output.mkdir(parents=True)
    write_json(a.output/'summary.json',summary)
    with (a.output/'metrics.csv').open('x',newline='',encoding='utf-8') as stream:
        fields=['episode','phase','step','arm_first5_rmse_rad','finger_first5_rmse_mm','finger_first5_bias_mm',
                'command_pose_first5_rmse_mm','video_effect_finger_first5_rmse_mm']
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for r in rows:
            e=r['raw_errors']['5']; writer.writerow({**{k:r[k] for k in ('episode','phase','step','command_pose_first5_rmse_mm')},
                'arm_first5_rmse_rad':e['arm_rmse_rad'],'finger_first5_rmse_mm':e['finger_rmse_mm'],
                'finger_first5_bias_mm':e['finger_bias_mm'],
                'video_effect_finger_first5_rmse_mm':r['raw_vs_video_action']['5']['finger_rmse_mm']})
    print(summary['aggregate'],flush=True)


if __name__=='__main__': main()
