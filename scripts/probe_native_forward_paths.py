"""Frozen offline cached-vs-joint Euler integration check; no model updates."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from learning.vla.runtime_io import read_json,write_json,write_status
from learning.vla.smolvla_data import sha256,observation
from learning.vla.source_archive import archive_source
from scripts.diagnose_cadence_repeatability import resources
from scripts.run_smolvla_job import arm_deadline,kill_owned_process
from scripts.diagnose_native_actions import errors


def worker(output):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from learning.vla.smolvla_policy import SmolVLAController
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    prior=read_json(ROOT/'outputs/native_action_diagnosis_20261022/report.json')
    candidate=Path(prior['checkpoint']); expected=prior['weights_sha256']
    if sha256(candidate/'model.safetensors')!=expected: raise ValueError('Changed model')
    actor=SmolVLAController(candidate,ROOT/'data/models/smolvlm_backbone',1000)
    rows,arrays=[],{}
    for parent in ('train_20002','train_20003'):
        folder=ROOT/'data/drive_v2c30_20261012/merged_all_v1'/parent
        metadata=read_json(folder/'metadata.json')
        with np.load(folder/'episode.npz') as raw:
            targets,states=raw['action_applied'],raw['observation_state']
            labels=raw['phase']; indices=np.flatnonzero(labels=='close'); close=int(indices[len(indices)//2])
            close_images={k:raw['image_'+k][close] for k in ('scene_rgb','wrist_rgb')}
        for label,step in (('initial',0),('close',close)):
            resources()
            if label=='initial':
                with np.load(ROOT/'outputs/native_pickup_eval_20261022'/('training_'+parent+'_policy')/'first_actor_input.npz') as saved:
                    obs={'state':saved['state'],'instruction':str(saved['instruction'].item()),
                         'images':{k:saved[k] for k in ('scene_rgb','wrist_rgb')}}
            else: obs={'state':states[step],'instruction':metadata['instruction'],'images':close_images}
            seed=1000+step
            actor.model.reset(); cached,_=actor.predict(obs,noise_seed=seed)
            batch=actor.pre(default_collate([observation(obs)]))
            # Joint training forward at exactly the chosen x_t. Supplying zero
            # normalized targets and noise=x_t/t reconstructs x_t; loss ignored.
            batch['action']=torch.zeros((1,50,8),device='cuda')
            x=torch.randn((1,50,32),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
            capture=[]
            handle=actor.model.model.action_out_proj.register_forward_hook(
                lambda module,args,result:capture.append(result))
            reconstruction=0.
            try:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for i in range(10):
                        t=torch.tensor([1-i/10],device='cuda',dtype=torch.float32)
                        noise=x/t[:,None,None]
                        reconstruction=max(reconstruction,float((t[:,None,None]*noise-x).abs().max()))
                        capture.clear(); actor.model.forward(batch,noise=noise,time=t)
                        x=x-.1*capture[-1]
                    joint=actor.post(x[:,:,:8])[0].float().cpu().numpy()
            finally: handle.remove()
            target=targets[step:step+50]
            row={'episode':parent,'phase':label,'step':step,'seed':seed,
                 'cached_vs_joint':errors(cached,joint),'cached_errors':errors(cached,target),
                 'joint_errors':errors(joint,target),'x_t_reconstruction_max_abs':reconstruction}
            if label=='initial':
                old=read_json(ROOT/'outputs/native_pickup_eval_20261022'/('training_'+parent+'_policy')/'chunks.json')[0]['prediction']
                row['fresh_vs_saved_initial_max_abs']=float(np.abs(cached-np.asarray(old)).max())
            rows.append(row); arrays[parent+'_'+label+'_cached']=cached; arrays[parent+'_'+label+'_joint']=joint
            write_json(output/'progress.json',{'rows':rows}); print(row,flush=True)
        del close_images,obs,states,targets
    if sha256(candidate/'model.safetensors')!=expected: raise ValueError('Changed model')
    np.savez_compressed(output/'predictions.npz',**arrays)
    write_json(output/'report.json',{'completed':True,'rows':rows,'weights_sha256':expected,
        'training':False,'physics_steps':0,
        'limits':'Four fixed observations, matched 10-step Euler and noise. Joint path reconstructs x_t within reported FP32 rounding; measures cached-vs-training numerical effect, not autonomous pickup or universal equivalence.'})


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true'); a=p.parse_args(); a.output=a.output.resolve()
    if a.worker: return worker(a.output)
    if a.output.parent!=ROOT/'outputs' or a.output.exists(): raise ValueError('New diagnostic directory required')
    resources(); a.output.mkdir(); begun=time.monotonic()
    write_json(a.output/'plan.json',{'source':archive_source('scripts/probe_native_forward_paths.py'),
        'training':False,'physics_steps':0,'max_seconds':240,'cases':['train_20002 initial/close','train_20003 initial/close']})
    with (a.output/'worker.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-u',__file__,'--output',str(a.output),'--worker'],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        fired=threading.Event(); timer=arm_deadline(child,240,fired)
        try:
            while child.poll() is None:
                write_status(a.output/'health.json',{'running':True,'elapsed_s':time.monotonic()-begun,**resources()})
                time.sleep(3)
            if child.returncode or fired.is_set(): raise RuntimeError('Forward-path diagnostic incomplete')
            write_status(a.output/'health.json',{'running':False,'completed':True,'elapsed_s':time.monotonic()-begun})
        except BaseException as error:
            write_status(a.output/'health.json',{'running':False,'completed':False,'error':repr(error)}); raise
        finally:
            if child.poll() is None: kill_owned_process(child)
            child.wait(); timer.cancel()


if __name__=='__main__': main()
