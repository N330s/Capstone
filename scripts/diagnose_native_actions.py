"""Bounded frozen-checkpoint action-generation/data-path diagnosis, not training."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.runtime_io import read_json, write_json, write_status
from learning.vla.smolvla_data import sha256, observation, CAMERAS
from learning.vla.source_archive import archive_source
from learning.vla.action_metrics import errors
from scripts.run_smolvla_job import arm_deadline, kill_owned_process
from scripts.diagnose_cadence_repeatability import resources

PHASES = ('survey', 'transit_pick', 'descend', 'close', 'lift', 'insert')


def worker(output):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    from types import MethodType
    from collections import Counter
    from safetensors import safe_open
    from torch.utils.data import default_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from learning.vla.smolvla_policy import SmolVLAController
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    begun = time.monotonic()
    plan = read_json(output/'plan.json')
    candidate = Path(plan['checkpoint']); export = ROOT/'data/drive_v2c30_20261012/export_all_v1'
    collection = ROOT/'data/drive_v2c30_20261012/merged_all_v1'
    receipt = read_json(export/'openarm_export.json')
    records = receipt['datasets']['train']['episodes']
    ds = LeRobotDataset(receipt['datasets']['train']['repo_id'], root=export/'train',
        delta_timestamps={'action': [i/50 for i in range(50)]}, video_backend='pyav')
    offsets, offset, counts = {}, 0, Counter()
    for record in records:
        offsets[record['episode']] = offset; offset += record['steps']
        with np.load(collection/record['episode']/'episode.npz', allow_pickle=False) as z:
            labels, number = np.unique(z['phase'], return_counts=True)
            counts.update(dict(zip(labels.tolist(), number.tolist())))
    actor = SmolVLAController(candidate, ROOT/'data/models/smolvlm_backbone', 1000)
    loaded = actor.model.state_dict(); checked = 0
    with safe_open(str(candidate/'model.safetensors'), framework='pt') as saved:
        for name in saved.keys():
            value, actual = saved.get_tensor(name), loaded[name].detach().cpu()
            if value.dtype != actual.dtype or not torch.equal(value, actual):
                raise AssertionError('Controller reload changed tensor: '+name)
            checked += 1
    del loaded, value, actual
    rows, arrays = [], {}
    for parent in plan['parents']:
        folder = collection/parent; meta = read_json(folder/'metadata.json')
        if sha256(folder/'episode.npz') != plan['parents'][parent]['data_sha256']:
            raise ValueError('Raw parent changed')
        with np.load(folder/'episode.npz', allow_pickle=False) as z:
            states, targets, phase = z['observation_state'], z['action_applied'], z['phase']
            images = {key: z['image_'+camera] for key, camera in CAMERAS.items()}
        for label in PHASES:
            if time.monotonic()-begun > 550: raise TimeoutError('Offline diagnostic budget')
            resources()
            indices = np.flatnonzero(phase == label)
            if not len(indices): raise ValueError('Missing planned phase: '+label)
            step = int(indices[len(indices)//2])
            end = np.minimum(np.arange(step, step+50), len(targets)-1)
            target = targets[end].astype(np.float32)
            obs = {'state': states[step], 'instruction': meta['instruction'],
                   'images': {camera: images[key][step] for key, camera in CAMERAS.items()}}
            sample = ds[offsets[parent]+step]
            np.testing.assert_allclose(sample['observation.state'], states[step], rtol=0, atol=1e-6)
            np.testing.assert_allclose(sample['action'], target, rtol=0, atol=1e-6)
            np.testing.assert_array_equal(sample['action_is_pad'], np.arange(step,step+50)>=len(targets))
            if sample['task'] != meta['instruction']: raise ValueError('Task differs')
            raw_batch = {**observation(obs), 'action': target,
                         'action_is_pad': np.arange(step,step+50)>=len(targets)}
            batch = actor.pre(default_collate([raw_batch]))
            for key in ('observation.state', 'action'):
                stats = ds.meta.stats[key]
                expected = (np.asarray(raw_batch[key])-np.asarray(stats['mean']))/(np.asarray(stats['std'])+1e-8)
                np.testing.assert_allclose(batch[key][0].cpu(), expected, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(actor.post(batch['action'])[0].cpu(), target, atol=1e-5, rtol=0)
            decoded = {'state': sample['observation.state'].numpy(), 'instruction': sample['task'],
                       'images': {camera: np.rint(sample[key].numpy().transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
                                  for key,camera in CAMERAS.items()}}
            raw_vs_video = {camera: float(np.abs(obs['images'][camera].astype(float)-decoded['images'][camera]).mean())
                            for camera in obs['images']}
            seed = 1000+step
            inner = actor.model.model; original = inner.denoise_step; cache_capture = []
            def capture(self, *a, **kw):
                value = original(*a, **kw)
                if not cache_capture: cache_capture.append(value.detach().float().cpu().numpy().copy())
                return value
            inner.denoise_step = MethodType(capture, inner)
            try:
                actor.model.reset(); prediction, latency = actor.predict(obs, noise_seed=seed)
            finally: inner.denoise_step = original
            actor.model.reset(); video_prediction, _ = actor.predict(decoded, noise_seed=seed)
            actor.model.reset(); second_seed, _ = actor.predict(obs, noise_seed=2000+step)
            noise = torch.randn((1,50,32), device='cuda',
                generator=torch.Generator(device='cuda').manual_seed(seed))
            padded_target = actor.model.prepare_action(batch)
            flow, joint_v = [], None
            for flow_time in (.1, .5, .9, 1.):
                outputs = []
                handle = inner.action_out_proj.register_forward_hook(
                    lambda module, arguments, result: outputs.append(result.detach().float().cpu().numpy().copy()))
                try:
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        loss, _ = actor.model.forward({**batch, 'actions_id_pad': batch['action_is_pad']}, noise=noise,
                            time=torch.tensor([flow_time], device='cuda'))
                finally: handle.remove()
                v = outputs[-1]
                error = v-(noise-padded_target).float().cpu().numpy()
                flow.append({'time': flow_time, 'native_loss_32dim': float(loss),
                             'active8_velocity_mse': float(np.mean(error[:,:,:8]**2)),
                             'padded24_velocity_mse': float(np.mean(error[:,:,8:]**2))})
                if flow_time == 1.: joint_v = v
            row = {'episode': parent, 'phase': label, 'step': step, 'noise_seed': seed,
                'raw_errors': errors(prediction,target), 'video_errors': errors(video_prediction,target),
                'second_seed_errors': errors(second_seed,target), 'raw_vs_video_action': errors(prediction,video_prediction),
                'raw_vs_video_rgb_mae_uint8': raw_vs_video, 'flow_losses': flow,
                'train_vs_cached_velocity_max_abs': float(np.abs(joint_v-cache_capture[0]).max()),
                'train_vs_cached_velocity_active8_rmse': float(np.sqrt(np.mean((joint_v[:,:,:8]-cache_capture[0][:,:,:8])**2))),
                'measured_finger_mm': float(states[step,15]*1000),
                'target_first5_finger_mm': float(target[:5,7].mean()*1000),
                'predicted_first5_finger_mm': float(prediction[:5,7].mean()*1000),
                'latency_s': latency}
            # Integration-resolution diagnostic only; no deployment/config file changes.
            if parent == next(iter(plan['parents'])) and label in ('survey','descend','close','lift'):
                inner.config.num_steps = 50
                try:
                    actor.model.reset(); dense, _ = actor.predict(obs, noise_seed=seed)
                finally: inner.config.num_steps = 10
                row['fifty_denoising_steps_errors'] = errors(dense,target)
                arrays[f'{parent}_{label}_steps50'] = dense
            key = f'{parent}_{label}'
            arrays.update({key+'_raw': prediction, key+'_video': video_prediction,
                           key+'_seed2': second_seed, key+'_target': target,
                           key+'_train_v1': joint_v, key+'_cached_v1': cache_capture[0]})
            rows.append(row); write_json(output/'progress.json', {'rows':rows})
            print({'episode':parent,'phase':label,'first5':row['raw_errors']['5'],
                   'cached_velocity_max_abs':row['train_vs_cached_velocity_max_abs']},flush=True)
        del images, states, targets, phase
    if sha256(candidate/'model.safetensors') != plan['weights_sha256']: raise ValueError('Model changed')
    np.savez_compressed(output/'predictions.npz', **arrays)
    write_json(output/'report.json', {'completed': True, 'rows':rows, 'checkpoint':str(candidate),
        'weights_sha256':plan['weights_sha256'], 'all_saved_tensors_reload_exact':True, 'checked_tensors':checked,
        'train_phase_frames':dict(counts), 'training_sample_equivalent_passes':12243*8/sum(counts.values()),
        'actor_binding':actor.binding, 'elapsed_s':time.monotonic()-begun,
        'training':False,'physics_steps':0,
        'limits':'Phase-selected observations from the five failed training resets, two fixed noise seeds; offline assisted observations, not autonomous pickup or validation success. No optimizer updates or deployment change.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--worker', action='store_true')
    args = p.parse_args(); args.output = args.output.resolve()
    if args.worker: return worker(args.output)
    if args.output.parent != ROOT/'outputs' or args.output.exists(): raise ValueError('New diagnostic output required')
    resources(); args.output.mkdir()
    prior = read_json(ROOT/'outputs/native_pickup_eval_20261022/plan.json')
    plan = {'checkpoint':prior['checkpoint'],'weights_sha256':prior['weights_sha256'],
            'parents':{name:value for name,value in prior['parents'].items() if value['group']=='training'},
            'phases':PHASES,'total_limit_s':600,'training':False,'physics_steps':0,
            'source':archive_source('scripts/diagnose_native_actions.py'),
            'source_bindings':prior['sources']}
    if sha256(Path(plan['checkpoint'])/'model.safetensors')!=plan['weights_sha256']: raise ValueError('Model changed')
    write_json(args.output/'plan.json', plan)
    flags = subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    started = time.monotonic()
    with (args.output/'worker.log').open('x') as log:
        child = subprocess.Popen([sys.executable,'-u',__file__,'--output',str(args.output),'--worker'],
                                 stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,creationflags=flags)
        fired = threading.Event(); timer = arm_deadline(child,600,fired)
        try:
            while child.poll() is None:
                for name,binding in plan['source_bindings'].items():
                    if sha256(ROOT/name)!=binding['sha256']: raise RuntimeError('Source changed: '+name)
                write_status(args.output/'health.json',{'running':True,'elapsed_s':time.monotonic()-started,**resources()})
                time.sleep(3)
            if child.returncode or fired.is_set(): raise RuntimeError('Diagnostic incomplete; inspect worker.log')
            write_status(args.output/'health.json',{'running':False,'completed':True,'elapsed_s':time.monotonic()-started})
        except BaseException as error:
            write_status(args.output/'health.json',{'running':False,'completed':False,'error':repr(error)})
            raise
        finally:
            if child.poll() is None: kill_owned_process(child)
            child.wait(); timer.cancel()


if __name__ == '__main__': main()
