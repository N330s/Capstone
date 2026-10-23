"""Pinned LeRobot training loop, with qualified OpenArm mask/precision adapters."""
from importlib.metadata import version
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT/'data/hf_cache'))
os.environ.setdefault('HF_DATASETS_CACHE', str(Path(os.environ['HF_HOME'])/'datasets'))
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['WANDB_MODE'] = 'disabled'


class NativeBudgetStop(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--native-max-seconds', type=float, required=True)
    parser.add_argument('--native-stop-after', type=int)
    parser.add_argument('--native-validation', action='store_true')
    parser.add_argument('--native-single-episode', help='Audit native dataset episode filtering and generated actions')
    parser.add_argument('--native-no-early-checkpoint', action='store_true')
    parser.add_argument('--native-export', type=Path, default=ROOT/'data/drive_v2c30_20261012/export_all_v1')
    parser.add_argument('--native-collection', type=Path, default=ROOT/'data/drive_v2c30_20261012/merged_all_v1')
    controls, remaining = parser.parse_known_args()
    if not 0 < controls.native_max_seconds <= 14400 or (controls.native_stop_after is not None and controls.native_stop_after < 1):
        raise ValueError('Bounded four-hour native run required')
    sys.argv = [sys.argv[0], *remaining]
    if version('lerobot') != '0.4.4':
        raise RuntimeError('Requalify adapters before a LeRobot version migration')
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from lerobot.scripts import lerobot_train as native
    from learning.vla.native_smolvla import install_native_adapters
    from learning.vla.runtime_io import write_status, write_json
    from learning.vla.smolvla_data import sha256
    from learning.vla.source_archive import archive_source
    from learning.vla.runtime_resources import resources as check_resources, existing_parent
    started = time.monotonic()
    output = next((Path(a.split('=', 1)[1]) for a in sys.argv[1:] if a.startswith('--output_dir=')), None)
    if output is None:
        raise ValueError('Explicit local --output_dir= required')
    def resources():
        return check_resources(output)

    def emit(event):
        import math
        event = {**event, 'elapsed_s': time.monotonic()-started}
        if event['type'] == 'update' and not all(math.isfinite(event[k]) for k in ('loss', 'grad_norm', 'lr')):
            raise FloatingPointError('Nonfinite native update')
        output.mkdir(parents=True, exist_ok=True)
        with (output/'native_events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, allow_nan=False)+'\n')
        write_status(output/'native_status.json', event)

    install_native_adapters(native, emit)
    monitor = None
    if controls.native_single_episode:
        from learning.vla.native_single_episode import SingleEpisodeMonitor
        monitor = SingleEpisodeMonitor(native, controls.native_single_episode, output, emit,
            export=controls.native_export, collection=controls.native_collection)
    state = {}
    original_processors = native.make_pre_post_processors
    original_optimizer = native.make_optimizer_and_scheduler
    original_save = native.save_checkpoint
    original_update = native.update_policy
    original_last = native.update_last_checkpoint

    def validation(policy, accelerator, step):
        import torch
        from torch.utils.data import default_collate
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        if 'validation_data' not in state:
            export = controls.native_export
            from learning.vla.runtime_io import read_json
            receipt = read_json(export/'openarm_export.json')
            ds = LeRobotDataset(receipt['datasets']['validation']['repo_id'], root=export/'validation',
                delta_timestamps={'action': [i/50 for i in range(50)]}, video_backend='pyav')
            windows, offset = [], 0
            for record in receipt['datasets']['validation']['episodes']:
                for fraction in (.02,.15,.30,.45,.60,.75,.90,.99):
                    index = min(int(record['steps']*fraction), record['steps']-1)
                    windows.append({'episode': record['episode'], 'frame': index, 'index': offset+index})
                offset += record['steps']
            state['validation_data'], state['validation_windows'] = ds, windows
            write_json(output/'native_validation_plan.json', {'windows': windows, 'noise_seed_base': 30000,
                'flow_time': .5, 'normalization_stats': 'training processors only',
                'train_stats_sha256': sha256(export/'train/meta/stats.json'),
                'limits': 'Eight fixed flow-loss windows per validation episode; not every validation frame or autonomous task success.'})
        ds, windows = state['validation_data'], state['validation_windows']
        was_training = policy.training
        policy.eval()
        total, count = 0., 0
        begun = time.monotonic()
        size = state['cfg'].batch_size
        try:
            with torch.random.fork_rng(devices=[accelerator.device.index or 0]), torch.no_grad(), accelerator.autocast():
                for start in range(0, len(windows), size):
                    group = windows[start:start+size]
                    batch = state['processors'][0](default_collate([ds[w['index']] for w in group]))
                    noise = torch.stack([torch.randn((50,32), device=accelerator.device,
                        generator=torch.Generator(device=accelerator.device).manual_seed(30000+i))
                        for i in range(start, start+len(group))])
                    loss, _ = policy.forward(batch, noise=noise,
                        time=torch.full((len(group),), .5, device=accelerator.device))
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite validation loss')
                    total += float(loss)*len(group); count += len(group)
        finally:
            policy.train(was_training)
        event = {'type': 'validation', 'step': step, 'loss': total/count,
            'windows': count, 'episodes': len({w['episode'] for w in windows}), 'seconds': time.monotonic()-begun,
            'scope': 'fixed-window flow loss, not full validation or rollout success'}
        emit(event)
        state['last_validation'] = event

    def processors(*args, **kwargs):
        pair = original_processors(*args, **kwargs)
        state['processors'] = pair
        return pair

    def optimizer_factory(cfg, policy):
        state['cfg'] = cfg
        emit({'type': 'effective_contract', 'state_shape': list(policy.config.input_features['observation.state'].shape),
              'action_shape': list(policy.config.output_features['action'].shape),
              'cameras': list(policy.config.image_features), 'batch_size': cfg.batch_size,
              'precision': 'FP32 trainable storage, Accelerator BF16 compute'})
        if monitor:
            monitor.bind(cfg, policy, state['processors'])
        return original_optimizer(cfg, policy)

    def last_checkpoint(checkpoint_dir):
        if os.name != 'nt': return original_last(checkpoint_dir)
        # No Windows symlink privilege is needed; resumption uses explicit paths.
        write_json(checkpoint_dir.parent/'last_checkpoint.json', {'checkpoint': checkpoint_dir.name})
        return checkpoint_dir.parent/'last_checkpoint.json'

    def save_checkpoint(*args, **kwargs):
        import shutil
        import torch
        from safetensors import safe_open
        from learning.vla.native_smolvla import assert_native_precision
        bound = inspect.signature(original_save).bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        policy, optimizer = values['policy'], values['optimizer']
        assert_native_precision(policy, optimizer)
        estimate = sum(p.numel()*p.element_size() for p in policy.state_dict().values())
        estimate += sum(v.numel()*v.element_size() for s in optimizer.state.values() for v in s.values() if isinstance(v, torch.Tensor))
        if shutil.disk_usage(existing_parent(output)).free < 25*2**30 + estimate + 256*2**20:
            raise RuntimeError('Native checkpoint would violate disk reserve')
        folder = values['checkpoint_dir']
        if folder.exists(): raise FileExistsError('Never overwrite an existing checkpoint')
        emit({'type': 'checkpoint_started', 'step': values['step'], 'path': str(folder), 'estimated_bytes': estimate})
        original_save(*args, **kwargs)
        if monitor:
            monitor.verify_saved_processors(folder, policy.config.pretrained_path)
        weights = folder/'pretrained_model/model.safetensors'
        with safe_open(str(weights), framework='pt') as stored:
            for name, parameter in policy.named_parameters():
                if parameter.requires_grad and stored.get_slice(name).get_dtype() != 'F32':
                    raise ValueError('Saved native trainable weights are not FP32')
        write_json(folder/'openarm_native_complete.json', {'complete': True, 'step': values['step'],
            'weights_sha256': sha256(weights), 'fp32_trainable_storage': True,
            'adapter': archive_source('learning/vla/native_smolvla.py'),
            'entrypoint': archive_source('scripts/train_native_smolvla.py'),
            'native_training_loop': True, 'optimizer_resume_format': 'lerobot-0.4.4',
            'validation': state.get('last_validation') if state.get('last_validation',{}).get('step')==values['step'] else None,
            'limits': 'Native RNG/optimizer/scheduler checkpoint; exact dataloader-cursor replay is not claimed.'})
        state['last_saved_step'] = values['step']
        emit({'type': 'checkpoint_complete', 'step': values['step'], 'path': str(folder)})

    def guarded_update(*args, **kwargs):
        result = original_update(*args, **kwargs)
        tracker = result[0]
        step = int(tracker.steps)+1
        if time.monotonic()-state.get('last_resource_check', 0) >= 10:
            emit({'type': 'resources', **resources()})
            state['last_resource_check'] = time.monotonic()
        stopped = ((controls.native_stop_after is not None and step >= controls.native_stop_after)
                   or time.monotonic()-started >= controls.native_max_seconds)
        if monitor and (step == 1 or step % 250 == 0 or stopped or step == state['cfg'].steps):
            policy = kwargs.get('policy', args[1] if len(args)>1 else None)
            monitor.probe(kwargs['accelerator'].unwrap_model(policy, keep_fp32_wrapper=True), step)
        if controls.native_validation and (step == 1 or step%1000 == 0 or stopped or step == state['cfg'].steps):
            policy = kwargs.get('policy', args[1] if len(args)>1 else None)
            validation(policy, kwargs['accelerator'], step)
        # One early recovery checkpoint, then the native configured frequency/final.
        if stopped or (step == 1000 and not controls.native_no_early_checkpoint):
            policy = kwargs.get('policy', args[1] if len(args)>1 else None)
            optimizer = kwargs.get('optimizer', args[3] if len(args)>3 else None)
            accelerator = kwargs['accelerator']
            cfg = state['cfg']
            folder = native.get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            if state.get('last_saved_step') != step:
                save_checkpoint(checkpoint_dir=folder, step=step, cfg=cfg,
                    policy=accelerator.unwrap_model(policy, keep_fp32_wrapper=True), optimizer=optimizer,
                    scheduler=kwargs.get('lr_scheduler'), preprocessor=state['processors'][0],
                    postprocessor=state['processors'][1])
                last_checkpoint(folder)
            if stopped:
                state['stop_reason'] = 'preflight_update_limit' if controls.native_stop_after else 'time_budget'
                raise NativeBudgetStop()
        return result

    native.make_pre_post_processors, native.make_optimizer_and_scheduler = processors, optimizer_factory
    native.save_checkpoint, native.update_last_checkpoint, native.update_policy = save_checkpoint, last_checkpoint, guarded_update
    accelerator = Accelerator(mixed_precision='bf16', step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    native.register_third_party_plugins()
    resources()
    try:
        native.train(accelerator=accelerator)
    except NativeBudgetStop:
        accelerator.end_training()
    import torch
    emit({'type': 'completed', 'reason': state.get('stop_reason', 'planned_steps'),
          'last_saved_step': state.get('last_saved_step'),
          'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20})


if __name__ == '__main__':
    main()
