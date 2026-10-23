"""Small real-model native loss/gradient/update parity and raw/export timing audit."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MethodType
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT/'data/hf_cache'))
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from learning.vla.runtime_io import read_json, write_json
from learning.vla.smolvla_data import sha256, CAMERAS


def tensor_digest(items):
    import torch
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(name.encode())
        if value is None:
            digest.update(b'NONE')
        else:
            value = value.detach().cpu().contiguous()
            digest.update(str((tuple(value.shape), str(value.dtype))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('parity', 'benchmark', 'reload'), default='parity')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--export', type=Path, default=ROOT/'data/drive_v2c30_20261012/export_all_v1')
    p.add_argument('--collection', type=Path, default=ROOT/'data/drive_v2c30_20261012/merged_all_v1')
    p.add_argument('--base', type=Path, default=ROOT/'data/models/smolvla_base')
    p.add_argument('--backbone', type=Path, default=ROOT/'data/models/smolvlm_backbone')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from accelerate import Accelerator
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.scripts import lerobot_train as native
    from lerobot.utils.logging_utils import MetricsTracker, AverageMeter
    from learning.vla.native_smolvla import install_native_adapters, local_processors, assert_native_precision
    from learning.vla.training_precision import parameter_dtypes
    from learning.vla.source_archive import archive_source
    torch.manual_seed(1000)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    export, raw_root = args.export, args.collection
    base = args.checkpoint if args.mode == 'reload' else args.base
    if base is None: raise ValueError('Reload qualification requires a checkpoint')
    receipt = read_json(export/'openarm_export.json')
    if sha256(export/'openarm_export.json') != 'cd3f7b7b02947f85dfbef4b46d218a932d753aa65bc1aa28412d3d5b819db63f':
        raise ValueError('Unexpected export release')
    if sha256(raw_root/'manifest.json') != receipt['source_manifest_sha256']:
        raise ValueError('Raw source manifest differs')
    ds = LeRobotDataset(receipt['datasets']['train']['repo_id'], root=export/'train',
        delta_timestamps={'action': [i/50 for i in range(50)]}, video_backend='pyav')
    cfg = PreTrainedConfig.from_pretrained(str(base))
    cfg.input_features, cfg.pretrained_path = {}, base
    cfg.vlm_model_name = str(args.backbone)
    cfg.device, cfg.push_to_hub, cfg.load_vlm_weights = 'cuda', False, False
    originals = install_native_adapters(native)
    model = native.make_policy(cfg=cfg, ds_meta=ds.meta)
    processor_kwargs = {} if args.mode == 'reload' else dict(
        preprocessor_overrides={'device_processor': {'device': 'cuda'},
            'normalizer_processor': {'stats': ds.meta.stats,
                'features': {**model.config.input_features, **model.config.output_features},
                'norm_map': model.config.normalization_mapping},
            'rename_observations_processor': {'rename_map': {}}},
        postprocessor_overrides={'unnormalizer_processor': {'stats': ds.meta.stats,
            'features': model.config.output_features, 'norm_map': model.config.normalization_mapping}})
    pre, post = local_processors(make_pre_post_processors, model.config, pretrained_path=str(base), **processor_kwargs)
    records = receipt['datasets']['train']['episodes']
    offsets, offset = {}, 0
    for record in records:
        offsets[record['episode']] = offset
        offset += record['steps']
    if len(records) != 81 or len(receipt['datasets']['validation']['episodes']) != 20:
        raise ValueError('Unexpected train/validation split')
    audit, samples = [], []
    if args.mode == 'parity':
        for parent in ('train_20003', 'train_22045', 'train_20002'):
            meta = read_json(raw_root/parent/'metadata.json')
            if sha256(raw_root/parent/'episode.npz') != meta['data_sha256']:
                raise ValueError('Raw episode changed')
            with np.load(raw_root/parent/'episode.npz', allow_pickle=False) as raw:
                states, actions = raw['observation_state'], raw['action_applied']
                images = {key: raw['image_'+camera] for key, camera in CAMERAS.items()}
                for step in (0, len(actions)//2, len(actions)-1):
                    sample = ds[offsets[parent]+step]
                    indices = np.minimum(np.arange(step, step+50), len(actions)-1)
                    np.testing.assert_allclose(sample['observation.state'], states[step], atol=1e-6, rtol=0)
                    np.testing.assert_allclose(sample['action'], actions[indices], atol=1e-6, rtol=0)
                    np.testing.assert_array_equal(sample['action_is_pad'], np.arange(step, step+50) >= len(actions))
                    if sample['task'] != meta['instruction']: raise ValueError('Instruction differs')
                    errors = {key: float(np.abs(sample[key].numpy()-images[key][step].transpose(2,0,1)/255).mean()*255)
                              for key in CAMERAS}
                    audit.append({'episode': parent, 'step': step, 'mask_true': int(sample['action_is_pad'].sum()),
                                  'rgb_mae_255': errors})
                    if parent == 'train_20003' and step in (0, len(actions)-1): samples.append(sample)
    else:
        generator = np.random.default_rng(1000)
        samples = [ds[int(i)] for i in generator.integers(0, len(ds), size=args.batch_size)]
    batch = pre(default_collate(samples))
    # Verify native train-only normalization, then physical postprocessing roundtrip.
    for key in ('observation.state', 'action'):
        stats = ds.meta.stats[key]
        raw_value = default_collate(samples)[key].numpy()
        expected = (raw_value-np.asarray(stats['mean']))/(np.asarray(stats['std'])+1e-8)
        np.testing.assert_allclose(batch[key].cpu().numpy(), expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(post(batch['action']).cpu().numpy(), default_collate(samples)['action'].numpy(), atol=1e-5, rtol=0)
    accelerator = Accelerator(mixed_precision='bf16', step_scheduler_with_optimizer=False)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    def optimizer_for():
        optimizer = cfg.get_optimizer_preset().build(model.get_optim_params())
        scheduler = cfg.get_scheduler_preset().build(optimizer, 20000)
        return optimizer, scheduler
    def metrics():
        return MetricsTracker(len(samples), len(ds), 81,
            {k: AverageMeter(k, ':.8f') for k in ('loss', 'grad_norm', 'lr', 'update_s')})
    if args.mode == 'reload':
        from safetensors import safe_open
        loaded = model.state_dict()
        checked = 0
        with safe_open(str(base/'model.safetensors'), framework='pt') as tensors:
            for name in tensors.keys():
                value = tensors.get_tensor(name)
                actual = loaded[name].detach().cpu()
                if value.dtype != actual.dtype or not torch.equal(value, actual):
                    raise AssertionError('Reload changed saved tensor: '+name)
                checked += 1
        result = {'all_saved_tensors_reload_exact': True, 'checked_tensors': checked,
                  'saved_processors_train_stats_verified': True, 'trainable_dtypes': parameter_dtypes(model)}
    elif args.mode == 'parity':
        initial = {n: p.detach().cpu().clone() for n, p in trainable.items()}
        frozen_before = tensor_digest((n, p) for n, p in model.named_parameters() if not p.requires_grad)
        noise = torch.randn((len(samples),50,32), device='cuda', generator=torch.Generator(device='cuda').manual_seed(2000))
        flow_time = torch.tensor([.25,.75], device='cuda')
        optimizer, scheduler = optimizer_for()
        reference = []
        for _ in range(2):
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = SmolVLAPolicy.forward(model, {**batch, 'actions_id_pad': batch['action_is_pad']}, noise=noise, time=flow_time)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            gradients = tensor_digest((n,p.grad) for n,p in trainable.items())
            optimizer.step(); scheduler.step(); assert_native_precision(model, optimizer)
            reference.append({'loss': float(loss), 'grad_norm': float(norm), 'gradients': gradients,
                              'weights': tensor_digest(trainable.items()), 'lr': scheduler.get_last_lr()})
        del optimizer, scheduler
        model.zero_grad(set_to_none=True)
        for n,p in trainable.items(): p.data.copy_(initial[n].to(p.device))
        del initial; gc.collect(); torch.cuda.empty_cache()
        adapted_forward = model.forward
        def fixed_forward(self, inputs, *a, **kw): return adapted_forward(inputs, noise=noise, time=flow_time)
        model.forward = MethodType(fixed_forward, model)
        optimizer, scheduler = optimizer_for()
        captured_gradients = []
        optimizer.register_step_pre_hook(lambda *_: captured_gradients.append(tensor_digest((n,p.grad) for n,p in trainable.items())))
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        tracker = metrics()
        actual = []
        for index in range(2):
            tracker, _ = native.update_policy(tracker, model, batch, optimizer, 10, accelerator, lr_scheduler=scheduler)
            row = {'loss': tracker.loss.val, 'grad_norm': tracker.grad_norm.val,
                   'gradients': captured_gradients[-1], 'weights': tensor_digest(trainable.items()),
                   'lr': scheduler.get_last_lr()}
            if row != reference[index]: raise AssertionError(f'Native/reference loss-gradient-update mismatch at {index}: {row} vs {reference[index]}')
            actual.append(row); tracker.step()
        if tensor_digest((n,p) for n,p in model.named_parameters() if not p.requires_grad) != frozen_before:
            raise AssertionError('Frozen backbone changed')
        result = {'native_reference_exact': True, 'compared_updates': 2, 'updates': actual,
                  'raw_export_samples': audit, 'trainable_dtypes': parameter_dtypes(model)}
    else:
        optimizer, scheduler = optimizer_for()
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        tracker, events = metrics(), []
        torch.cuda.reset_peak_memory_stats()
        for index in range(12):
            before = time.perf_counter()
            tracker, _ = native.update_policy(tracker, model, batch, optimizer, 10, accelerator, lr_scheduler=scheduler)
            torch.cuda.synchronize()
            events.append({'step': index+1, 'loss': tracker.loss.val, 'grad_norm': tracker.grad_norm.val,
                           'seconds': time.perf_counter()-before})
            tracker.step()
        result = {'batch_size': args.batch_size, 'events': events,
            'warm_update_seconds': float(np.mean([r['seconds'] for r in events[3:]])),
            'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
            'limits': 'Twelve real native updates on one fixed randomly selected batch; data-loader throughput measured in subsequent native preflight.'}
    result.update(completed=True, mode=args.mode, elapsed_s=time.monotonic()-started,
        checkpoint=str(base), base_sha256=sha256(base/'model.safetensors'),
        export_receipt_sha256=sha256(export/'openarm_export.json'), gpu=torch.cuda.get_device_name(),
        source=archive_source('scripts/qualify_native_smolvla.py'),
        adapter=archive_source('learning/vla/native_smolvla.py'))
    write_json(args.output/'report.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
