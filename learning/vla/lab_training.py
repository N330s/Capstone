"""Portable command planning for the pinned native SmolVLA baseline.

No model, optimizer or dataset sampler is implemented here. Actual training is
delegated to scripts/train_native_smolvla.py and public LeRobot 0.4.4.
"""
from pathlib import Path
import math
import sys

from learning.vla.runtime_io import read_json
from learning.vla.smolvla_data import CONTRACT, sha256

ROOT = Path(__file__).resolve().parents[2]
PATH_KEYS = ('export', 'collection', 'base', 'checkpoint', 'backbone', 'snapshot', 'runs')


def load_config(path):
    cfg = read_json(path)
    for key in PATH_KEYS:
        value = Path(cfg[key])
        if not value.is_absolute():
            raise ValueError('Absolute runtime path required: '+key)
        cfg[key] = value.resolve()
    for key in ('batch_size', 'additional_steps', 'wall_seconds', 'save_freq', 'seed', 'num_workers'):
        if type(cfg[key]) is not int or cfg[key] < (0 if key in ('seed', 'num_workers') else 1):
            raise ValueError('Invalid integer: '+key)
    if not 180 <= cfg['wall_seconds'] <= 14400:
        raise ValueError('Wall budget must be 180..14400 seconds; longer campaigns need separate authorization')
    return cfg


def verify_inputs(cfg):
    bindings = {
        cfg['export']/'openarm_export.json': cfg['export_sha256'],
        cfg['export']/'train/meta/stats.json': cfg['stats_sha256'],
        cfg['base']/'model.safetensors': cfg['base_sha256'],
        cfg['checkpoint']/'model.safetensors': cfg['checkpoint_sha256'],
    }
    for path, expected in bindings.items():
        if sha256(path) != expected:
            raise ValueError('Input hash mismatch: '+str(path))
    receipt = read_json(cfg['export']/'openarm_export.json')
    if receipt['contract'] != CONTRACT or receipt['lerobot_version'] != '0.4.4' or receipt['fps'] != 50:
        raise ValueError('Wrong data contract/version/rate')
    if sha256(cfg['collection']/'manifest.json') != receipt['source_manifest_sha256']:
        raise ValueError('Raw manifest differs from export')
    train = receipt['datasets']['train']['episodes']
    val = receipt['datasets']['validation']['episodes']
    names = [r['episode'] for r in train+val]
    if len(train) != 81 or len(val) != 20 or len(set(names)) != len(names):
        raise ValueError('Expected frozen, disjoint 81/20 episode split')
    frames = sum(r['steps'] for r in train)
    if frames != 147499:
        raise ValueError('Unexpected training frame count')
    from scripts.rollout_smolvla import verify_snapshot
    verify_snapshot(cfg['snapshot'], read_json(cfg['collection']/'train_20002/metadata.json'))
    for parent in ('base', 'checkpoint'):
        if not (cfg[parent]/'config.json').is_file():
            raise ValueError('Missing model configuration: '+parent)
    if not (cfg['backbone']/'tokenizer.json').is_file():
        raise ValueError('Stage the local tokenizer and backbone configuration')
    if not cfg['runs'].is_dir():
        raise ValueError('Create the approved persistent runs mount first')
    return {'passed': True, 'training_frames': frames, 'training_episodes': len(train),
        'validation_episodes': len(val), 'bindings': {str(k): v for k,v in bindings.items()},
        'scope': 'Pinned release/manifest/model/source bindings; run full export readback separately.'}


def run_folder(cfg, name):
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Run name must contain only letters, numbers, underscore or hyphen')
    result = cfg['runs']/name
    if result.exists():
        raise FileExistsError('Use a fresh run name: '+str(result))
    for key in PATH_KEYS[:-1]:
        if result.is_relative_to(cfg[key]) or cfg[key].is_relative_to(result):
            raise ValueError('Output must not overlap input data/models/assets')
    return result


def train_command(cfg, output, *, steps=None, max_seconds=None, smoke=False):
    receipt = read_json(cfg['export']/'openarm_export.json')
    steps = steps if steps is not None else cfg['additional_steps']
    max_seconds = max_seconds if max_seconds is not None else cfg['wall_seconds']-120
    command = [sys.executable, '-u', str(ROOT/'scripts/train_native_smolvla.py'),
        '--native-max-seconds', str(max_seconds), '--native-no-early-checkpoint',
        '--native-export', str(cfg['export']), '--native-collection', str(cfg['collection']),
        f'--policy.path={cfg["checkpoint"]}', '--policy.input_features={}',
        f'--policy.vlm_model_name={cfg["backbone"]}', '--policy.load_vlm_weights=false',
        '--policy.device=cuda', '--policy.push_to_hub=false', '--policy.freeze_vision_encoder=true',
        '--policy.train_expert_only=true', '--policy.train_state_proj=true',
        '--policy.adapt_to_pi_aloha=false', '--policy.use_delta_joint_actions_aloha=false', '--policy.empty_cameras=0',
        f'--dataset.repo_id={receipt["datasets"]["train"]["repo_id"]}', f'--dataset.root={cfg["export"]/"train"}',
        '--dataset.video_backend=pyav', '--dataset.image_transforms.enable=false',
        f'--output_dir={output}', f'--steps={steps}', f'--batch_size={cfg["batch_size"]}',
        f'--num_workers={cfg["num_workers"]}', f'--seed={cfg["seed"]}',
        '--eval_freq=0', '--log_freq=20', f'--save_freq={steps if smoke else cfg["save_freq"]}', '--wandb.enable=false']
    if not smoke:
        command.append('--native-validation')
    return command


def qualification_command(cfg, output, *, checkpoint=None):
    command = [sys.executable, '-u', str(ROOT/'scripts/qualify_native_smolvla.py'),
        '--mode', 'reload' if checkpoint else 'parity', '--output', str(output),
        '--export', str(cfg['export']), '--collection', str(cfg['collection']),
        '--base', str(cfg['base']), '--backbone', str(cfg['backbone']), '--batch-size', '1']
    if checkpoint:
        command += ['--checkpoint', str(checkpoint)]
    return command


def throughput(events, batch_size):
    updates = [r for r in events if r['type'] == 'update']
    if len(updates) < 10:
        raise ValueError('Need at least ten updates for a warmed-up throughput estimate')
    first, last = updates[7], updates[-1]
    elapsed = last['elapsed_s']-first['elapsed_s']
    if elapsed <= 0:
        raise ValueError('Nonpositive timing interval')
    rate = (last['step']-first['step'])*batch_size/elapsed
    if not math.isfinite(rate):
        raise ValueError('Invalid throughput')
    return {'windows_per_second': rate, 'warm_updates': last['step']-first['step'],
        'scope': 'Real loader, preprocessing and optimizer, discarding first eight updates; excludes final save/reload.'}
