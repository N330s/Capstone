"""Causal RGB A/B and native LeRobot queue parity; offline, immutable actor."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.runtime_io import read_json, write_json
from learning.vla.smolvla_data import observation, sha256
from learning.vla.source_archive import archive_source
from scripts.diagnose_cadence_repeatability import resources
from scripts.report_cadence_repeatability import difference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    if not read_json(args.probe/'health.json').get('completed'):
        raise ValueError('Initial repeatability probe must finish first')
    resources()
    if not args.worker:
        import psutil
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output/'request.json', {'worker_limit_s': 180,
            'source': archive_source('scripts/probe_cadence_fixed_inputs.py'),
            'training': False, 'physics_steps': 0})
        with (args.output/'worker.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, __file__, '--probe', str(args.probe),
                '--output', str(args.output), '--worker'], stdout=log, stderr=subprocess.STDOUT)
            begun = time.monotonic()
            try:
                while proc.poll() is None:
                    if time.monotonic()-begun > 180:
                        raise TimeoutError('Fixed-input audit deadline')
                    write_json(args.output/'health.json', {'running': True,
                        'elapsed_s': time.monotonic()-begun, **resources()})
                    time.sleep(2)
                if proc.returncode:
                    raise RuntimeError('Fixed-input worker failed; see worker.log')
                write_json(args.output/'health.json', {'running': False, 'completed': True,
                    'elapsed_s': time.monotonic()-begun})
            except BaseException as error:
                write_json(args.output/'health.json', {'running': False, 'completed': False,
                    'error': repr(error), 'elapsed_s': time.monotonic()-begun})
                raise
            finally:
                if proc.poll() is None:
                    process = psutil.Process(proc.pid)
                    for child in process.children(recursive=True):
                        try: child.kill()
                        except psutil.NoSuchProcess: pass
                    try: process.kill()
                    except psutil.NoSuchProcess: pass
                    proc.wait(timeout=10)
        return
    # Process-local deterministic numerics; tests the recorded RGB difference, not physics.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from learning.vla.smolvla_policy import SmolVLAController
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    base = ROOT/'outputs/v2c30_sft_overnight_v1/checkpoints/step_0009426_best/pretrained_model'
    overlay = ROOT/'outputs/five_episode_openhold_20261020/training/expert'
    plan = read_json(ROOT/'outputs/inference_cadence_20261020_resume_v1/plan.json')
    if sha256(overlay/'expert.safetensors') != plan['candidate_sha256']:
        raise ValueError('Candidate changed')
    actor = SmolVLAController(base, ROOT/'data/models/smolvlm_backbone', 1000, expert_overlay=overlay)
    rows, predictions = [], {}
    for parent in plan['parents']:
        if time.monotonic()-started > 160:
            raise TimeoutError('Offline fixed-input audit deadline')
        resources()
        metadata = read_json(ROOT/'data/drive_v2c30_20261012/merged_all_v1'/parent/'metadata.json')
        observations, fixtures = {}, {}
        for label, index in (('A', 0), ('B', 1)):
            fixture = args.probe/f'legacy_{index}'/(parent+'_reset.npz')
            fixtures[label] = {'path': str(fixture), 'sha256': sha256(fixture)}
            with np.load(fixture, allow_pickle=False) as raw:
                observations[label] = {'state': raw['state'].copy(), 'instruction': metadata['instruction'],
                    'images': {k: raw[k].copy() for k in ('scene_rgb', 'wrist_rgb')}}
        state_equal = np.array_equal(observations['A']['state'], observations['B']['state'])
        if not state_equal:
            raise ValueError('This A/B fixture does not isolate RGB')
        image_differences = {}
        for camera in ('scene_rgb', 'wrist_rgb'):
            a, b = (observations[k]['images'][camera] for k in ('A', 'B'))
            image_differences[camera] = {'changed_channel_values': int(np.count_nonzero(a != b)),
                'total_channel_values': int(a.size),
                'max_abs_uint8': int(np.abs(a.astype(int)-b.astype(int)).max())}
        image_a, image_b = (observations[k]['images']['scene_rgb'] for k in ('A', 'B'))
        outputs = {}
        for label, key in (('A', 'A0'), ('B', 'B0'), ('A', 'A1'), ('B', 'B1')):
            actor.model.reset()
            value, _ = actor.predict(observations[label], noise_seed=1000)
            outputs[key] = value
            predictions[parent+'_'+key] = value
        references = {}
        for label, index in (('A', 0), ('B', 1)):
            with np.load(args.probe/f'legacy_{index}'/'predictions.npz') as previous:
                references[label] = difference(outputs[label+'0'], previous[parent+'_0'])
        queue_rows = []
        for prefix in (1, 5, 25):
            actor.model.config.n_action_steps = prefix
            actor.model.reset()
            noise = torch.randn((1, 50, actor.model.config.max_action_dim), device='cuda',
                                generator=torch.Generator(device='cuda').manual_seed(1000))
            native = []
            for _ in range(prefix):
                batch = actor.pre(default_collate([observation(observations['A'])]))
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    command = actor.post(actor.model.select_action(batch, noise=noise))
                native.append(command[0].float().cpu().numpy())
            value = np.asarray(native)
            predictions[f'{parent}_native_prefix{prefix}'] = value
            queue_rows.append({'prefix': prefix,
                'max_abs_difference': float(np.abs(value-outputs['A0'][:prefix]).max()),
                'exact_equal': bool(np.array_equal(value, outputs['A0'][:prefix]))})
        actor.model.config.n_action_steps = 50
        rows.append({'episode': parent, 'fixtures': fixtures,
            'only_rgb_differs': True, 'image_differences': image_differences,
            'changed_scene_channel_values': int(np.count_nonzero(image_a != image_b)),
            'total_scene_channel_values': int(image_a.size),
            'scene_max_abs_uint8': int(np.abs(image_a.astype(int)-image_b.astype(int)).max()),
            'same_A_repeat': difference(outputs['A0'], outputs['A1']),
            'same_B_repeat': difference(outputs['B0'], outputs['B1']),
            'rgb_AB_effect': difference(outputs['A0'], outputs['B0']),
            'vs_previous_process': references, 'native_queue_parity': queue_rows})
        print(rows[-1], flush=True)
    np.savez_compressed(args.output/'predictions.npz', **predictions)
    write_json(args.output/'report.json', {'completed': True, 'rows': rows,
        'candidate_sha256': plan['candidate_sha256'],
        'source': archive_source('scripts/probe_cadence_fixed_inputs.py'),
        'native_api': 'lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy.select_action',
        'elapsed_s': time.monotonic()-started,
        'limits': 'Frozen-weight offline causal RGB and queue tests, not rollouts/training. Original historical raw RGB was not saved; no retrospective cadence-gate override.'})


if __name__ == '__main__':
    main()
