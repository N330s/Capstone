"""Bounded offline repeatability audit of the frozen cadence actor; no training/steps."""
import argparse
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.runtime_io import read_json, write_json
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source


def digest_array(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, str(value.dtype))).encode() + value.tobytes()).hexdigest()


def worker(output, profile):
    # Must precede CUDA initialization. No global/site-packages changes.
    if profile == 'deterministic':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    from learning.vla.smolvla_policy import SmolVLAController
    from learning.vla.smolvla_data import observation
    from learning.vla.physical_grasp import install_physical_grasp_metrics
    from scripts.rollout_smolvla import verify_snapshot
    from torch.utils.data import default_collate
    if profile == 'deterministic':
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    campaign = ROOT / 'outputs/inference_cadence_20261020_resume_v1'
    plan = read_json(campaign / 'plan.json')
    base = ROOT / 'outputs/v2c30_sft_overnight_v1/checkpoints/step_0009426_best/pretrained_model'
    overlay = ROOT / 'outputs/five_episode_openhold_20261020/training/expert'
    if sha256(overlay / 'expert.safetensors') != plan['candidate_sha256']:
        raise ValueError('Frozen candidate changed')
    actor = SmolVLAController(base, ROOT / 'data/models/smolvlm_backbone', 1000, expert_overlay=overlay)
    # Hash values, not just the source file, to expose partial/random loading.
    parameter_digest = hashlib.sha256()
    for name, value in actor.model.state_dict().items():
        parameter_digest.update(name.encode())
        parameter_digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        parameter_digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    snapshot = ROOT / 'third_party/capstone_automateData'
    sys.path.insert(0, str(snapshot))
    from envs.openarm_insert import OpenArmInsertEnv
    if not Path(sys.modules['envs.openarm_insert'].__file__).resolve().is_relative_to(snapshot):
        raise ValueError('Wrong scene import')
    rows, predictions = [], {}
    for parent in plan['parents']:
        source = ROOT / 'data/drive_v2c30_20261012/merged_all_v1' / parent
        metadata = read_json(source / 'metadata.json')
        if sha256(source / 'episode.npz') != metadata['data_sha256']:
            raise ValueError('Source episode changed')
        workspace = verify_snapshot(snapshot, metadata)
        env = OpenArmInsertEnv(images=True, workspace=workspace)
        install_physical_grasp_metrics(env)
        try:
            obs, _ = env.reset(seed=metadata['seed'], options=metadata['reset_options'])
            inputs = {'state': obs['state'], **obs['images'], 'qpos': env.data.qpos.copy(),
                      'qvel': env.data.qvel.copy()}
            input_hashes = {key: digest_array(value) for key, value in inputs.items()}
            np.savez_compressed(output / (parent + '_reset.npz'), **inputs)
            for repeat in range(3):
                # Same public controller API as the original campaign, no model reset intervention.
                batch = actor.pre(default_collate([observation(obs)]))
                processed_hashes = {k: digest_array(v.detach().cpu().contiguous().view(torch.uint8).numpy())
                                    for k, v in batch.items() if isinstance(v, torch.Tensor)}
                rng_before = digest_array(torch.cuda.get_rng_state().numpy())
                value, latency = actor.predict(obs, noise_seed=1000)
                rng_after = digest_array(torch.cuda.get_rng_state().numpy())
                if value.shape != (50, 8) or not np.isfinite(value).all():
                    raise ValueError('Invalid prediction')
                key = f'{parent}_{repeat}'
                predictions[key] = value
                rows.append({'episode': parent, 'repeat': repeat, 'input_hashes': input_hashes,
                             'processed_hashes': processed_hashes, 'prediction_sha256': digest_array(value),
                             'cuda_rng_advanced': rng_before != rng_after, 'latency_s': latency})
        finally:
            env.close()
    np.savez_compressed(output / 'predictions.npz', **predictions)
    write_json(output / 'report.json', {'completed': True, 'profile': profile, 'rows': rows,
        'loaded_state_sha256': parameter_digest.hexdigest(), 'binding': actor.binding,
        'torch': torch.__version__, 'lerobot': version('lerobot'),
        'deterministic': torch.are_deterministic_algorithms_enabled(),
        'cublas_workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'limits': 'Three selected training resets; reset/render and offline inference only, no env.step or optimizer.'})


def resources():
    import psutil
    from learning.vla.inference_cadence import resource_violations
    row = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu,memory.free',
                                  '--format=csv,noheader,nounits'], text=True, timeout=10).strip().splitlines()[0]
    temperature, free_gpu = map(float, row.split(','))
    values = {'temperature_c': temperature, 'gpu_free_mib': free_gpu,
              'ram_available_bytes': psutil.virtual_memory().available,
              'disk_free_bytes': shutil.disk_usage(ROOT).free}
    violations = resource_violations(**values)
    if violations:
        raise RuntimeError(str((violations, values)))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=('legacy', 'deterministic'))
    args = parser.parse_args()
    if args.worker:
        worker(args.output, args.worker)
        return
    import psutil
    started = time.monotonic()
    resources()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'request.json', {'source': archive_source('scripts/diagnose_cadence_repeatability.py'),
        'total_limit_s': 600, 'worker_limit_s': 180, 'training': False, 'physics_steps': 0})
    for profile in ('legacy', 'deterministic'):
        for index in range(2):
            folder = args.output / f'{profile}_{index}'
            folder.mkdir()
            with (folder / 'worker.log').open('w') as log:
                proc = subprocess.Popen([sys.executable, __file__, '--output', str(folder), '--worker', profile],
                                        stdout=log, stderr=subprocess.STDOUT)
                begun = time.monotonic()
                try:
                    while proc.poll() is None:
                        elapsed = time.monotonic() - started
                        if elapsed > 600 or time.monotonic() - begun > 180:
                            raise TimeoutError('Offline probe deadline')
                        write_json(args.output / 'health.json', {'running': True,
                            'stage': folder.name, 'elapsed_s': elapsed, **resources()})
                        time.sleep(2)
                    if proc.returncode:
                        raise RuntimeError(f'Worker failed: {folder}/worker.log')
                finally:
                    if proc.poll() is None:
                        process = psutil.Process(proc.pid)
                        for child in process.children(recursive=True):
                            try: child.kill()
                            except psutil.NoSuchProcess: pass
                        try: process.kill()
                        except psutil.NoSuchProcess: pass
                        proc.wait(timeout=10)
    write_json(args.output / 'health.json', {'running': False, 'completed': True,
                                          'elapsed_s': time.monotonic() - started})


if __name__ == '__main__':
    main()
