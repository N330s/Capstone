"""Qualified native LeRobot preflight then bounded four-hour clean SmolVLA SFT."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.runtime_io import read_json, write_json, write_status
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source
from scripts.run_smolvla_job import arm_deadline, kill_owned_process
from scripts.diagnose_cadence_repeatability import resources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--qualification', type=Path, default=ROOT/'outputs/native_qualification_20261021')
    p.add_argument('--prior-campaign', type=Path)
    args = p.parse_args()
    output = args.output.resolve()
    if output.parent != ROOT/'outputs' or output.exists(): raise ValueError('Dedicated new output root required')
    resources()
    parity = read_json(args.qualification/'report.json')
    if not parity.get('native_reference_exact') or parity['compared_updates'] != 2:
        raise ValueError('Native real-model parity must pass first')
    if parity['adapter']['sha256'] != sha256(ROOT/'learning/vla/native_smolvla.py'):
        raise ValueError('Qualified adapter changed')
    # Conservatively include the preceding standalone qualification (52s observed).
    prior_seconds = 60
    prior = None
    if args.prior_campaign:
        prior = args.prior_campaign.resolve()
        failure = read_json(prior/'failure.json')
        original_plan = read_json(prior/'plan.json')
        if (prior.parent != ROOT/'outputs' or original_plan['protocol'] != 'native-smolvla-clean-v1'
                or failure['completed'] or failure['stages'][-1]['stage'] != 'preflight'
                or any(s['stage']=='training' for s in failure['stages'])):
            raise ValueError('Only a stopped pre-training preflight may continue into a new campaign')
        # Count the stopped run AND intervening diagnosis/requalification wall time.
        prior_seconds = math.ceil(failure['elapsed_s']+max(0,time.time()-(prior/'failure.json').stat().st_mtime)+2)
    if prior_seconds >= 14400: raise TimeoutError('Original four-hour allowance already exhausted')
    begun = time.monotonic()-prior_seconds
    deadline = begun+14400
    output.mkdir()
    sources = {f: archive_source(f) for f in ('scripts/run_native_smolvla_campaign.py',
        'scripts/train_native_smolvla.py', 'scripts/qualify_native_smolvla.py',
        'learning/vla/native_smolvla.py', 'learning/vla/training_precision.py',
        'learning/vla/smolvla_policy.py', 'learning/vla/smolvla_data.py',
        'learning/vla/runtime_io.py', 'scripts/diagnose_cadence_repeatability.py')}
    export = ROOT/'data/drive_v2c30_20261012/export_all_v1'
    bindings = {str(f.relative_to(ROOT)): sha256(f) for f in
        [ROOT/'data/models/smolvla_base/model.safetensors',
         export/'openarm_export.json', export/'train/meta/stats.json', export/'train/meta/info.json']}
    if prior and bindings != original_plan['bindings']: raise ValueError('Prior base/data bindings changed')
    write_json(output/'plan.json', {'protocol': 'native-smolvla-clean-v1', 'total_limit_s': 14400,
        'charged_qualification_s': prior_seconds, 'sources': sources, 'bindings': bindings,
        'pretrained_base': 'data/models/smolvla_base', 'training_episodes': 81, 'validation_episodes': 20,
        'steps_ceiling': 20000, 'checkpoint_frequency': 5000, 'early_checkpoint': 1000,
        'training_changes': 'terminal-mask alias; FP32 trainable storage/Adam with native BF16 Accelerator',
        'native_components': ['dataset','pre/postprocessors','policy','flow loss','uniform sampling','optimizer','scheduler','training loop','checkpoint serialization'],
        'validation_scope': '160 fixed windows across all 20 validation episodes, at step 1/every1000/final; not full-frame validation',
        'sampling_heuristics': False, 'api_tokens': False, 'deletion': False,
        'limits': 'Four-hour total including benchmarks/preflight/reload. Offline SFT, not evidence of pickup/generalization. No scene/hardware/RL writes.'})
    if prior:
        write_json(output/'continuation.json', {'original': str(prior), 'charged_prior_seconds': prior_seconds,
            'prior_failure_sha256': sha256(prior/'failure.json'), 'prior_plan_sha256': sha256(prior/'plan.json'),
            'qualification_sha256': sha256(args.qualification/'report.json'),
            'reason': 'CLI empty-dictionary feature merge repaired; helper loss/precision requalified, compute benchmarks reused.'})
    stages = []
    def frozen():
        for name, record in sources.items():
            if sha256(ROOT/name) != record['sha256']: raise RuntimeError('Source changed during native campaign: '+name)
    def run(name, command, cap):
        frozen(); resources()
        cap = min(cap, deadline-time.monotonic())
        if cap < 60: raise TimeoutError('Campaign budget exhausted before stage')
        started = time.monotonic()
        snapshots = []
        with (output/(name+'.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            fired = threading.Event(); timer = arm_deadline(child, cap, fired)
            try:
                while child.poll() is None:
                    resource = resources(); snapshots.append(resource)
                    frozen()
                    write_status(output/'health.json', {'running': True, 'stage': name, 'pid': child.pid,
                        'elapsed_s': time.monotonic()-begun, 'remaining_s': deadline-time.monotonic(),
                        'case_elapsed_s': time.monotonic()-started, **resource})
                    time.sleep(3)
                code = child.wait()
            finally:
                if child.poll() is None: kill_owned_process(child)
                child.wait(); timer.cancel()
        row = {'stage': name, 'command': command, 'returncode': code, 'timed_out': fired.is_set(),
               'elapsed_s': time.monotonic()-started,
               'minimum_free_gpu_mib': min((r['gpu_free_mib'] for r in snapshots), default=None)}
        stages.append(row); write_json(output/'stages.json', stages)
        if code or fired.is_set(): raise RuntimeError('Native stage failed; inspect '+name+'.log')
        return snapshots
    common = [sys.executable, '-u', str(ROOT/'scripts/qualify_native_smolvla.py')]
    try:
        candidates = []
        for size in (4,8):
            destination = (prior or output)/f'batch_{size}'
            initial_gpu_free = resources()['gpu_free_mib']
            if not prior:
                run(f'batch_{size}', [*common, '--mode', 'benchmark', '--batch-size', str(size),
                                     '--output', str(destination)], 300)
            report = read_json(destination/'report.json')
            if not report.get('completed') or report['base_sha256'] != parity['base_sha256']:
                raise ValueError('Unqualified benchmark source/base')
            # Baseline includes desktop/CUDA driver usage; reserve at least 1 GiB beyond PyTorch peak.
            conservative_free = initial_gpu_free-report['peak_reserved_mib']
            safe = conservative_free >= 1024 and all(math.isfinite(e[k]) for e in report['events'] for k in ('loss','grad_norm'))
            candidates.append({'batch_size': size, 'safe': safe, 'conservative_free_mib': conservative_free,
                'samples_per_second': size/report['warm_update_seconds'], 'report': str(destination/'report.json')})
        safe = [c for c in candidates if c['safe']]
        if not safe: raise RuntimeError('No native batch passed conservative memory/finite-gradient qualification')
        batch = max(safe, key=lambda c:c['samples_per_second'])['batch_size']
        write_json(output/'batch_selection.json', {'selected_batch': batch, 'candidates': candidates,
            'limits': 'Native fixed-batch compute benchmark, then end-to-end dataloader preflight. Sizes above 8 not tested to prioritize safe startup.'})
        receipt = read_json(export/'openarm_export.json')
        def train_command(destination, max_seconds, stop_after=None, validation=False):
            command = [sys.executable, '-u', str(ROOT/'scripts/train_native_smolvla.py'),
                '--native-max-seconds', str(max_seconds),
                f'--policy.path={ROOT / "data/models/smolvla_base"}', '--policy.input_features={}',
                f'--policy.vlm_model_name={ROOT / "data/models/smolvlm_backbone"}',
                '--policy.load_vlm_weights=false', '--policy.device=cuda', '--policy.push_to_hub=false',
                '--policy.freeze_vision_encoder=true', '--policy.train_expert_only=true', '--policy.train_state_proj=true',
                '--policy.adapt_to_pi_aloha=false', '--policy.use_delta_joint_actions_aloha=false', '--policy.empty_cameras=0',
                f'--dataset.repo_id={receipt["datasets"]["train"]["repo_id"]}', f'--dataset.root={export / "train"}',
                '--dataset.video_backend=pyav', '--dataset.image_transforms.enable=false',
                f'--output_dir={destination}', '--steps=20000', f'--batch_size={batch}', '--num_workers=0',
                '--seed=1000', '--eval_freq=0', '--log_freq=50', '--save_freq=5000', '--wandb.enable=false']
            if stop_after: command += ['--native-stop-after', str(stop_after)]
            if validation: command += ['--native-validation']
            return command
        preflight = output/'preflight'
        run('preflight', train_command(preflight, 600, stop_after=12), 900)
        checkpoint = preflight/'checkpoints/000012'
        status = read_json(preflight/'native_status.json')
        if status.get('type') != 'completed' or status['last_saved_step'] != 12:
            raise ValueError('Native preflight did not complete its checkpoint')
        complete = read_json(checkpoint/'openarm_native_complete.json')
        if not complete['complete'] or sha256(checkpoint/'pretrained_model/model.safetensors') != complete['weights_sha256']:
            raise ValueError('Native preflight checkpoint differs')
        reload_dir = output/'reload'
        run('reload', [*common, '--mode','reload','--checkpoint',str(checkpoint/'pretrained_model'),
                       '--batch-size','1','--output',str(reload_dir)], 300)
        if not read_json(reload_dir/'report.json').get('all_saved_tensors_reload_exact'):
            raise ValueError('Native checkpoint reload unqualified')
        remaining = deadline-time.monotonic()
        if remaining < 600: raise TimeoutError('Insufficient remaining budget for useful training')
        # Fresh original pretrained initialization, not the preflight actor or any old fine-tuned actor.
        run('training', train_command(output/'training', remaining-180, validation=True), remaining)
        final = read_json(output/'training/native_status.json')
        if final.get('type') != 'completed' or not final.get('last_saved_step'):
            raise RuntimeError('Native training did not finish with a checkpoint')
        frozen()
        for relative, expected in bindings.items():
            if sha256(ROOT/relative) != expected: raise RuntimeError('Initialization/data binding changed')
        write_json(output/'completion.json', {'completed': True, 'elapsed_s': time.monotonic()-begun,
                                             'training': final, 'batch_size': batch})
        write_status(output/'health.json', {'running': False, 'completed': True, 'elapsed_s': time.monotonic()-begun})
    except BaseException as error:
        failure = {'completed': False, 'elapsed_s': time.monotonic()-begun, 'error': repr(error), 'stages': stages,
                   'automatic_retry': False}
        write_json(output/'failure.json', failure)
        write_status(output/'health.json', {'running': False, **failure})
        raise


if __name__ == '__main__': main()
