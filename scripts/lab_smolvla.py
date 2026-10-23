"""Check, qualify, smoke-test or explicitly launch a bounded native lab run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.lab_training import (
    load_config, verify_inputs, run_folder, train_command, qualification_command, throughput,
)
from learning.vla.runtime_io import read_json, write_json
from learning.vla.runtime_resources import resources
from learning.vla.process_control import arm_deadline, kill_owned_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'qualify', 'smoke', 'train'))
    parser.add_argument('--config', type=Path, default=ROOT/'configs/lab_smolvla.json')
    parser.add_argument('--name', help='Fresh directory name below the configured runs root')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--steps', type=int, help='Additional updates, not a cumulative old checkpoint step')
    parser.add_argument('--dry-run', action='store_true', help='Verify inputs and print commands without starting a job')
    args = parser.parse_args()
    begun = time.monotonic()
    cfg = load_config(args.config)
    for field, value in (('batch_size', args.batch_size), ('num_workers', args.workers), ('additional_steps', args.steps)):
        if value is not None:
            if value < (0 if field == 'num_workers' else 1):
                raise ValueError('Invalid override: '+field)
            cfg[field] = value
    bindings = verify_inputs(cfg)
    if args.action == 'check':
        print(json.dumps(bindings, indent=2))
        return
    output = run_folder(cfg, args.name)
    training = output/'training'
    budget = min(cfg['wall_seconds'], 900) if args.action != 'train' else cfg['wall_seconds']
    commands = ([qualification_command(cfg, output/'parity')] if args.action == 'qualify' else
        [train_command(cfg, training, steps=cfg['additional_steps'] if args.action == 'train' else (args.steps or 64),
            max_seconds=budget-120, smoke=args.action == 'smoke')])
    plan = {'action': args.action, 'config': {k: str(v) if isinstance(v, Path) else v for k,v in cfg.items()},
        'bindings': bindings, 'commands': commands, 'wall_seconds': budget,
        'initialization': 'Original full81 checkpoint; NEW optimizer/scheduler, NOT an exact resume',
        'automatic_deletion': False, 'automatic_retry': False}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    resources(cfg['runs'])
    output.mkdir()
    write_json(output/'plan.json', plan)
    os.environ.setdefault('OPENARM_SOURCE_ARCHIVE', str(cfg['runs']/'source_archives'))
    deadline = begun+budget
    def run(command, log_name):
        remaining = deadline-time.monotonic()
        if remaining < 30:
            raise TimeoutError('No budget left for the next stage')
        with (output/log_name).open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            fired = threading.Event()
            timer = arm_deadline(child, remaining, fired)
            try:
                while child.poll() is None:
                    resources(cfg['runs'])
                    time.sleep(3)
                if child.wait() or fired.is_set():
                    raise RuntimeError('Stage failed or exceeded wall budget; inspect '+log_name)
            finally:
                if child.poll() is None:
                    kill_owned_process(child)
                child.wait()
                timer.cancel()
    try:
        run(commands[0], 'run.log')
        if args.action == 'qualify':
            report = read_json(output/'parity/report.json')
            if not report.get('native_reference_exact'):
                raise ValueError('Native/reference parity did not pass')
        else:
            status = read_json(training/'native_status.json')
            if status.get('type') != 'completed' or not status.get('last_saved_step'):
                raise ValueError('Training ended without a completed native checkpoint')
            checkpoints = list((training/'checkpoints').glob('*/openarm_native_complete.json'))
            selected = max(checkpoints, key=lambda p: read_json(p)['step'])
            checkpoint = selected.parent/'pretrained_model'
            run(qualification_command(cfg, output/'reload', checkpoint=checkpoint), 'reload.log')
            if not read_json(output/'reload/report.json').get('all_saved_tensors_reload_exact'):
                raise ValueError('Fresh checkpoint reload failed')
            events = [json.loads(line) for line in (training/'native_events.jsonl').read_text().splitlines()]
            report = {'training': status, 'checkpoint': str(checkpoint), 'reload_exact': True,
                'throughput': throughput(events, cfg['batch_size']) if len([e for e in events if e['type']=='update'])>=10 else None,
                'limits': 'Offline qualification only; live pickup evaluation is a separate required gate.'}
        if verify_inputs(cfg) != bindings:
            raise ValueError('Frozen inputs changed during the job')
        write_json(output/'completion.json', {'completed': True, 'elapsed_s': time.monotonic()-begun, **report})
    except BaseException as error:
        write_json(output/'failure.json', {'completed': False, 'elapsed_s': time.monotonic()-begun,
            'error': repr(error), 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
