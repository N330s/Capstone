"""Compare fresh-process/input repeatability without relaxing the cadence gate."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.runtime_io import read_json, write_json
from learning.vla.source_archive import archive_source
from learning.vla.smolvla_data import sha256


def difference(a, b):
    if a.shape != (50, 8) or b.shape != a.shape or not np.isfinite([a, b]).all():
        raise ValueError('Invalid full physical-unit chunks')
    return {'max_abs': float(np.abs(a-b).max()),
            'joint_max_abs_rad': float(np.abs(a[:, :7]-b[:, :7]).max()),
            'finger_max_abs_mm': float(np.abs(a[:, 7]-b[:, 7]).max()*1000)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fixed-inputs', type=Path)
    p.add_argument('--repeat-fixed', type=Path)
    p.add_argument('--input-smoke', type=Path)
    args = p.parse_args()
    if args.repeat_fixed and not args.fixed_inputs:
        raise ValueError('Replication requires its primary fixed-input audit')
    if not read_json(args.probe/'health.json').get('completed'):
        raise ValueError('Probe incomplete')
    reports = {f'{profile}_{i}': read_json(args.probe/f'{profile}_{i}'/'report.json')
               for profile in ('legacy', 'deterministic') for i in range(2)}
    parents = ['train_20003', 'train_22045', 'train_20002']
    rows = []
    for profile in ('legacy', 'deterministic'):
        left, right = (reports[f'{profile}_{i}'] for i in range(2))
        with np.load(args.probe/f'{profile}_0'/'predictions.npz') as a, \
                np.load(args.probe/f'{profile}_1'/'predictions.npz') as b:
            for parent in parents:
                matching = lambda report: [r for r in report['rows'] if r['episode'] == parent]
                ar, br = matching(left), matching(right)
                rows.append({'profile': profile, 'episode': parent,
                    'loaded_state_identical': left['loaded_state_sha256'] == right['loaded_state_sha256'],
                    'raw_inputs_identical': all(r['input_hashes'] == ar[0]['input_hashes'] for r in ar+br),
                    'processed_inputs_identical': all(r['processed_hashes'] == ar[0]['processed_hashes'] for r in ar+br),
                    'cuda_rng_advanced': any(r['cuda_rng_advanced'] for r in ar+br),
                    'within_process': [difference(z[f'{parent}_{i}'], z[f'{parent}_0'])
                                       for z in (a, b) for i in (1, 2)],
                    'across_process': difference(a[f'{parent}_0'], b[f'{parent}_0'])})
    old = ROOT/'outputs/inference_cadence_20261020_resume_v1'
    conditions = read_json(old/'conditions.json')
    original = []
    for parent in parents:
        chunks = {r['prefix']: np.asarray(read_json(Path(r['directory'])/'chunks.json')[0]['prediction'])
                  for r in conditions if r['mode'] == 'policy' and r['episode'] == parent}
        original.append({'episode': parent, 'initial_prediction_differences_vs_prefix25':
                         {str(k): difference(v, chunks[25]) for k, v in chunks.items()}})
    result = {'completed': True, 'rows': rows, 'original_campaign': original,
        'sources': {'probe_request_sha256': sha256(args.probe/'request.json'),
                    'reporter': archive_source('scripts/report_cadence_repeatability.py')},
        'limits': 'Offline reset/inference audit only. The original gate remains failed; no rescoring, retraining or held-out generalization claim.'}
    if args.fixed_inputs:
        fixed = read_json(args.fixed_inputs/'report.json')
        if not fixed.get('completed') or not read_json(args.fixed_inputs/'health.json').get('completed'):
            raise ValueError('Fixed-input audit incomplete')
        with np.load(args.fixed_inputs/'predictions.npz') as values:
            for row in fixed['rows']:
                parent = row['episode']
                for fixture in row['fixtures'].values():
                    if sha256(fixture['path']) != fixture['sha256']:
                        raise ValueError('Captured input fixture changed')
                for label in ('A', 'B'):
                    if difference(values[parent+'_'+label+'0'], values[parent+'_'+label+'1']) != row['same_'+label+'_repeat']:
                        raise ValueError('Repeated fixed-input metrics differ')
                if difference(values[parent+'_A0'], values[parent+'_B0']) != row['rgb_AB_effect']:
                    raise ValueError('RGB intervention metrics differ')
                for queue in row['native_queue_parity']:
                    native = values[f"{parent}_native_prefix{queue['prefix']}"]
                    reference = values[parent+'_A0'][:queue['prefix']]
                    if (native.shape != reference.shape or
                        float(np.abs(native-reference).max()) != queue['max_abs_difference'] or
                        bool(np.array_equal(native, reference)) != queue['exact_equal']):
                        raise ValueError('Native queue metrics differ')
        result['fixed_input_audit'] = fixed
        result['sources']['fixed_input_report_sha256'] = sha256(args.fixed_inputs/'report.json')
        result['sources']['fixed_input_predictions_sha256'] = sha256(args.fixed_inputs/'predictions.npz')
        if args.repeat_fixed:
            if not read_json(args.repeat_fixed/'health.json').get('completed'):
                raise ValueError('Fixed-input replication incomplete')
            repeated = read_json(args.repeat_fixed/'report.json')
            if repeated['source']['sha256'] != fixed['source']['sha256']:
                raise ValueError('Replication changed probe implementation')
            if (repeated['candidate_sha256'] != fixed['candidate_sha256'] or
                    [(r['episode'], r['fixtures']) for r in repeated['rows']] !=
                    [(r['episode'], r['fixtures']) for r in fixed['rows']]):
                raise ValueError('Replication changed actor or fixed observations')
            with np.load(args.fixed_inputs/'predictions.npz') as a, \
                    np.load(args.repeat_fixed/'predictions.npz') as b:
                if set(a.files) != set(b.files): raise ValueError('Different replication cases')
                differences = {key: float(np.abs(a[key]-b[key]).max()) for key in a.files}
            result['same_profile_fresh_process_replication'] = {
                'directory': str(args.repeat_fixed), 'max_abs_by_array': differences,
                'all_exact_equal': all(v == 0 for v in differences.values()),
                'report_sha256': sha256(args.repeat_fixed/'report.json'),
                'predictions_sha256': sha256(args.repeat_fixed/'predictions.npz'),
                'limits': 'Same numerical settings, fixed RGB/state/noise and call order. Does not isolate the earlier legacy/different-call-order discrepancy.'}
    if args.input_smoke:
        from scripts.evaluate_inference_cadence import audit_condition
        smoke_report = read_json(args.input_smoke/'report.json')
        if smoke_report['steps'] != 1 or smoke_report['safety_abort']:
            raise ValueError('Expected one safe logging-smoke step, not a scored pickup trial')
        # A separate prospective smoke binding; never edits the original plan.
        smoke_plan = read_json(old/'plan.json')
        helper = archive_source('learning/vla/inference_inputs.py')
        smoke_plan['sources']['learning/vla/inference_inputs.py'] = helper
        case = {'episode': smoke_report['source_episode'], 'prefix': smoke_report['execution_prefix'],
                'mode': 'policy', 'noise_schedule': 'step-keyed'}
        result['input_logging_smoke'] = {'audit': audit_condition(args.input_smoke, case, smoke_plan),
            'first_input_sha256': sha256(args.input_smoke/'first_actor_input.npz'),
            'helper': helper, 'limits': 'One-step logging smoke only; not task success or a new cadence score.'}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'summary.json', result)
    if args.fixed_inputs:
        with (args.output/'rgb_and_native_queue.csv').open('w', newline='', encoding='utf-8') as stream:
            columns = ['episode', 'scene_changed_values', 'wrist_changed_values', 'image_max_uint8',
                       'rgb_effect_joint_max_rad', 'rgb_effect_finger_max_mm',
                       'fixed_repeat_max_abs', 'native_queue_all_equal', 'cross_context_max_abs']
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in fixed['rows']:
                writer.writerow({'episode': row['episode'],
                    'scene_changed_values': row['image_differences']['scene_rgb']['changed_channel_values'],
                    'wrist_changed_values': row['image_differences']['wrist_rgb']['changed_channel_values'],
                    'image_max_uint8': max(v['max_abs_uint8'] for v in row['image_differences'].values()),
                    'rgb_effect_joint_max_rad': row['rgb_AB_effect']['joint_max_abs_rad'],
                    'rgb_effect_finger_max_mm': row['rgb_AB_effect']['finger_max_abs_mm'],
                    'fixed_repeat_max_abs': max(row['same_A_repeat']['max_abs'], row['same_B_repeat']['max_abs']),
                    'native_queue_all_equal': all(v['exact_equal'] for v in row['native_queue_parity']),
                    'cross_context_max_abs': max(v['max_abs'] for v in row['vs_previous_process'].values())})
    for row in rows:
        print(row['profile'], row['episode'],
              {k: row[k] for k in ('loaded_state_identical', 'raw_inputs_identical', 'processed_inputs_identical')},
              'within', max(r['max_abs'] for r in row['within_process']), 'across', row['across_process'])


if __name__ == '__main__':
    main()
