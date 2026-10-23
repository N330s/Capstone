"""Independent saved-motion/input/video audit for the native SmolVLA candidate."""
from pathlib import Path
import numpy as np
from learning.vla.runtime_io import read_json
from learning.vla.smolvla_data import sha256
from learning.vla.pickup_evaluation import audit_pickup_trace
from learning.vla.inference_cadence import audit_cadence_chunks
from learning.vla.inference_inputs import inference_input_fingerprint


TRAIN_PARENTS = ('train_20002','train_20003','train_20042','train_22001','train_22045')


def training_gate(rows):
    policies = [r for r in rows if r['group']=='training' and r['mode']=='policy']
    experts = [r for r in rows if r['group']=='training' and r['mode']=='replay']
    return (len(policies)==len(experts)==5 and
        {r['episode'] for r in policies}=={r['episode'] for r in experts}==set(TRAIN_PARENTS) and
        all(r['pickup_success'] and not r['safety_abort'] for r in policies+experts))


def audit_native_trial(folder, case, plan):
    import av
    folder = Path(folder)
    report, manifest = read_json(folder/'report.json'), read_json(folder/'manifest.json')
    parent = plan['parents'][case['episode']]
    expected = {'mode':case['mode'], 'execution_prefix':case['prefix'], 'policy_seed':1000,
        'inference_noise_schedule':case['noise_schedule'], 'inference_numerics':case['numerics'],
        'source_data_sha256':parent['data_sha256'], 'checkpoint_weights_sha256':plan['weights_sha256'],
        'recorded_input_assistance':False, 'simulator_state_forced_to_recording':False,
        'expert_overlay_weights_sha256':None, 'reset_state_max_error':0}
    if any(report[k]!=v for k,v in expected.items()): raise ValueError('Frozen candidate/source/input binding differs')
    if report['actuation_filter'] != {'gripper_resolution_mm':0., 'reference_or_phase_inputs':False, 'arm_targets_changed':False}:
        raise ValueError('Action filtering differs')
    if (report['grasp_metrics']['extension_sha256'] != plan['sources']['learning/vla/physical_grasp.py']['sha256']
            or report['pickup_evaluation']['helper_sha256'] != plan['sources']['learning/vla/pickup_evaluation.py']['sha256']):
        raise ValueError('Physical pickup implementation differs')
    for key, value in parent['manifest_binding'].items():
        if manifest[key]!=value: raise ValueError('Scene/robot/guard/reset differs: '+key)
    chunks = read_json(folder/'chunks.json')
    with np.load(folder/'trajectory.npz',allow_pickle=False) as trace:
        metrics = audit_pickup_trace(report, trace, read_json(folder/'diagnostics.json'))
        predictions = np.concatenate([np.asarray(c['prediction'])[:c['executed_steps']] for c in chunks])
        if not np.array_equal(predictions,trace['action_requested']): raise ValueError('Wrong executed action prefix')
    audit_cadence_chunks(chunks,prefix=case['prefix'],steps=report['steps'],seed=1000,schedule=case['noise_schedule'])
    if case['mode']=='replay':
        if not metrics['pickup_success'] or max(report['replay_errors'].values())!=0:
            raise ValueError('Exact expert control failed; do not score model')
    else:
        control = Path(plan['output'])/f"training_{case['episode']}_replay" if case['group']=='training' else (
            Path(plan['output'])/f"validation_{case['episode']}_replay")
        if manifest!=read_json(control/'manifest.json'): raise ValueError('Policy/expert environment differs')
        flags = report['torch_inference_flags']
        if flags != {'deterministic_algorithms':True,'cudnn_benchmark':False,'matmul_allow_tf32':False,
                     'cublas_workspace_config':':4096:8','forward_dtype':'torch.bfloat16'}:
            raise ValueError('Numerical profile differs')
        repeat = report['inference_repeat_audit']
        if repeat is None or repeat['max_abs_difference']>1e-6 or not repeat['before_any_command'] or not repeat['same_live_observation'] or repeat['noise_seed']!=1000:
            raise ValueError('Initial same-input/noise prediction audit failed')
        inputs = report['inference_input_audit']
        if (inputs is None or inputs['observations_modified'] or not inputs['per_chunk_fingerprints']
                or inputs['first_input_file']!='first_actor_input.npz'
                or inputs['first_input_sha256']!=sha256(folder/'first_actor_input.npz')
                or inputs['helper_sha256']!=plan['sources']['learning/vla/inference_inputs.py']['sha256']):
            raise ValueError('Raw-input evidence binding differs')
        with np.load(folder/'first_actor_input.npz',allow_pickle=False) as raw:
            fingerprint = inference_input_fingerprint({'state':raw['state'],'instruction':str(raw['instruction'].item()),
                'images':{key:raw[key] for key in ('scene_rgb','wrist_rgb')}})
        if fingerprint!=chunks[0]['input_fingerprint'] or any('input_fingerprint' not in c for c in chunks):
            raise ValueError('Input fingerprints missing/different')
        if report['inference_binding']['trainable_storage']['trainable_dtypes'] != {'torch.float32':99880992}:
            raise ValueError('Native trainable storage was downcast on reload')
    with av.open(str(folder/'rollout.mp4')) as video:
        if video.streams.video[0].average_rate!=50: raise ValueError('Video is not 50 simulated fps')
        frames = sum(1 for _ in video.decode(video=0))
    if frames!=report['steps']+1: raise ValueError('Missing actual video observations')
    return {**case, **metrics, 'steps':report['steps'], 'outcome':report['outcome'],
        'safety_abort':report['safety_abort'], 'directory':str(folder), 'video_frames':frames,
        'trajectory_sha256':sha256(folder/'trajectory.npz'), 'video_sha256':sha256(folder/'rollout.mp4'),
        'first_prediction_repeat_max_abs':report['inference_repeat_audit']['max_abs_difference'] if case['mode']=='policy' else None,
        'warm_inference_seconds':report['latency_warm_mean_s'], 'wall_s':report['wall_s']}
