"""Exact raw actor-input identity; never transforms the policy's observations."""
import hashlib
import numpy as np


def inference_input_fingerprint(obs):
    state = np.asarray(obs['state'])
    if state.shape != (16,) or state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError('Expected exact finite float32 OpenArm state')
    instruction = obs['instruction']
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Expected nonempty instruction')
    arrays = {'state': state}
    for name in ('scene_rgb', 'wrist_rgb'):
        value = np.asarray(obs['images'][name])
        if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ValueError('Expected HWC uint8 camera input')
        arrays[name] = value
    return {'format': 'openarm-raw-actor-input-v1',
        'instruction_sha256': hashlib.sha256(instruction.encode('utf-8')).hexdigest(),
        'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                        'sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
                   for key, value in arrays.items()}}
