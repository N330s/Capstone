"""Offline open-hold supervision and sensor-noise augmentation; no phase actor input."""
import numpy as np


def open_hold_anchors(states, actions, horizon=50):
    states, actions = np.asarray(states), np.asarray(actions)
    if (states.shape != (len(actions) + 1, 16) or actions.ndim != 2 or actions.shape[1] != 8
            or not np.isfinite(states).all() or not np.isfinite(actions).all() or horizon < 1):
        raise ValueError("Invalid physical state/action recording")
    eligible = []
    for index in range(len(actions) - horizon + 1):
        targets = actions[index:index + horizon, 7]
        if targets.min() >= .030 and np.ptp(targets) < 1e-7 and abs(states[index, 15] - targets[0]) < .001:
            eligible.append(index)
    return eligible


def perturbed_gripper(value, offset):
    if not np.isfinite([value, offset]).all() or not 0 <= value <= .0441 or abs(offset) > .002000001:
        raise ValueError("Expected physical finger travel and bounded two-mm sensor noise")
    return float(np.clip(value + offset, 0., .044))


def quantize_finger_command(command, resolution_m):
    """Task-independent actuator resolution, without changing arm targets/bounds."""
    value = np.asarray(command, dtype=float)
    if (value.shape != (8,) or not np.isfinite(value).all()
            or not np.isfinite(resolution_m) or not 0 < resolution_m <= .001):
        raise ValueError("Finite eight-target command and <=one-mm resolution required")
    result = value.copy()
    result[7] = np.floor(value[7] / resolution_m + .5) * resolution_m
    return result  # Environment still owns bounds, slew and contact guards.
