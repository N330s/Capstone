"""Training-state capture for deterministic, single-process checkpoint resume."""
import random

import numpy as np


def capture_rng():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def assert_state_equal(actual, expected, path="state"):
    """Compare nested optimizer/scheduler/RNG states without numeric tolerances."""
    import torch
    if isinstance(expected, torch.Tensor):
        if not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
            raise AssertionError(f"{path}: tensor mismatch")
    elif isinstance(expected, np.ndarray):
        if not np.array_equal(actual, expected):
            raise AssertionError(f"{path}: array mismatch")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: key mismatch")
        for key in expected:
            assert_state_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: length mismatch")
        for index, value in enumerate(expected):
            assert_state_equal(actual[index], value, f"{path}[{index}]")
    elif actual != expected:
        raise AssertionError(f"{path}: value mismatch")


def cpu_state(value):
    """Detach copies so continued updates cannot mutate the comparison reference."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.copy()
    return value
