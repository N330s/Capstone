"""Strict replay of a saved episode: re-issue every recorded command and compare the physics."""
from __future__ import annotations
from pathlib import Path
import numpy as np

from data_pipeline.episodes import validate_episode

MANIFEST_KEYS = ("scene_sha256", "source_hashes", "robot_config", "contact_config",
                 "mujoco_version", "workspace")
QPOS_TOL, QVEL_TOL = 1e-8, 1e-7


def replay_episode(env, path):
    """Replay `path` on `env` -> {"max_qpos_error", "max_qvel_error", "outcome"}.

    Raises ValueError (changed manifest) or AssertionError (action/outcome mismatch or drift beyond
    1e-8 qpos / 1e-7 qvel). Render state is irrelevant; set env.images = False to skip cameras."""
    path = Path(path)
    meta = validate_episode(path)
    env.reset(seed=meta["seed"], options=meta["reset_options"])
    current = env.manifest()
    for key in MANIFEST_KEYS:
        if current[key] != meta[key]:
            raise ValueError(f"Cannot replay changed {key}")
    with np.load(path / "episode.npz", allow_pickle=False) as a:
        max_error = float(np.max(np.abs(env.data.qpos - a["qpos"][0])))
        velocity_error = float(np.max(np.abs(env.data.qvel - a["qvel"][0])))
        for i, action in enumerate(a["action_requested"]):
            _, reward, term, trunc, info = env.step(action)
            max_error = max(max_error, float(np.max(np.abs(env.data.qpos - a["qpos"][i + 1]))))
            velocity_error = max(velocity_error, float(np.max(np.abs(env.data.qvel - a["qvel"][i + 1]))))
            if not np.array_equal(info["applied_action"], a["action_applied"][i]):
                raise AssertionError("Action execution mismatch")
            if reward != a["reward"][i] or term != a["terminated"][i] or trunc != a["truncated"][i]:
                raise AssertionError("Outcome timing mismatch")
    if max_error > QPOS_TOL or velocity_error > QVEL_TOL:
        raise AssertionError(f"Replay drift: {max_error}, {velocity_error}")
    return {"max_qpos_error": max_error, "max_qvel_error": velocity_error, "outcome": env.outcome}
