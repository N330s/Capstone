"""Randomized table-top scenes: free plug + free socket, sampled with rejection.

Replaces the fixed ten-row bank. Two guarantees are kept from the old bank:

  1. Evaluation scenes live in a disjoint seed namespace and are never emitted by
     ``collection_bank``. ``assert_disjoint`` is called at import time.
  2. Sampling is a pure function of (master_seed, index), so any scene can be
     regenerated exactly from its id without storing the whole bank.

Rejection is *logged*, not silent: ``sample_scene`` returns the number of draws
it burned and the constraint that rejected each one, and the collector writes
those to ``rejected_scenes.json``.

Env contract -- ``env.reset(seed=..., options=...)`` must accept these keys and
place the bodies before the first observation:

    plug_pos_m        [x, y, z]  world, z is the table surface + half plug height
    plug_yaw_deg      float      rotation about world z
    socket_pos_m      [x, y, z]  world, socket_entry site sits at the face
    socket_yaw_deg    float
    socket_tilt_deg   float      tilt of the mating axis off horizontal
    table_height_m    float
    visual            dict       optional domain randomization (see WORKSPACE)

If your env names things differently, remap in ``as_reset_options`` only.
"""
from dataclasses import dataclass, field, asdict, replace
import argparse
import json
import numpy as np

COLLECTION_SEED_BASE = 1_000_000
EVALUATION_SEED_BASE = 9_000_000
SEED_NAMESPACE_WIDTH = 500_000


@dataclass(frozen=True)
class Workspace:
    """All lengths in metres, angles in degrees. Tune once against your model."""
    base_pos_m: tuple = (0.0, 0.0, 0.0)      # robot base, for the reach annulus
    table_height_m: float = 0.32
    plug_half_height_m: float = 0.012
    x_range: tuple = (0.30, 0.45)            # Changed from (0.30, 0.62)
    y_range: tuple = (-0.30, 0.0)           # Changed from (-0.30, 0.30)
    reach_range_m: tuple = (0.32, 0.66)      # planar distance from base
    min_separation_m: float = 0.16           # plug centre to socket face
    max_separation_m: float = 0.46
    plug_yaw_range_deg: tuple = (0,0)
    socket_yaw_range_deg: tuple = (0, 90.0)
    socket_tilt_range_deg: tuple = (-90.0, -90.0)
    standoff_m: float = 0.10                 # pre-insert point along -mating axis
    keepout_m: float = 0.09                  # plug must clear the socket approach cone
    randomize_visual: bool = True
    light_azimuth_deg: tuple = (-60.0, 60.0)
    light_elevation_deg: tuple = (35.0, 80.0)
    camera_jitter_m: float = 0.004
    camera_jitter_deg: float = 1.5
    table_hue_jitter: float = 0.08

WORKSPACE = Workspace()
MAX_DRAWS = 200


def assert_disjoint():
    lo, hi = COLLECTION_SEED_BASE, COLLECTION_SEED_BASE + SEED_NAMESPACE_WIDTH
    if lo <= EVALUATION_SEED_BASE < hi:
        raise RuntimeError("collection and evaluation seed namespaces overlap")

assert_disjoint()


def _yaw_axis(yaw_deg, tilt_deg):
    """Unit mating axis: the direction a plug travels to enter the socket."""
    y, t = np.radians(yaw_deg), np.radians(tilt_deg)
    return np.array([np.cos(y) * np.cos(t), np.sin(y) * np.cos(t), np.sin(t)])


def _planar_reach(pos, ws):
    return float(np.linalg.norm(np.asarray(pos)[:2] - np.asarray(ws.base_pos_m)[:2]))


def _check(plug, socket, axis, ws):
    """Return the name of the first violated constraint, or None."""
    standoff = socket - axis * ws.standoff_m
    if not ws.reach_range_m[0] <= _planar_reach(plug, ws) <= ws.reach_range_m[1]:
        return "plug_out_of_reach"
    if not ws.reach_range_m[0] <= _planar_reach(socket, ws) <= ws.reach_range_m[1]:
        return "socket_out_of_reach"
    if not ws.reach_range_m[0] <= _planar_reach(standoff, ws) <= ws.reach_range_m[1]:
        return "standoff_out_of_reach"
    separation = float(np.linalg.norm(plug - socket))
    if separation < ws.min_separation_m:
        return "objects_too_close"
    if separation > ws.max_separation_m:
        return "objects_too_far"
    # The plug must not sit inside the corridor the gripper sweeps on approach.
    along = float(np.dot(plug - socket, axis))
    lateral = float(np.linalg.norm((plug - socket) - along * axis))
    if -ws.standoff_m * 1.4 < along < 0.02 and lateral < ws.keepout_m:
        return "plug_blocks_socket_approach"
    return None


def _visual(rng, ws):
    if not ws.randomize_visual:
        return {}
    return {
        "light_azimuth_deg": float(rng.uniform(*ws.light_azimuth_deg)),
        "light_elevation_deg": float(rng.uniform(*ws.light_elevation_deg)),
        "camera_offset_m": rng.uniform(-ws.camera_jitter_m, ws.camera_jitter_m, 3).tolist(),
        "camera_rotation_deg": rng.uniform(-ws.camera_jitter_deg, ws.camera_jitter_deg, 3).tolist(),
        "table_hue_shift": float(rng.uniform(-ws.table_hue_jitter, ws.table_hue_jitter)),
    }


def sample_scene(seed, ws=WORKSPACE):
    """Pure function of the seed. Returns (options, diagnostics)."""
    rng = np.random.default_rng(seed)
    rejected = []
    for draw in range(MAX_DRAWS):
        z = ws.table_height_m + ws.plug_half_height_m
        plug = np.array([rng.uniform(*ws.x_range), rng.uniform(*ws.y_range), z])
        socket = np.array([rng.uniform(*ws.x_range), rng.uniform(*ws.y_range), z])
        socket_yaw = float(rng.uniform(*ws.socket_yaw_range_deg))
        socket_tilt = float(rng.uniform(*ws.socket_tilt_range_deg))
        axis = _yaw_axis(socket_yaw, socket_tilt)
        bad = _check(plug, socket, axis, ws)
        if bad:
            rejected.append(bad)
            continue
        options = {
            "plug_pos_m": plug.tolist(),
            "plug_yaw_deg": float(rng.uniform(*ws.plug_yaw_range_deg)),
            "socket_pos_m": socket.tolist(),
            "socket_yaw_deg": socket_yaw,
            "socket_tilt_deg": socket_tilt,
            "table_height_m": ws.table_height_m,
            "visual": _visual(rng, ws),
        }
        return options, {"draws": draw + 1, "rejected": rejected}
    raise ValueError(f"no feasible scene for seed {seed} in {MAX_DRAWS} draws; widen Workspace")


def as_reset_options(options):
    """Single place to remap keys if your env uses different names."""
    return dict(options)


def workspace_for_env(env, ws=WORKSPACE, **overrides):
    """A Workspace with base_pos_m read from the actual model instead of the
    (0,0,0) guess. Call this once with a live env before generating a bank:

        env = OpenArmInsertEnv(images=False)
        ws = workspace_for_env(env)
        bank = collection_bank(200, ws=ws)

    If your robot really is mounted at the world origin this changes nothing;
    if it isn't (e.g. it sits on a column, as in the screenshot where the arm
    was stretched flat-out reaching for a socket the sampler placed near the
    edge of an annulus centered on the wrong point), this corrects it.
    """
    base = tuple(float(v) for v in env.reach_anchor_m())
    return replace(ws, base_pos_m=base, **overrides)


def _row(index, seed, split, ws):
    options, diag = sample_scene(seed, ws)
    return {"id": f"{split}_{index:05d}", "seed": seed, "split": split,
            "options": options, "sampler": diag}


def collection_bank(n=200, validation_fraction=0.1, ws=WORKSPACE, start=0):
    """Deterministic list of n collectible scenes. Validation is a held-out slice
    of the same distribution; evaluation is a different namespace entirely."""
    n_val = int(round(n * validation_fraction))
    rows = []
    for i in range(start, start + n):
        seed = COLLECTION_SEED_BASE + i
        split = "validation" if (i - start) >= n - n_val else "train"
        rows.append(_row(i, seed, split, ws))
    return rows


def evaluation_bank(n=100, ws=WORKSPACE):
    """Reserved. Never executed by the collector, never used for tuning."""
    return [_row(i, EVALUATION_SEED_BASE + i, "evaluation", ws) for i in range(n)]


def _preview(argv=None):
    p = argparse.ArgumentParser(description="Print sampled scenes as JSON.")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--evaluation", action="store_true")
    a = p.parse_args(argv)
    bank = evaluation_bank(a.n) if a.evaluation else collection_bank(a.n)
    print(json.dumps(bank, indent=2))
    draws = [r["sampler"]["draws"] for r in bank]
    print(f"# mean draws per accepted scene: {np.mean(draws):.2f}", flush=True)


if __name__ == "__main__":
    _preview()