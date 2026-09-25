"""Randomized table-top scenes: free plug + free socket, sampled with rejection.

Replaces the fixed ten-row bank. Two guarantees are kept from the old bank:

  1. Evaluation scenes live in a disjoint seed namespace and are never emitted by
     ``collection_bank``. ``assert_disjoint`` is called at import time.
  2. Sampling is a pure function of (master_seed, index), so any scene can be
     regenerated exactly from its id without storing the whole bank.

Rejection is *logged*, not silent: ``sample_scene`` returns the number of draws
it burned and the constraint that rejected each one, and the collector writes
those to ``rejected_scenes.json``.

Besides the geometric constraints, each draw must pass a kinematic pick check
(``Workspace.grasp_feasibility``): the expert's own planner
(``controllers.pick_insert_expert.PickFeasibility``) must find a top-down grasp,
descend, lift and transit from the reset posture. It runs on a private model
with fixed IK seeds, so it is deterministic and sampling stays a pure function
of the seed. Changing the expert's grasp constants can therefore change which
scenes a seed produces; rejections are logged as ``pick_infeasible:<reason>``.

Env contract -- ``env.reset(seed=..., options=...)`` must accept these keys and
place the bodies before the first observation:

    plug_pos_m        [x, y, z]  world, z is the table surface + housing half-height
    plug_yaw_deg      float      rotation about world z
    socket_pos_m      [x, y, z]  world, socket_entry site sits at the face
    socket_yaw_deg    float      heading of the mating axis (direction the plug travels)
    socket_tilt_deg   float      tilt of the mating axis off horizontal
    table_height_m    float
    visual            dict       optional domain randomization (see WORKSPACE)

If your env names things differently, remap in ``as_reset_options`` only.

Socket distribution (``sample_socket`` / ``Workspace.socket_*``)
----------------------------------------------------------------
The socket stands upright on the table: tilt 0, its bottom face flush with the
table top (socket_entry is SOCKET_BASE_BELOW_ENTRY_M above the surface), so it is
supported and never sinks into the table. Its face looks back at the robot:
the mating axis is horizontal and heads away from the robot (heading -10..30
deg about world +x), so the plug, held in a top-down grasp with its blades
pointing away from the robot, is pushed horizontally into the face. The pads
stay behind the housing front face and ~8 mm above the table.

Why this heading: the closed fingers are 61 mm wide along the plug's mating
axis and overlap the socket face laterally, so only grasps whose fingertips lean
back from the blades clear the face when seated. For the right arm those are
reachable with the socket facing the robot (hundreds of IK-checked poses) and
almost nowhere with it facing away (heading 180: only an untilted grasp in a
small corner). The old distribution (tilt 180 at table-surface height) put the
socket upside down and 3 mm into the table and left no finger clearance.

Each socket pose must also pass a deterministic kinematic insertion screen
(``controllers.place_insert.InsertFeasibility``): for at least one nominal
top-down grasp the seated pose, the straight insertion line, the preinsert pose
and the vertical descent must be reachable with joint-limit margin and free of
carried-plug/finger collisions. It runs on a private model with fixed IK seeds,
so sampling stays a pure function of the seed. Rejections are logged as
``insert_infeasible:<reason>``.
"""
from dataclasses import dataclass, field, asdict, replace
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

# The socket boxes (assets/connector/socket.xml) reach 9.2 + 5.8 mm below
# socket_entry and the lead-in meshes 15.14 mm: an upright socket resting on the
# table has its entry this far above the surface (0.06 mm clear).
SOCKET_BASE_BELOW_ENTRY_M = 0.0152

COLLECTION_SEED_BASE = 1_000_000
EVALUATION_SEED_BASE = 9_000_000
SEED_NAMESPACE_WIDTH = 500_000


@dataclass(frozen=True)
class Workspace:
    """All lengths in metres, angles in degrees. Tune once against your model."""
    # Right-arm mount (OpenArmInsertEnv.reach_anchor_m()), not the world origin.
    base_pos_m: tuple = (0.0, -0.031, 0.698)
    table_height_m: float = 0.32
    plug_half_height_m: float = 0.008        # housing half-height: the plug rests at this z
    x_range: tuple = (0.30, 0.45)            # fallback for socket_x/y_range if unset
    y_range: tuple = (-0.30, 0.0)
    # Top-down grasps are only reachable in part of the table (the arm runs out
    # of reach beyond x~0.42 and the wrist runs out of range near y~0).
    plug_x_range: tuple = (0.30, 0.42)
    plug_y_range: tuple = (-0.30, -0.12)
    reach_range_m: tuple = (0.32, 0.66)      # planar distance from base
    min_separation_m: float = 0.16           # plug centre to socket face
    max_separation_m: float = 0.46
    # A top-down pinch does not reorient the housing about world z, so this is
    # (up to the grasp's flip/tilt) the yaw the plug is carried at, and it DOES
    # have to roughly match the socket's heading: place_insert.py's half-turn
    # equivalence (ConnectorMetrics' two equal blades) is a symmetry about the
    # horizontal mating axis (a roll), not about world z, so it cannot rescue a
    # plug picked up pointing the wrong way in yaw -- a plug grasped near 180
    # deg from the socket heading has its blade tip aimed away from the socket
    # and no top-down re-grasp can flip that. A kinematic sweep (yaw_study,
    # 2026-09-24) over plug_yaw in -180..180 crossed with the plug/socket
    # ranges below found GraspPlanner.plan_grasp's chosen pick is
    # insert-consistent almost exactly when pick-feasible for yaw in about
    # (-45, 75), peaking at 0-15 deg (matches the socket_yaw_range_deg below),
    # and effectively never for yaw beyond about +/-90 deg (including the old
    # (150, 210), which was tuned for pick-only reachability and never checked
    # against the socket).
    plug_yaw_range_deg: tuple = (-30.0, 60.0)
    # Socket (see module docstring's "Socket distribution"). Upright on the
    # table, face toward the robot; x/y ranges bound where the insertion
    # screen finds plans.
    socket_x_range: tuple = (0.26, 0.42)
    socket_y_range: tuple = (-0.34, -0.06)
    socket_yaw_range_deg: tuple = (-10.0, 30.0)
    socket_tilt_range_deg: tuple = (0.0, 0.0)
    socket_base_below_entry_m: float = SOCKET_BASE_BELOW_ENTRY_M
    insert_feasibility: bool = True
    standoff_m: float = 0.10                 # pre-insert point along -mating axis
    keepout_m: float = 0.09                  # plug must clear the socket approach cone
    # Reject draws the expert cannot plan a pick for (kinematic, deterministic;
    # see PickFeasibility). require_insert_feasible also demands that the same
    # grasp can reach the socket standoff and seated poses.
    grasp_feasibility: bool = True
    require_insert_feasible: bool = True
    # A scene that is only pick-feasible at its exact (privileged) pose is not
    # safe to certify: the noisy Detector's per-episode bias + per-step noise
    # (controllers.pick_insert_expert.Detector) moves the plug/socket the
    # expert actually plans against, and a few mm can drop a marginal grasp
    # below MIN_JOINT_MARGIN_RAD at runtime (see docs/AUTOMATED_COLLECTION_LOG.md,
    # collection-bank row 105 / seed 1000105). When True, also re-run the pick
    # check at the plug/socket pose perturbed by pick_noise_sigma combined
    # standard deviations of that noise model, one axis at a time (x, y, yaw),
    # and reject the scene if any perturbed pose is infeasible.
    pick_noise_margin: bool = True
    pick_noise_sigma: float = 3.0
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


_FEASIBILITY = {}


def _pick_check(options, ws):
    """Name of the reason the expert cannot plan this pick, or None.

    Pure function of the options: a private model, fixed IK seeds, the reset
    posture. Imported lazily so the sampler stays importable without MuJoCo
    work until a check is actually needed."""
    if not ws.grasp_feasibility:
        return None
    key = (float(ws.table_height_m), bool(ws.require_insert_feasible))
    if key not in _FEASIBILITY:
        from controllers.pick_insert_expert import PickFeasibility
        _FEASIBILITY[key] = PickFeasibility(key[0], require_insert=key[1])
    return _FEASIBILITY[key](options)


_NOISE_BOUNDS = None


def _detector_noise_bounds(n_sigma):
    """(position_m, yaw_deg) bound at ``n_sigma`` combined standard deviations
    of the noisy Detector's per-episode bias plus per-step measurement noise
    (bias and noise are independent zero-mean, so their variances add). Read
    from ``Detector.__init__``'s defaults so this stays in sync with the
    runtime noise model instead of duplicating magic numbers."""
    global _NOISE_BOUNDS
    if _NOISE_BOUNDS is None:
        import inspect
        from controllers.pick_insert_expert import Detector
        p = inspect.signature(Detector.__init__).parameters
        pos_sigma = float(np.hypot(p["bias_pos_m"].default, p["noise_pos_m"].default))
        yaw_sigma = float(np.hypot(p["bias_yaw_deg"].default, p["noise_yaw_deg"].default))
        _NOISE_BOUNDS = (pos_sigma, yaw_sigma)
    pos_sigma, yaw_sigma = _NOISE_BOUNDS
    return n_sigma * pos_sigma, n_sigma * yaw_sigma


def _perturbed(options, key_pos, key_yaw, axis, delta):
    o = dict(options)
    if axis == "yaw":
        o[key_yaw] = float(options[key_yaw]) + delta
    else:
        pos = list(options[key_pos])
        pos[0 if axis == "x" else 1] += delta
        o[key_pos] = pos
    return o


def _pick_noise_margin_check(options, ws):
    """None if the pick plan survives the plug/socket pose perturbed by the
    noisy Detector's error bound, one axis at a time; else the axis/reason
    that failed. Axis-aligned (not a full cross product) because that already
    finds real violations cheaply (see sensitivity study in
    docs/AUTOMATED_COLLECTION_LOG.md) and stays deterministic and bounded-cost:
    2 objects x 3 axes x 2 signs = 12 extra PickFeasibility calls, only paid
    for draws that already pass the exact-pose check."""
    if not ws.pick_noise_margin:
        return None
    pos_bound, yaw_bound = _detector_noise_bounds(ws.pick_noise_sigma)
    for obj, key_pos, key_yaw in (("plug", "plug_pos_m", "plug_yaw_deg"),
                                   ("socket", "socket_pos_m", "socket_yaw_deg")):
        for axis, bound in (("x", pos_bound), ("y", pos_bound), ("yaw", yaw_bound)):
            for sign in (-1.0, 1.0):
                perturbed = _perturbed(options, key_pos, key_yaw, axis, sign * bound)
                bad = _pick_check(perturbed, ws)
                if bad:
                    tag = f"{obj}_{axis}{'+' if sign > 0 else '-'}"
                    return f"noise_margin:{tag}:{bad}"
    return None


def sample_socket(rng, ws=WORKSPACE):
    """Socket entry position, mating heading and tilt. Four rng draws, in the
    same order as the original sampler (x, y, yaw, tilt), so seeds keep
    producing the same draw sequence as the range fields are tuned."""
    x_range = ws.socket_x_range or ws.x_range
    y_range = ws.socket_y_range or ws.y_range
    socket = np.array([rng.uniform(*x_range), rng.uniform(*y_range),
                       ws.table_height_m + ws.socket_base_below_entry_m])
    return socket, float(rng.uniform(*ws.socket_yaw_range_deg)), float(rng.uniform(*ws.socket_tilt_range_deg))


_INSERT_FEASIBILITY = []


def _insert_check(options, ws):
    """Reason the socket pose admits no insertion plan, or None. Deterministic
    (private model, fixed seeds). Imported lazily: MuJoCo work only when needed."""
    if not ws.insert_feasibility:
        return None
    if not _INSERT_FEASIBILITY:
        from controllers.place_insert import InsertFeasibility
        _INSERT_FEASIBILITY.append(InsertFeasibility())
    reason, _ = _INSERT_FEASIBILITY[0](options)
    return reason


def sample_scene(seed, ws=WORKSPACE):
    """Pure function of the seed. Returns (options, diagnostics)."""
    rng = np.random.default_rng(seed)
    rejected = []
    plug_x = ws.plug_x_range or ws.x_range
    plug_y = ws.plug_y_range or ws.y_range
    for draw in range(MAX_DRAWS):
        z = ws.table_height_m + ws.plug_half_height_m
        plug = np.array([rng.uniform(*plug_x), rng.uniform(*plug_y), z])
        socket, socket_yaw, socket_tilt = sample_socket(rng, ws)
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
        }
        bad = _pick_check(options, ws)
        if bad:
            rejected.append(f"pick_infeasible:{bad}")
            continue
        bad = _pick_noise_margin_check(options, ws)
        if bad:
            rejected.append(f"pick_infeasible:{bad}")
            continue
        bad = _insert_check(options, ws)
        if bad:
            rejected.append(f"insert_infeasible:{bad}")
            continue
        options["visual"] = _visual(rng, ws)
        return options, {"draws": draw + 1, "rejected": rejected}
    raise ValueError(f"no feasible scene for seed {seed} in {MAX_DRAWS} draws; widen Workspace")


# On-disk cache for sample_scene(seed, ws): generation now runs the expert's
# kinematic pick/insert planner (PickFeasibility, InsertFeasibility, plus the
# noise-margin sweep) up to 25x per candidate draw, so a bank of any size is
# expensive to build from scratch. The cache key is (seed, a hash of the
# Workspace field values, a hash of the source that decides what a seed
# produces: this module plus the feasibility checks it calls into). Any edit
# to Workspace or to that source changes the key, so a stale hit is
# impossible -- at worst a change costs a slower first read, never a wrong
# one. Lives under data/.scene_cache, which is gitignored like the rest of
# data/; nothing here is committed or part of dataset provenance.
CACHE_ROOT = Path(__file__).resolve().parents[1] / "data" / ".scene_cache"
_SAMPLING_SOURCES = ("data_pipeline/scene_bank.py",
                    "controllers/pick_insert_expert.py",
                    "controllers/place_insert.py")

_source_hash_cache = None


def _source_hash():
    """Hash of the source files that determine sample_scene's output: this
    module plus the pick/insert feasibility checks it imports. Read fresh
    from disk (not from imported module state) so an edit is picked up even
    if this process already imported the old bytecode."""
    global _source_hash_cache
    if _source_hash_cache is None:
        root = Path(__file__).resolve().parents[1]
        h = hashlib.sha256()
        for rel in _SAMPLING_SOURCES:
            h.update((root / rel).read_bytes())
        _source_hash_cache = h.hexdigest()[:16]
    return _source_hash_cache


def _workspace_hash(ws):
    payload = json.dumps(asdict(ws), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_path(ws):
    return CACHE_ROOT / f"{_workspace_hash(ws)}_{_source_hash()}.json"


_cache_mem = {}


def _load_cache(path):
    if path not in _cache_mem:
        try:
            _cache_mem[path] = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            _cache_mem[path] = {}
    return _cache_mem[path]


def _save_cache(path, cache):
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    os.replace(tmp, path)


def cached_sample_scene(seed, ws=WORKSPACE):
    """sample_scene(seed, ws), memoized on disk. Returns the exact (options,
    diagnostics) sample_scene would return -- values round-trip through JSON
    unchanged since sample_scene already returns plain floats/lists/strings,
    so a cache hit is byte-identical to a fresh draw."""
    path = _cache_path(ws)
    cache = _load_cache(path)
    key = str(seed)
    if key in cache:
        entry = cache[key]
        return entry["options"], entry["sampler"]
    options, diag = sample_scene(seed, ws)
    cache[key] = {"options": options, "sampler": diag}
    _save_cache(path, cache)
    return options, diag


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
    options, diag = cached_sample_scene(seed, ws)
    return {"id": f"{split}_{index:05d}", "seed": seed, "split": split,
            "options": options, "sampler": diag}


def iter_collection_bank(n=200, validation_fraction=0.1, ws=WORKSPACE, start=0):
    """Same rows as collection_bank(n, validation_fraction, ws, start), one at
    a time. A caller that stops early (e.g. once it has enough accepted
    scenes) never pays to generate the rows it doesn't consume -- unlike
    collection_bank, which materializes the whole n up front. Split/seed
    assignment is computed identically to collection_bank for the same
    (n, start, validation_fraction), so switching between the two changes
    nothing about which seed lands in which split."""
    n_val = int(round(n * validation_fraction))
    for i in range(start, start + n):
        seed = COLLECTION_SEED_BASE + i
        split = "validation" if (i - start) >= n - n_val else "train"
        yield _row(i, seed, split, ws)


def collection_bank(n=200, validation_fraction=0.1, ws=WORKSPACE, start=0):
    """Deterministic list of n collectible scenes. Validation is a held-out slice
    of the same distribution; evaluation is a different namespace entirely."""
    return list(iter_collection_bank(n, validation_fraction, ws, start))


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