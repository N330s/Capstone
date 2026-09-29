# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MuJoCo simulation of an OpenArm v1 bimanual robot (right arm active, left parked) picking up a
two-blade plug from a table and inserting it into a socket, plus the data pipeline and VLA
(pi0 / SmolVLA) adapters intended to train on scripted-expert demonstrations. No VLA has been
trained yet; the roadmap is in `plan.md`, current status in `docs/*.md`.

## Environment setup

- Python 3.11 + `mujoco==3.11.0` (`requirements.txt`). Windows; PowerShell commands in docs.
- Robot assets are **not** in git. Fetch once with `powershell -File scripts/fetch_openarm.ps1`
  (clones `third_party/openarm_mujoco` pinned to commit `161039cd…`; refuses to overwrite).
  `envs/scene.py` raises `FileNotFoundError` without it. `scripts/fetch_openpi.ps1` similarly pins
  openpi source (optional; only `learning/vla/openpi_integration.py` imports it, and only on a
  Linux GPU host).
- `data/` (collected episodes) and `third_party/` are gitignored. `results/` is committed evidence.
- Always run from the repository root: scripts do `sys.path.insert(0, <root>)`, tests import
  top-level packages (`envs`, `connector`, …).
- On this machine use `.venv\Scripts\python.exe` (Python 3.12 venv with `mujoco==3.11.0`);
  `python` on PATH may be the Microsoft Store alias. `third_party/openarm_mujoco` is fetched.

## Commands

```powershell
python -m unittest discover -s tests -v          # all tests (40); needs third_party assets
python -m unittest tests.test_connector -v       # holder-only tests, no robot assets needed
python -m unittest tests.test_openarm.OpenArmTests.test_v1_bimanual_and_no_weld  # single test
```

Only `tests/test_connector.py`, `tests/test_workspace.py` and `tests/test_vla_adapter.py` run without `third_party/`;
`test_openarm`, `test_table_pickup`, `test_carry_path`, `test_collection` build the robot scene
(slow: each `OpenArmInsertEnv()` compiles the model and settles 1 s of physics at 2 kHz).

Heavier acceptance runs (each writes a report to `--output`, which **must not already exist**):

```powershell
python scripts/validate_connector.py                  # holder benchmark repeatability/timestep checks
python scripts/validate_openarm.py                    # 20 aligned + signed offsets + jam recovery
python scripts/validate_table_pickup.py               # 8 pickup cases
python scripts/probe_table_insert.py --render --output results/full_task_new   # full task, exit 0/1 (identity socket; --socket-roll-deg 90 for the old variant)
python scripts/calibrate_insertion_force.py --output results/insertion_force_new  # robot-free leaf calibration (mocap-driven plug)
python scripts/calibrate_insertion_force.py --workspace configs/workspace_v2.json --plug-type type_o --output results/insertion_force_new_o  # catalog plug variant
python scripts/replay_table_insert.py --episode results/full_task_new
python scripts/collect_varied.py --record --output data/openarm_v1_varied_new  # omit --record = preflight
python scripts/replay_pilot.py --dataset data/openarm_v1_varied_new
python scripts/prepare_vla_data.py --dataset data/openarm_v1_varied_new
python scripts/validate_vla_adapter.py --replay --output results/vla_adapter_new_run
python scripts/view_openarm.py [--play] [--task insertion]   # interactive viewer (Space run, R reset)
python scripts/test_insert.py --offset-y-mm 0.5 ; python scripts/sweep_alignment.py   # holder-only
```

## Architecture

Layers, bottom to top (each imports only from below):

1. **`connector/`** — robot-free foundation. `geometry.py` composes
   `assets/connector/plug_socket.xml` (+ `plug.xml`, `socket.xml` includes) and optionally patches in
   the 1 mm lead-in variant. `simulation.py` has `ConnectorMetrics` (geometry-aware success oracle:
   blade depth, slot-corridor fit, face gap, orientation, contact force/penetration, hold time) and
   `ConnectorSimulation`/`run_trial` (the ideal spring-damper "holder" benchmark, config in
   `configs/holder.json`). `ConnectorMetrics.bind(model, data, config, fingers)` is reused by the
   robot env so both scenes share one success definition. `spec.py`/`catalog.py`/`slab.py`/
   `builder.py` are the **parametric connector catalog** (`docs/CONNECTOR_CATALOG.md`): named plug
   and socket variants (`legacy_two_blade`, `type_a/b/c/o`, `universal_th`) generated from a
   `ConnectorSpec`, which is embedded in the model and read back with `ConnectorSpec.from_model` —
   metrics, leaves, env and probes never copy a dimension. `legacy_two_blade` reproduces the
   hand-written XML exactly, so a spec-less scene is unchanged. Select one with `--plug-type` /
   `--socket` or a workspace `connector` section; `derived()` supplies every distance the
   controllers used to hard-code (`preinsert_x_m`, `success_depth_m`, `housing_width_m`, …).
2. **`envs/`** — `scene.py` builds the robot scene XML at runtime: parses the upstream v1 bimanual
   XML, injects solver options (2 kHz, elliptic cone, impratio 1000), the `robot_grasp` site, the
   lead-in connector bodies, and everything described by **`configs/workspace_v1.json`** through
   `envs/workspace.py` (table + legs + pedestal, appliance, 14-segment ball-joint cable rooted at
   the plug with a `connect` to the appliance, socket spring leaves, finger pad plates, cameras +
   action-cam visual). `docs/WORKSPACE.md` tabulates every dimension; edit the JSON, never
   duplicate numbers in code. `openarm_insert.py` is `OpenArmInsertEnv`, a gym-like
   `reset(seed, options)/step(action)` env; `env.place_plug(pos, quat)` is the only sanctioned way
   to position the plug (it also lays the cable consistently). `OpenArmInsertEnv(workspace=...)` and
   the scripts' `--workspace` flag select a spec: **`configs/workspace_v2.json`** is the Type O /
   `universal_th` variant with a 1.5 m cable to a floor appliance. The full task passes there in one
   run (`results/full_task_v2_type_o_cable150_fix`; `..._cable150` is the preserved pre-fix abort),
   but the held-plug expert is not validated in v2. See the workspace_v2 section of
   `docs/WORKSPACE.md` before using it, and do not collect data in it yet.
3. **`controllers/`** — `expert.py` (privileged held-plug insertion expert with jam retry),
   `table_pickup.py` (`PickupProbe`: downward-rest → raise → approach → descend → close → lift),
   `carry_path.py` (`plan_carry`: bidirectional RRT with a rigid carried-plug proxy on scratch
   `MjData`), `chunks.py` (`execute_chunk`, the one executor policies must also use).
   Note `controllers/table_pickup.py` imports `inspect_ik`/`unexpected_contacts` from
   `scripts/preview_table_task.py` (a script acting as a library).
4. **`data_pipeline/`** — `episodes.py` (lossless NPZ episode + metadata/diagnostics JSON,
   `validate_episode` checks checksums), `collection_bank.py` (fixed 10-reset collection bank and a
   100-reset evaluation bank that must never be used for collection/tuning), `vla_dataset.py`
   (train-only normalization, framework-neutral samples).
5. **`learning/vla/`** — pi0 adapter (`OpenArmCodec` owns normalization; upstream `norm_stats`
   must be `{}`), `FullWindowPilotDataset` (complete same-episode windows only), optional upstream
   transform wiring.

`scripts/probe_table_insert.py` is the current full-task demonstrator: it subclasses `PickupProbe`,
optionally rolls the socket **at model-construction time** (`--socket-roll-deg`, default 0 = the
workspace orientation; `m.body_quat`, before reset), plans with `plan_carry`, then runs a
force-aware closed-loop Jacobian insertion (lateral gain 2x with a +/-1 mm anti-wind-up bound,
target advances only below `push_force_cap_n`). It is a scripted probe, not yet an env.

### Contracts that everything depends on

- **Action** (8): absolute right joint1..7 targets in rad, then per-finger slide travel in m
  (0..0.044; not aperture). **State** (16): left joint1..7 + mean finger travel, then right same.
  Cable/leaf joints are physics-only: `nq = 25 + 4*segments + 2`, never in the state vector.
  Command rate 50 Hz, physics 0.5 ms → `env.substeps == 40`. Rate limits are applied to targets
  inside `step`; `info["applied_action"]` is what actually reached the controller.
- Motor control is joint PD + `qfrc_bias` feedforward through upstream torque-limited actuators
  (`configs/openarm_v1.json`). Right fingers are position servos, left fingers are force actuators.
- The plug is always a free body held by physical finger contact. Never weld it, apply external
  force, or overwrite its pose outside `reset`/probe reset.
- Privileged info (`env.info`, `ConnectorMetrics.diagnostics()`, IK, planner) is for experts,
  labels and aborts only; policy observations are RGB + 16-d state + instruction.
- `contact_force_n` is rigid-wall force only; spring-leaf load is `leaf_*`, cable load is
  `cable_*` (from the `cable_root` force/torque sensors). With leaves enabled the wall abort is
  `wall_force_abort_n` (20 N) instead of holder.json's 8 N. Cable-robot contact is reported
  (`cable_robot_contact_n`), not aborted. `plan_carry` ignores the cable (stale on scratch data).
- `solve_ik`, `plan_carry` and `inspect_ik` write only to scratch `MjData`; a test asserts live
  `qpos/qvel` are untouched.
- Frames: socket +X is insertion, Y separates blades, Z is blade width; roll/pitch/yaw =
  Rz(yaw)·Ry(pitch)·Rx(roll). CLI angles are degrees and offsets mm; internals are SI.
- Every `step` calls `mj_forward` at the command boundary; replay must reproduce this exactly
  (a replay audit once failed by omitting it). Strict replay tolerance is 1e-8 qpos / 1e-7 qvel.

## Provenance and evidence conventions

- `env.manifest()` embeds SHA-256 of `envs/scene.py`, `envs/workspace.py`, `envs/openarm_insert.py`,
  `connector/geometry.py`, `connector/simulation.py`, `controllers/expert.py`, the scene XML hash and
  `configs/workspace_v1.json` (`manifest["workspace"]`); `replay_pilot.py` refuses to replay datasets
  whose hashes differ. **Editing those files invalidates strict replay of all existing datasets** —
  expect to recollect or document. Everything recorded before workspace_v1 is already in that state.
  `.gitattributes` sets `* -text` so hashed source bytes survive checkout.
- Output directories are created with `exist_ok=False` and collectors refuse to overwrite.
  Write new runs to new `results/<name>` / `data/<name>` paths; existing `results/` are the
  cited evidence for docs and must stay intact (including preserved *failed* audits).
- Physics/solver/contact/gain changes are made one at a time and recorded in
  `PHYSICS_CHANGELOG.md` with measured effect. The holder benchmark keeps its original solver
  settings so the straight-vs-lead-in comparison stays matched; don't "harmonize" it.
- Docs deliberately state what is *not* validated (e.g. "scripted demonstrator, not a trained
  policy"; "not a sim-to-real result"). Keep that precision when updating `README.md`, `plan.md`
  or `docs/`; do not upgrade claims without new evidence in `results/`.
- Bank IDs, seeds and splits are fixed (`collection_bank`, `evaluation_bank`); normalization is
  fit on training episodes only and splits are whole-episode.
