# Workspace specification

Two workspaces exist. **workspace_v1** is described in full below and is the one the full task
passes in. **workspace_v2** — the Thai Type O connector from
[`docs/CONNECTOR_CATALOG.md`](CONNECTOR_CATALOG.md) and a 1.5 m cable running to the floor — is a
delta on top of it and is written up in the last section; the full task does **not** pass there yet.

## workspace_v1

Source of truth: [`configs/workspace_v1.json`](../configs/workspace_v1.json). `envs/scene.py`
generates the floor, table, pedestal, fixture, appliance, cable, socket spring leaves, finger pads
and cameras from that file through `envs/workspace.py`; nothing below is typed twice in code. Its
SHA-256 is part of `env.manifest()["workspace"]` and of every full-task report, so a run always
names the exact layout it was produced with. Edit the JSON, not this page, and keep both in step.

All dimensions are **design values chosen for the simulation**. None is a measurement of real
hardware; the sim-to-real caveats in `docs/SIM2REAL_CONTRACT.md` apply to every row.

### Frames

- World origin = OpenArm v1 base body (`openarm_body_link0`). +X runs across the table away from
  the robot, +Y along the table, +Z up.
- The table top is **0.32 m above the robot base**. This is the same relationship as the earlier
  floating-slab scene, so every plug / socket / waypoint coordinate in `configs/table_task_v*.json`
  and the controllers is unchanged. The floor was moved down instead (z = −0.43 m).
- Socket frame: +X insertion, Y separates the blades, Z is the blade width (see README).

### Top view (not to scale)

```
            +Y (table length 1.20 m) ─────────────────────────────►
   y=+0.60 ┌───────────────────────────────────────────────────────┐ x=0.85 (far edge)
           │ ▪leg                                             leg▪ │
           │                                                       │
           │   fixture post + socket  (0.4391,-0.155,0.478)        │
           │           ┃                                           │
           │   plug spawn (0.381,-0.22,0.330) ═════╗               │
           │           ╚═ cable (0.35 m, 14 segments) ═╗           │
           │                                appliance ▓▓▓ (0.46,-0.42,0.35)
           │ ▪leg                                             leg▪ │
   y=-0.60 └───────────────────────────────────────────────────────┘ x=0.25 (near edge)
                         ┌──────────┐
                         │ pedestal │  robot base at (0,0,0), 0.25 m behind the near edge
                         └──────────┘
```

### Table, floor, pedestal

| Item | Size (L×W×H) | Position / extent | Notes |
| --- | --- | --- | --- |
| Table top | 120 × 60 × 3 cm | centre (0.55, 0, 0.305) m; x 0.25…0.85, y ±0.60, top z = 0.32 | geom `work_table` (name kept for older scripts) |
| Legs (4) | 5 × 5 × 72 cm | inset 3 cm from each edge; centres x ∈ {0.305, 0.795}, y ∈ {±0.545}; z −0.43…0.29 | `table_leg_{near,far}_{left,right}` |
| Table height | 75 cm | floor z = −0.43 → top z = +0.32 | `frames.floor_z_m` must equal top − height (validated) |
| Floor | 6 × 6 m plane | z = −0.43 | |
| Robot pedestal | 20 × 26 × 37 cm | centre (−0.052, 0.032); z −0.43…−0.06 | column under the upstream base meshes (they reach z ≈ −0.06) |
| Fixture post | 1.2 × 4.8 × 15.8 cm | centre (0.4631, −0.155, 0.399); z 0.32…0.478 | unchanged from earlier scenes |

### Socket and retention leaves

| Item | Value |
| --- | --- |
| Socket entry frame | (0.4391, −0.155, 0.478) m, identity orientation (+X insertion) |
| Slots | 2 × (2.3 × 6.8 mm), 13 mm apart, 18 mm deep, 1 mm lead-in (unchanged connector) |
| Spring leaves | one per slot on the **outer** wall, pressing the blade toward the socket centre |
| Leaf plate | 12 mm long (socket x 4…16 mm), 5 mm tall, 0.3 mm embedded in the wall, 0.65 mm max protrusion |
| Ramp | 8 mm long, 0.95 mm rise (≈ 6.8°); blades meet the ramp at ≈ 9.9 mm depth, full plate from 12 mm |
| Leaf spring | slide joint, k = 4000 N/m, damping 10 N·s/m, armature 0.2 kg, travel −2…0 mm, preload 11.5 N at the stop |
| Interference | outer-wall clearance 0.40 mm − protrusion 0.65 mm → 0.25 mm deflection when engaged |
| Normal force per leaf | 12.5 N engaged → insertion / withdrawal ≈ 2 · μ · N = **10 N** with μ = 0.4 |
| Calibrated | `results/insertion_force_v1` (harness): −9.96 N insert, +9.96 N withdraw on the flat; 5 N variant in `results/insertion_force_v1_5n` |
| Abort limits | leaf force 80 N, leaf penetration 0.3 mm, rigid-wall force 20 N (was 5–8 N for the rigid socket, see changelog), wall penetration 0.2 mm |

### Plug and cable

| Item | Value |
| --- | --- |
| Plug | 28 × 24 × 16 mm housing, two 16 × 1.5 × 6 mm blades, 30 g (unchanged) |
| Spawn | mating frame at (0.381, −0.22, 0.330), identity orientation; housing rests at z = 0.328 |
| Cable | 0.35 m, Ø 6 mm, 70 g/m (24.5 g total), 14 capsule segments of 25 mm |
| Attachment | rigid at the strain relief, plug-local (−0.040, 0, 0); far end `connect` equality to the appliance anchor |
| Bending | ball joint per segment, 0.02 N·m/rad stiffness, 0.002 N·m·s/rad damping, armature 1e-6 kg·m² |
| Rest shape | circular arc from the spawn pose to the anchor (chord 0.209 m, slack 0.141 m) bulging toward −X (toward the robot), lying on the table top |
| Contacts | collides with everything (table, floor, appliance, fixture, socket, robot) except itself; friction 0.6 |
| Anchor | appliance-local (−0.06, 0, −0.015) → world (0.40, −0.42, 0.335), on the appliance's −X face |
| Appliance | 12 × 8 × 6 cm box, centre (0.46, −0.42, 0.35), static |

When the plug is placed anywhere else (held-plug reset, pickup reset), `OpenArmInsertEnv.place_plug`
recomputes the arc from the current plug pose to the anchor and writes the ball-joint quaternions,
so the cable starts kinematically consistent (no constraint whip). The chord must stay shorter than
the cable; `place_plug` raises otherwise.

### Gripper

| Item | Value |
| --- | --- |
| Finger servos | position servos, kp raised 100 → 2000 N/m (`configs/openarm_v1.json: finger_servo_kp_n_m`) → ≈ 11 N per pad at the held grasp |
| Pad plates | 24 × 16 × 1 mm box on each right finger, 0.1 mm proud of the upstream mesh face (finger-local y = ±7.8 mm, z 56…80 mm) |
| Pad contact | friction (1.0, 0.01, 0.01), condim 4, solref 2 ms, priority 2 → 4 contact points per pad on the housing |

### Cameras

| Camera | Parent | Position | Aim | FOV |
| --- | --- | --- | --- | --- |
| `scene_rgb` | world | (0.85, −0.90, 0.85) | xyaxes (0.8, 0.6, 0, −0.25, 0.333, 0.91) | 48° |
| `inspection` | world | (0.56, −0.32, 0.59) | xyaxes (0.75, 0.66, 0, −0.35, 0.4, 0.847) | 45° |
| `wrist_rgb` | `openarm_right_hand` | hand-local (0.05, −0.04, 0.01) | hand-local (0, 0.006, 0.123) | 55° |

The wrist camera carries an action-camera style visual body: 62 × 45 × 32 mm housing 18 mm behind
the optical centre, Ø 18 × 8 mm lens, and a Ø 10 mm bracket to hand-local (0.02, −0.02, 0). All
three geoms are visual only (no mass, no collision). `wrist_rgb` intrinsics are unchanged.

### Solver options (robot scene only)

2 kHz, `implicitfast`, Newton 100 iterations, elliptic cone, **impratio 1000** (was 100). The
holder benchmark keeps its own original settings so the straight-vs-lead-in comparison stays matched.

### Model bookkeeping

- `nq = 25 + 4·14 + 2 = 83`, `nv = 24 + 3·14 + 2 = 68` (arm/fingers/plug + cable balls + leaf slides).
- Bodies `cable_seg_00…13`, joints `cable_j_00…13`, sites `cable_root` (force/torque sensors),
  `cable_end`, `cable_anchor`; bodies `socket_leaf_left/right` with joints `*_slide`; contact
  excludes `socket_leaf_*_wall` (the static socket is welded to the world, so the parent-child
  filter would not stop the wall from colliding with the embedded leaf).
- Policy observations are unchanged: RGB + 16-d arm/finger state + instruction. Cable and leaf
  states are physics only; they appear in `info` (privileged) and in recorded `qpos/qvel`.

### What is not validated

- Cable stiffness, mass and friction are plausible cord values, not measurements.
- The 10 N retention target and its calibration are simulation-internal; no real socket was measured.
- The finger pad plates and the 2000 N/m servo gain are simulation choices that make the grasp
  capable of reacting cable and retention loads; real gripper force, pad compliance and slip are
  open sim-to-real items.
- Datasets and results produced before workspace_v1 no longer replay strictly (scene hash changed).

## workspace_v2

Source of truth: [`configs/workspace_v2.json`](../configs/workspace_v2.json). Same file format and
the same `envs/workspace.py` generator, so only the **differences from workspace_v1** are listed
here. The connector itself is described in [`docs/CONNECTOR_CATALOG.md`](CONNECTOR_CATALOG.md).

> **Status: not passing.** The scene is defined and the connector is calibrated, but the full task
> aborts with this spec. See "Full-task status" below before using workspace_v2 for anything.

### What differs

| Item | workspace_v1 | workspace_v2 |
| --- | --- | --- |
| Connector | legacy two-blade (no `connector` section) | `type_o` in `universal_th` |
| Socket entry | (0.4391, −0.155, 0.478) | (0.4455, −0.155, 0.472) |
| Fixture post | centre (0.4631, −0.155, 0.399), 1.2 × 4.8 × 15.8 cm | centre (0.4745, −0.155, 0.415), 1.2 × 5.2 × 19 cm |
| Plug spawn | mating frame z = 0.330 | z = 0.326 |
| Cable | 0.35 m, 14 segments, 70 g/m (24.5 g) | 1.5 m, 60 segments, 35 g/m (52.5 g) |
| Cable route | one arc across the table top | three arcs: table → over the near edge → floor coil |
| Appliance | 12 × 8 × 6 cm on the table at (0.46, −0.42, 0.35) | same box on the **floor** at (0.36, −0.22, −0.40) |
| Cable anchor | (0.40, −0.42, 0.335) | (0.30, −0.22, −0.415) |
| Leaves | 2, one per slot | 3, one per opening the plug uses (see catalog for the axes) |
| `wall_force_abort_n` | 20 N | 30 N |

Table, legs, floor, pedestal, gripper pads, servo gain, cameras and solver options are identical.

The socket moves **6.4 mm further from the robot** so the home pose clears the 21.4 mm earth pin
(pre-insert distance 27.4 mm) and **6 mm lower** because an earthed plug's grasp frame sits 6 mm
above its mating frame. The fixture post follows. Everything plug-dependent (socket x, spawn z,
cable attachment) is checked against the connector spec by `envs/workspace.validate`, so a mismatch
is an error at load time rather than a silent offset.

### Cable route

The 1.5 m cord is laid as three arcs through two waypoints, at the plug's own y so a randomised
spawn keeps the route:

1. along the table top to just past the robot-side edge (x = 0.235, z = 0.33), bulging +Y;
2. straight down to the floor — 31 segments over the 0.75 m drop (x = 0.22, z = −0.42);
3. the remaining ~0.55 m coiled on the floor under the table to the appliance box.

At the table spawn pose the generated rest shape ends exactly on the anchor (0.00 mm), with 7 of
the 60 segments on the table and 52 below it.

Cable mass was **halved, 70 → 35 g/m**, during this layout: with 0.75 m hanging to the floor the
heavier cord pulled the 45 g plug across the table (≈0.5 N against ≈0.2 N of table friction). At
35 g/m the edge friction leaves ≈0.1 N on the plug.

### Model bookkeeping

`nq = 25 + 4·60 + 3 = 268`, `nv = 24 + 3·60 + 3 = 207`, 94 bodies. Leaf geoms
`socket_leaf_{left,right,earth}`. Finger travel derives from the 34 mm housing: contact at
18.8 mm, insertion grip 13.0 mm, pickup 11.0 mm (v1: 13.8 / 8.0 / 6.0 for the 24 mm housing).

### Full-task status

**Update (2026-09-28): the full task passes with the committed spec** after a fix to the
insertion controller, not the scene: `results/full_task_v2_type_o_cable150_fix` (same workspace
sha256 `7263b394…`) seated 19.01 mm, 0.26 s hold, grasp drift 0.061 mm / 0.17°, peak wall 22.9 N,
peak leaf 50.1 N, final axial −12.9 N, cable 0.52 N. The low plug was caused by the probe's ±1 mm lateral
anti-wind-up bound. It counted the free-space alignment of the ~1 mm carry error, so the z bound
saturated within 5 steps and the upward correction was zeroed while nothing touched the plug.
The bound now counts from first socket contact (`PHYSICS_CHANGELOG.md`). One full-task run passes;
no multi-reset audit of the table-to-socket task exists yet in v2.

The held-plug expert (the one `collect_varied.py` uses) is validated against the same spec in
`results/openarm_v2_type_o_cable150`. It passes 9 of 10 checks:
- 20/20 aligned (4.38 s, seated 19.01 mm, peak wall 13.0 N, no retries, repeatable)
- 8/8 signed ±0.5/1 mm offsets and recovery after one retry
- half-timestep success and depth; parked arm, grasp slip (max 0.052 mm) and robot contact
`half_timestep_force` fails as in workspace_v1 (13.03 N vs 10.15 N, 0.2 N tolerance; seating
impact transient). The held-plug reset lays the 1.5 m cable without disturbing the grasp: slip
0.009 mm, cable load 0.33 N after settling.

The preserved failure before the fix, `results/full_task_v2_type_o_cable150`, aborted during
insertion:

```
wall 30.84 N, wall penetration 0.002 mm, leaf 16.1 N, depth 14.44 mm,
offset y/z 17 / −153 µm, axial −17.0 N, t = 42.795 s
```

The contact dump in that report shows all three pins bearing on the **upper** wall of their own
openings (contact normals ≈ (0, 0, −1) spread across `socket_w01`, `socket_w03`, `socket_w46` and
the matching lead-in hulls) while the plug sits 153 µm low. The load is a binding load, not a
seating load — the plug jams in Z rather than meeting the retention head-on. Raising
`wall_force_abort_n` from 20 to 30 N did not help and should not be raised further; the alignment
is the problem. The plug was low because of the controller bound described above. The same
signature (plug 150 µm low before any contact) is in the passing 1.0 m run's trace, so that
pass was marginal rather than evidence that the shorter cable fixes the problem.

Pickup and transport are unaffected: that same failed run records a clean pickup against the
committed spec, and `results/table_pickup_v2_longcable` passes all eight pickup cases (worst grasp
drift 0.063 mm) — though `validate_table_pickup.py` does not embed a workspace spec in its report,
so that run names no hash and is dated evidence for the 1.0 m layout only.

The connector and the retention do mate end to end when the cable is shorter:
`results/full_task_v2_type_o_longcable` passes with a **1.0 m / 40-segment** cable — seated
19.01 mm, 0.26 s hold, grasp drift 0.097 mm / 0.25°, peak wall 26.7 N, peak leaf 52.1 N, final
axial 13.8 N. That run's report embeds its own spec, which differs from the committed file in five
`cable.*` keys only (`length_m` 1.0, `segments` 40, explicit `rest_waypoints_world`
[[0.235, −0.22, 0.33], [0.22, −0.22, −0.42]], no `rest_segments`, third `rest_bulges_world` +Y), so
it can be reproduced by restoring those five values.

**Held-plug collection in v2** meets the same bar as workspace_v1: `validate_openarm.py` passes
9/10 against the committed spec (above), and the ten-reset preflight
`collect_varied.py --workspace configs/workspace_v2.json` passes 10/10 with no retries. Keep v1 and
v2 datasets separate (different scene, plug and manifest hashes). The table-to-socket task in v2
has one passing run only and is not a collector.

### What is not validated (workspace_v2)

- Everything in the workspace_v1 list above still applies.
- The cable length, mass and routing are simulation choices; no real appliance cord was measured.
- Only `type_o` has been run through the robot task at all. The other catalog plugs have leaf
  calibration only.
- For the table-to-socket task there is one passing full-task run against the committed spec
  (`full_task_v2_type_o_cable150_fix`), plus the preserved failure from before the fix. Cite it
  as a single demonstration; only the held-plug task has a multi-episode audit in v2.
