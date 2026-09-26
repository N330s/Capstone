# Workspace specification (workspace_v1)

Source of truth: [`configs/workspace_v1.json`](../configs/workspace_v1.json). `envs/scene.py`
generates the floor, table, pedestal, fixture, appliance, cable, socket spring leaves, finger pads
and cameras from that file through `envs/workspace.py`; nothing below is typed twice in code. Its
SHA-256 is part of `env.manifest()["workspace"]` and of every full-task report, so a run always
names the exact layout it was produced with. Edit the JSON, not this page, and keep both in step.

All dimensions are **design values chosen for the simulation**. None is a measurement of real
hardware; the sim-to-real caveats in `docs/SIM2REAL_CONTRACT.md` apply to every row.

## Frames

- World origin = OpenArm v1 base body (`openarm_body_link0`). +X runs across the table away from
  the robot, +Y along the table, +Z up.
- The table top is **0.32 m above the robot base**. This is the same relationship as the earlier
  floating-slab scene, so every plug / socket / waypoint coordinate in `configs/table_task_v*.json`
  and the controllers is unchanged. The floor was moved down instead (z = −0.43 m).
- Socket frame: +X insertion, Y separates the blades, Z is the blade width (see README).

## Top view (not to scale)

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

## Table, floor, pedestal

| Item | Size (L×W×H) | Position / extent | Notes |
| --- | --- | --- | --- |
| Table top | 120 × 60 × 3 cm | centre (0.55, 0, 0.305) m; x 0.25…0.85, y ±0.60, top z = 0.32 | geom `work_table` (name kept for older scripts) |
| Legs (4) | 5 × 5 × 72 cm | inset 3 cm from each edge; centres x ∈ {0.305, 0.795}, y ∈ {±0.545}; z −0.43…0.29 | `table_leg_{near,far}_{left,right}` |
| Table height | 75 cm | floor z = −0.43 → top z = +0.32 | `frames.floor_z_m` must equal top − height (validated) |
| Floor | 6 × 6 m plane | z = −0.43 | |
| Robot pedestal | 20 × 26 × 37 cm | centre (−0.052, 0.032); z −0.43…−0.06 | column under the upstream base meshes (they reach z ≈ −0.06) |
| Fixture post | 1.2 × 4.8 × 15.8 cm | centre (0.4631, −0.155, 0.399); z 0.32…0.478 | unchanged from earlier scenes |

## Socket and retention leaves

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

## Plug and cable

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

## Gripper

| Item | Value |
| --- | --- |
| Finger servos | position servos, kp raised 100 → 2000 N/m (`configs/openarm_v1.json: finger_servo_kp_n_m`) → ≈ 11 N per pad at the held grasp |
| Pad plates | 24 × 16 × 1 mm box on each right finger, 0.1 mm proud of the upstream mesh face (finger-local y = ±7.8 mm, z 56…80 mm) |
| Pad contact | friction (1.0, 0.01, 0.01), condim 4, solref 2 ms, priority 2 → 4 contact points per pad on the housing |

## Cameras

| Camera | Parent | Position | Aim | FOV |
| --- | --- | --- | --- | --- |
| `scene_rgb` | world | (0.85, −0.90, 0.85) | xyaxes (0.8, 0.6, 0, −0.25, 0.333, 0.91) | 48° |
| `inspection` | world | (0.56, −0.32, 0.59) | xyaxes (0.75, 0.66, 0, −0.35, 0.4, 0.847) | 45° |
| `wrist_rgb` | `openarm_right_hand` | hand-local (0.05, −0.04, 0.01) | hand-local (0, 0.006, 0.123) | 55° |

The wrist camera carries an action-camera style visual body: 62 × 45 × 32 mm housing 18 mm behind
the optical centre, Ø 18 × 8 mm lens, and a Ø 10 mm bracket to hand-local (0.02, −0.02, 0). All
three geoms are visual only (no mass, no collision). `wrist_rgb` intrinsics are unchanged.

## Solver options (robot scene only)

2 kHz, `implicitfast`, Newton 100 iterations, elliptic cone, **impratio 1000** (was 100). The
holder benchmark keeps its own original settings so the straight-vs-lead-in comparison stays matched.

## Model bookkeeping

- `nq = 25 + 4·14 + 2 = 83`, `nv = 24 + 3·14 + 2 = 68` (arm/fingers/plug + cable balls + leaf slides).
- Bodies `cable_seg_00…13`, joints `cable_j_00…13`, sites `cable_root` (force/torque sensors),
  `cable_end`, `cable_anchor`; bodies `socket_leaf_left/right` with joints `*_slide`; contact
  excludes `socket_leaf_*_wall` (the static socket is welded to the world, so the parent-child
  filter would not stop the wall from colliding with the embedded leaf).
- Policy observations are unchanged: RGB + 16-d arm/finger state + instruction. Cable and leaf
  states are physics only; they appear in `info` (privileged) and in recorded `qpos/qvel`.

## What is not validated

- Cable stiffness, mass and friction are plausible cord values, not measurements.
- The 10 N retention target and its calibration are simulation-internal; no real socket was measured.
- The finger pad plates and the 2000 N/m servo gain are simulation choices that make the grasp
  capable of reacting cable and retention loads; real gripper force, pad compliance and slip are
  open sim-to-real items.
- Datasets and results produced before workspace_v1 no longer replay strictly (scene hash changed).
