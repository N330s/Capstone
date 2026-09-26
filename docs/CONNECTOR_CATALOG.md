# Connector catalog (parametric plugs and sockets)

Source of truth: [`connector/spec.py`](../connector/spec.py) (the data model) and
[`connector/catalog.py`](../connector/catalog.py) (the named variants). A scene asks for a variant
by name — `configs/workspace_v2.json: connector`, or `--plug-type` / `--socket` on the holder and
probe scripts — and the geometry is generated from that description. Nothing below is typed twice
in code: the spec is embedded in every compiled model and read back with
`ConnectorSpec.from_model`.

All dimensions are **nominal design values**: they follow the published TIS 166-2549 (Thai Type O),
CEE 7/16 (Europlug, Type C) and NEMA 1-15 / 5-15 (Type A / B) figures, or they are simulation
choices. **None is a measurement of real hardware** and no real plug or socket was measured. The
caveats in [`docs/SIM2REAL_CONTRACT.md`](SIM2REAL_CONTRACT.md) apply to every row on this page.

## Why it exists

The hand-written `assets/connector/*.xml` describes exactly one connector: a two-blade plug in a
two-slot socket. It cannot express a three-round-pin Thai plug, and a socket face with holes in it
is concave, which MuJoCo has no collision primitive for. Rather than hand-write a second set of
assets, the geometry is generated from a description.

The catalog entry `legacy_two_blade` reproduces the hand-written assets exactly, so
`connector_tree()` without a spec, the holder benchmark and every pre-existing result are
unaffected. `configs/workspace_v1.json` has no `connector` section and therefore uses it.

## Pipeline

| Module | Role |
| --- | --- |
| `connector/spec.py` | Pure data (`PlugSpec`, `SocketSpec`, `PinSpec`, `OpeningSpec`, `Part`) plus `derived()`. No mujoco import. |
| `connector/slab.py` | 2-D decomposition of the socket face into convex pieces. No mujoco import. |
| `connector/builder.py` | Turns a spec into MuJoCo bodies, assets and the embedded spec text. |
| `connector/catalog.py` | Named variants and the `--plug-type` / `--socket` CLI wiring. |
| `connector/geometry.py` | Picks the legacy tree or the generated one (`spec=`). |

`builder.py` writes every rotation as a quaternion so the same tree compiles under the holder
scene's `angle="degree"` and the robot scene's `angle="radian"`. It embeds the spec as
`<custom><text name="connector_spec">`, so `ConnectorMetrics`, the retention leaves,
`OpenArmInsertEnv` and the probes all read the geometry back out of the compiled model instead of
copying constants. A model without that text maps onto the legacy catalog entry.

### Slab decomposition

MuJoCo has no concave collision primitive, so the socket block — the face rectangle minus the
openings, extruded along +X — is cut into horizontal Z-strips at every opening vertex and at every
crossing between opening edges. Inside a strip no boundary crosses, so each opening section and
each solid gap between openings is a trapezoid; extruded along X it is a box (vertical edges) or an
8-vertex convex hull. The 1 mm lead-in reuses the same strips with the openings dilated 0.3 mm at
the mouth, giving one tapered hull per solid piece, exactly like the hand-written `*_lead` hulls.
Round openings are approximated as 16-sided polygons (`polygon_sides`).

## Plugs

Shared constants (`catalog.py`): blade pitch 12.7 mm, round-pin pitch 19 mm, earth pin 11.89 mm
above the line/neutral axis (the NEMA 5-15 ground position, shared by Type O and B).
`grasp_offset_m` is the mating frame relative to the grasp frame, −16 mm on X for every variant.

| Plug | Pins | Housing (depth × width × height) | Centre z | Mass | Symmetry rolls |
| --- | --- | --- | --- | --- | --- |
| `legacy_two_blade` | 2 blades 16 × 6 × 1.5 mm at y ±6.5 mm | 28 × 24 × 16 mm | 0 | 30 g | 0°, 180° |
| `type_a` (NEMA 1-15) | 2 blades 15.9 × 6.35 × 1.5 mm at y ±6.35 mm | 30 × 34 × 16 mm | 0 | 35 g | 0°, 180° |
| `type_b` (NEMA 5-15) | 2 blades as Type A + Ø4.8 × 19 mm earth | 30 × 34 × 20 mm | +6 mm | 45 g | 0° only |
| `type_c` (CEE 7/16) | 2 round Ø4 × 19 mm at y ±9.5 mm | 30 × 34 × 16 mm | 0 | 35 g | 0°, 180° |
| `type_o` (TIS 166-2549) | 2 round Ø4.8 × 19 mm sleeved 10 mm + Ø4.8 × 21.4 mm earth | 30 × 34 × 20 mm | +6 mm | 45 g | 0° only |

The width column is the pinch axis: the fingers close on the 34 mm faces of the catalog plugs and
the 24 mm faces of the legacy one, which is why finger travel is derived from the housing width
(`envs/workspace.finger_travel_for_width`) rather than hard-coded.

An **earthed plug is polarised**: `symmetry_rolls_deg = (0,)`, so the success oracle no longer
accepts a half-turn. The two-pin plugs keep `(0, 180)`.

## The `universal_th` socket

One socket accepts all four plugs, like a real Thai universal outlet.

| Item | Value |
| --- | --- |
| Face | 40 × 40 mm, z −14…+26 mm (offset up to cover the earth hole); depth 23 mm |
| Line / neutral openings | keyhole: 1.9 × 6.9 mm slot at y ±6.35 mm **plus** Ø5.1 mm hole at y ±9.5 mm |
| Earth opening | Ø5.1 mm hole at z +11.89 mm |
| Clearances | holes 0.15 mm radial (4.8 mm pin); slots 0.2 mm per side on thickness, 0.275 mm on width |
| Lead-in | 1 mm deep, openings dilated 0.3 mm at the mouth |
| Legacy socket, for comparison | 40 × 30 mm face, 18 mm deep, two 2.3 × 6.8 mm slots 13 mm apart |

### Leaf axes

Each opening carries a `leaf_axis_yz`, the direction its retention leaf presses. The earth leaf
presses straight down (−Z); the line/neutral leaves press 30° above the inward horizontal
(∓cos30, +sin30). The three equal leaf forces and their torques about X therefore sum to zero, like
paired contacts in a real socket, so a seated plug is not pushed against one hole wall. Blades in a
keyhole are pressed outward from the inner web instead.

## Derived distances

`ConnectorSpec.derived()` replaces the constants the controllers and probes used to hard-code.
Values in mm:

| Plug | `preinsert_x` | `retreat_x` | `probe_preplug_x` | `success_depth` | `flat_window_top` | spawn above table |
| --- | --- | --- | --- | --- | --- | --- |
| `legacy_two_blade` | −22.0 | −18.0 | −25.0 | 15.0 | 15.5 | 10.0 |
| `type_a` | −21.9 | −17.9 | −24.9 | 14.9 | 15.4 | 10.0 |
| `type_b` | −25.0 | −21.0 | −28.0 | 14.9 | 15.4 | 6.0 |
| `type_c` | −25.0 | −21.0 | −28.0 | 18.0 | 18.5 | 10.0 |
| `type_o` | −27.4 | −23.4 | −30.4 | 18.0 | 18.5 | 6.0 |

`preinsert` and `retreat` follow the **longest** pin (`controllers/expert.py` replaced its −22 /
−18 mm literals with them); `success_depth` follows the **shortest**, so a plug counts as seated
only when every element is in. `cable_attach_local_m` puts the cable root at the strain relief
(housing depth + 12 mm behind the mating frame, at the housing centre height), and
`spawn_height_above_table_m` gives the table rest pose. An earthed plug's grasp frame sits 6 mm
above its mating frame, which is why the v2 socket is 6 mm lower than the v1 one.

## Generalised success oracle

`ConnectorMetrics` (`connector/simulation.py`) reads the spec and tests each element against the
opening it is assigned:

- **Blades** keep the original test: all eight corners of the collision box, clipped at the lead-in
  throat, must lie inside the slot corridor.
- **Round pins** use a radial test: the capsule axis from the housing face to the tip-sphere centre
  must stay within (hole radius − pin radius) of the hole axis, which bounds the whole surface.
- **Assignment** is computed once per allowed symmetry roll at `bind()` time; a pin with no opening
  of its kind within 1 mm is an error rather than a silent mismatch.

## Per-plug retention calibration

Robot-free harness (`scripts/calibrate_insertion_force.py`), mocap-driven plug, 5 mm/s, against the
committed `configs/workspace_v2.json` (leaf design values are in
[`docs/WORKSPACE.md`](WORKSPACE.md)). Target is 10 N insertion resistance.

| Plug | Leaves | Insert (flat) | Withdraw (flat) | Peak axial | Peak leaf normal | Leaf penetration | Peak wall | Depth | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `type_o` | 3 × 8.33 N | −10.82 N | +8.86 N | 18.45 N | 47.3 N | 0.013 mm | 0.14 N | 18.99 mm | `results/insertion_force_v2_type_o_ws2` |
| `type_a` | 2 × 12.5 N | −9.96 N | +9.96 N | 20.03 N | 50.1 N | 0.014 mm | 0 | 15.86 mm | `results/insertion_force_v2_type_a_ws2` |
| `type_b` | 3 × 8.33 N | −10.45 N | +11.23 N | 18.66 N | 43.8 N | 0.010 mm | 9.18 N | 15.87 mm | `results/insertion_force_v2_type_b_ws2` |
| `type_c` | 2 × 12.5 N | −9.38 N | +8.58 N | 13.44 N | 27.9 N | 0.018 mm | 0 | 18.96 mm | `results/insertion_force_v2_type_c_ws2` |

All four seat and hold a valid pose for the whole hold window with zero solver warnings. Notes:

- The leaf normal force is shared between however many leaves the plug engages, so the two-pin
  plugs get 12.5 N per leaf and the earthed ones 8.33 N; insertion resistance is n · μ · N with
  μ = 0.4 either way.
- `type_c` has the loosest fit (0.55 mm clearance against 0.10 mm interference: a 4 mm pin in a
  5.1 mm hole), which is why its plateau is the lowest and its leaf barely deflects (0.09 mm
  against 0.45–0.57 mm for the others).
- `type_b` is the only plug that also loads the rigid wall (9.18 N, 0.024 mm penetration): its
  blades sit in the slot half of the keyhole while its earth pin is in a round hole, so the two
  opening kinds fight slightly over the axis. It is the least clean of the four.
- The peak axial force is the ramp entry, roughly twice the plateau; the flat window is the seated
  region quoted in the table.
- These four runs supersede the earlier `results/insertion_force_v2_type_*` directories, which were
  produced against an older workspace spec. Type A, C and O reproduce bit for bit; type B moved
  (−9.87 → −10.45 N insert) because `connector/catalog.py` was edited after its original run.

## What is not validated

- Every dimension is a nominal standard or simulation value, never a measurement.
- The clearances are design choices; real sockets vary and are not modelled per manufacturer.
- The 10 N retention target and its per-plug calibration are simulation-internal. No real
  insertion force was measured.
- Only `type_o` has been driven by the robot through a full task, and only with a shorter cable
  than the committed workspace_v2 spec: see the workspace_v2 status in
  [`docs/WORKSPACE.md`](WORKSPACE.md). `type_a`, `type_b` and `type_c` exist as calibrated
  geometry only.
- Pin sleeves, bezels, housing bevels and cable boots are visual only (no mass, no collision).
