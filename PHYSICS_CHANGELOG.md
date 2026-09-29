# Physics changes

## box_v0 (preserved baseline)

The previous implementation lowered constant push from 1 N to 0.15 N after
excessive backstop impacts. Contact solref changed from `0.02 1` to `0.002 1`
after approximately 2 mm penetration at 0.15 N. Original scene, scripts, and
sweep are archived under `baselines/box_v0/`.

## two_blade_v1_straight

This is a new geometry/controller benchmark, not a one-variable comparison with
box_v0. Do not attribute changed outcomes solely to connector geometry.

- Geometry: one block/cavity becomes two 16 x 1.5 x 6 mm blades and separate
  2.3 x 6.8 mm straight slots. Housing is 28 x 24 x 16 mm; its front seats at
  the socket face. This models the constraint and grasp affordance in the user's
  reference. Equal blades allow 180-degree roll symmetry with exchanged slots.
- Mount height: mating origin is 100 mm above the table, preventing table-induced
  pitch/vertical correction. The slots have no rear stop at the blade seating
  depth. The fixture support begins beyond their 18 mm tunnels.
- Mass/inertia: total mass remains 30 g, with an explicit approximate COM and
  diagonal inertia for the new body. This is a design assumption, not measured
  hardware. Non-colliding strain relief does not affect the inertial model.
- Driving: constant free acceleration is replaced by a 10 mm/s position-target
  ramp, 300 N/m and 6 Ns/m translation gains, 0.12 Nm/rad and 0.002 Nms/rad
  rotation gains, 2 N/0.03 Nm wrench caps, and 1 mm target overtravel. These
  initial design settings make approach controlled and allow passive contact
  deflection while preserving commanded error. No gains were subsequently tuned.
- Gravity compensation: ideal compensation at COM supports the held plug without
  introducing a grasp-point gravity torque. It is separate from the capped
  spring wrench and must be replaced by the actual arm/grasp during integration.
- Contacts: inherited 0.5 ms timestep, Newton/100, implicitfast, solref `0.002 1`,
  solimp `0.9 0.95 0.001`, condim 3, and friction `0.4 0.005 0.0001` are unchanged.
- Success: both blades must fit their full straight slot corridors, reach 15 mm,
  satisfy the face-gap/orientation tolerances, and hold for 0.25 s. Maximum
  allowed penetration is 0.2 mm; force-abort threshold is 8 N. These are test
  limits, not measured safe hardware limits.
- Lead-ins, axial spring resistance, cable dynamics, and robot attachment are
  deferred so this straight-slot result remains an interpretable baseline.

Evidence is saved in `results/two_blade_v1_straight/`: each trial has model hashes,
configuration, MuJoCo version, summary, and timestamped traces. See
`validation.json` for timestep/repeatability checks and `sweep/` for signed cases.

## two_blade_v2_leadin — matched connector comparison

The first 1 mm of each slot is replaced by convex tapered pieces, expanding the
mouth 0.3 mm per side. Straight throat dimensions and holder/contact parameters
are unchanged. Geometry is composed by connector/geometry.py; original XML remains
the straight baseline. The fit test clips blade volume at the throat for the
lead-in variant. This avoids rejecting a valid root segment inside the wider mouth.

The 49-case matched comparison, 20 aligned repetitions and four half-timestep
checks passed. Both +/-0.5 mm lateral and vertical errors can now self-align;
larger-error failures remain. Evidence: results/lead_in_comparison/report.json.

## openarm_v1 — physical grasp and robot controller

Imported the official v1 bimanual model at commit
161039cd74ea8675fb8197836fe5674659825c75. Right arm is active, left arm is held at
home; both retain collision geometry. The plug remains free and is pinched by
the upstream v1 fingers. All insertion motion comes through robot actuators.

The following grasp changes were tested sequentially, not silently tuned together:

1. Inherited condim=3 produced about 77 degrees of plug spin during closure.
   Enabled condim=4 on the two active fingertip collision geoms, activating the
   existing torsional friction coefficient. Spin reduced to about 12 degrees.
2. Changed robot-scene friction cone from pyramidal to elliptic; closure drift
   reduced from about 7.2 mm to 4.5 mm but still failed the grasp test.
3. Raised impratio from 1 to 10 to stiffen friction constraints; closure drift
   reduced to about 2 mm. This is a numerical grasp setting, not a material fit.
4. Changed initial finger travel 16 -> 13.8 mm so the already-held reset begins
   near pad contact instead of letting the unsupported plug fall while closing.
   The closure target remains 8 mm; initial drift reduced to about 0.36 mm.
5. Changed active fingertip solref 5 -> 2 ms to reduce ongoing compliant creep.
   Completed insertion then had about 0.60 mm maximum displacement from the
   nominal grasp, compared with about 1.32 mm before this change.
6. Raised robot-scene impratio 10 -> 100 to reduce remaining friction creep.
   Insertion displacement reduced to about 0.23 mm. This setting was checked
   at half timestep along with the final controller. Grasp abort tolerance was
   tightened from 3 -> 1 mm after the stable grasp was demonstrated.

The standalone holder still uses its original cone/impedance, so these changes
do not rewrite the before/after connector comparison. Robot-scene forces cannot
be attributed to lead-ins alone: controller and grasp also differ.

Robot control uses explicit joint PD plus model bias compensation through the
upstream limited torque actuators. No plug gravity wrench is applied. Gains and
rates are recorded in configs/openarm_v1.json and require hardware calibration.

The initial expert jam guard missed a transient and let the plug slip. It now
uses the peak contact force within a command interval, starts withdrawal earlier,
and removes accumulated target error during withdrawal. A 1 mm deliberate probe
recovers in one retry; the faulty probe is validation-only, not BC demonstration
data. Final physical tests and limits are in results/openarm_v1/validation.json.
# Table pickup v3: rest, path and closure correction

- Both table-task arms now rest at official v1 zero joint angles, hands downward.
  The held-plug benchmark retains its original reset and dynamics.
- Replaced forward-start/direct reach with raise-behind-table and approach
  waypoints. Vertical pickup reference is 3 mm above the housing grasp site.
- Finger target changed from 8 mm to 6 mm per finger, closure speed 10 mm/s.
  An isolated vertical/8 mm trial still dropped the plug after closure-induced
  rotation; vertical/6 mm retained bilateral contact during lift/hold.
- No contact mesh, friction, mass, solver, or attachment change. Added substep
  unwanted-contact checks. Eight pickup trials pass, worst drift 0.472 mm;
  half-timestep trial passes. See results/table_pickup_v3/report.json.
- Pickup rotates the plug during closure. Transport/insertion must use the
  measured acquired grasp, not the prior insertion-only grasp assumption.

# workspace_v1: real table, physical cable, spring-leaf retention, pad plates

Layout, cable, leaf and gripper numbers live in `configs/workspace_v1.json` and are
described in `docs/WORKSPACE.md`. Every change below alters the robot-scene hash;
datasets and results recorded before it no longer replay strictly. The holder
benchmark (`connector/`, `assets/connector/*.xml`, `configs/holder.json`) is untouched:
a nominal `scripts/test_insert.py` run is column-for-column identical to the
pre-change code on the same machine (only new zero-valued diagnostic columns appear).

Changes were introduced one at a time; measured effects:

1. **Table 120x60x75 cm with legs, floor at z=-0.43, robot pedestal.** Table top stays
   0.32 m above the robot base, so plug/socket/waypoint coordinates are unchanged.
   Pickup rest height unchanged (plug z = 0.32800 m); no new robot contacts at rest.
2. **Physical cable** (0.35 m, 14 capsules with ball joints, 24.5 g, connect equality
   to an appliance box). Table pickup still passes (drift 0.04 mm, 0.27 deg). Held-plug
   reset showed a 12 deg plug rotation from the hanging cable with the upstream
   100 N/m finger servos, which motivated change 3.
3. **Right finger servo gain 100 -> 2000 N/m** (`finger_servo_kp_n_m`, robot config, in
   the manifest). Pinch ~11 N per pad (was ~0.5 N), pad penetration 0.19 mm; cable
   loads now hold within 0.08 mm / 0.4 deg.
4. **Finger pad plates** (24x16x1 mm boxes, 0.1 mm proud of the mesh face). The
   upstream mesh gave one contact point per finger, so any moment about the pinch
   line was carried by torsional friction only and the plug pitched 2.6 deg under
   the first 11 N of insertion load. With 4 box-box contacts per pad the same load
   pitches 0.1 deg.
5. **impratio 100 -> 1000.** A sustained 10 N axial load on the pinch crept 0.30 mm/s
   at 100 and 0.03 mm/s at 1000 (measured on the held plug). Grasp slip during
   insertion dropped from 1.0 mm (abort) to 0.15 mm at 10.7 N.
6. **Socket spring leaves** (one per slot on the outer wall, k=4000 N/m, 11.5 N
   preload, 6.8 deg ramp, armature 0.2 kg, geom/limit solref 1 ms). Calibrated in
   `scripts/calibrate_insertion_force.py`: -9.96 N insertion / +9.96 N withdrawal on
   the flat, leaf penetration 0.014 mm, wall force 0 (`results/insertion_force_v1`).
   Design notes: a 14 deg ramp needed ~17 N to start the wedge (1.7x the plateau) and
   the wound-up arm then slammed the plug into the socket end; a static socket is
   welded to the world so `<exclude>` pairs are required to stop the wall from
   colliding with the embedded leaf; leaf inertia (armature) is what makes MuJoCo's
   soft joint limit hold the preload (0.4 mm sag at 0.02 kg, 3 um at 0.2 kg).
7. **Abort semantics with retention.** `contact_force_n` remains rigid-wall force only;
   leaf load is reported separately (`leaf_normal_n`, `leaf_friction_n`, ...). The
   rigid-socket 5-8 N wall abort would reject every successful seating with retention
   (the housing face bottoms out under the leftover push force and the walls guide the
   blades while the compliant arm droops), so with leaves enabled the wall abort is
   `wall_force_abort_n` = 20 N; wall penetration stays limited to 0.2 mm, leaf force to
   80 N and leaf penetration to 0.3 mm.
8. **Scripted insertion controller** (not physics): lateral corrections run 2x the
   axial advance with a +/-1 mm anti-wind-up bound, and the target advances only while
   the measured axial force is below `push_force_cap_n` = 15 N. Without the bound the
   loop integrated while friction held the plug and released the wind-up as a 20 N
   slam into the top slot wall.

Full task outcome with the default 10 N retention: pickup -> carry -> insert -> hold
passes (`results/full_task_cable_v1`, seated 15.98 mm, 0.26 s hold, grasp drift
0.19 mm / 0.34 deg, peak wall 17 N, final push 13.6 N); the 5 N variant also passes
(`results/full_task_cable_v1_5n`). The socket is no longer rotated 90 deg for the
demonstrator: with the firm grasp the plug stays flat and pinched on its 24 mm faces,
so the identity socket orientation from the workspace spec is reachable.

Not passing: the held-plug expert (`controllers/expert.py`, `validate_openarm.py`)
was tuned for the rigid socket and times out against the 10 N retention: all 20
aligned episodes end at 0.5 mm depth after its two 0.3 N-triggered withdrawals,
peak wall force 6.4 N (`results/openarm_cable_v1`, preserved failed audit). Eight table pickup cases pass
with the cable attached (`results/table_pickup_cable_v1`, worst drift 0.054 mm).

# workspace_v2: Thai Type O connector, three-leaf retention, cable to the floor

Geometry lives in `configs/workspace_v2.json` and `connector/catalog.py`, described in
`docs/CONNECTOR_CATALOG.md` and the workspace_v2 section of `docs/WORKSPACE.md`.
workspace_v1 is untouched and remains the scene the full task passes in. **The full
task does not pass in workspace_v2**; the changes below are recorded with what each one
measured, including the one that failed.

1. **Catalog connector: `type_o` in `universal_th`** (Thai 3-round-pin plug, 45 g,
   Ø4.8 mm pins, 21.4 mm earth, in a universal socket with keyhole line/neutral openings).
   Forced two layout changes: the socket moves 6.4 mm further from the robot so the home
   pose clears the longer earth pin (pre-insert distance 27.4 mm, was 22 mm) and 6 mm
   lower because an earthed plug's grasp frame sits 6 mm above its mating frame. Finger
   travel follows the 34 mm housing (contact 18.8 mm, grip 13.0 mm) instead of the 24 mm
   one. The socket wall/lead-in geometry is generated, not hand-written.
2. **Three retention leaves instead of two**, one per opening the plug uses. The earth
   leaf presses −Z and the line/neutral leaves 30° above the inward horizontal, so the
   three equal forces and their torques about X cancel and a seated plug is not pushed
   against one hole wall. Normal force per leaf drops 12.5 → 8.33 N so the total stays at
   the 10 N target. Calibrated robot-free per plug type: Type O −10.82 N insert /
   +8.86 N withdraw on the flat, Type A −9.96/+9.96, Type B −10.45/+11.23, Type C
   −9.38/+8.58 (`results/insertion_force_v2_type_*_ws2`; full table in
   `docs/CONNECTOR_CATALOG.md`). Type B is the only one that also loads the rigid wall
   (9.18 N) because its blades and its earth pin use different opening kinds.
3. **Cable mass 70 → 35 g/m.** With the cable routed over the table edge and 0.75 m
   hanging to the floor, the heavier cord dragged the 45 g plug across the table: ≈0.5 N
   of pull against ≈0.2 N of table friction. At 35 g/m the edge friction leaves ≈0.1 N on
   the plug and the spawn pose holds.
4. **Cable route, three attempts, one change each.**
   - 0.35 m / 14 segments on the table (the v1 cable with the v2 connector): full task
     passes, seated 19.01 mm (`results/full_task_v2_type_o`).
   - 1.0 m / 40 segments over the near edge to the floor: full task passes, seated
     19.01 mm, 0.26 s hold, grasp drift 0.097 mm / 0.25°, peak wall 26.7 N, peak leaf
     52.1 N (`results/full_task_v2_type_o_longcable`).
   - 1.5 m / 60 segments to an appliance box standing on the floor: **full task aborts**
     at wall 30.84 N, depth 14.44 mm, plug 153 µm low
     (`results/full_task_v2_type_o_cable150`, preserved failed audit). This is the layout
     in the committed spec.
5. **`wall_force_abort_n` 20 → 30 N — and it was not enough.** The 1.5 m route still
   aborts at 30.84 N. The contact dump shows all three pins bearing on the *upper* wall
   of their openings while the plug sits low, i.e. the plug binds in Z rather than
   meeting the retention axially. Raising the limit again would hide a misalignment, so
   the limit stays at 30 N and the failure stays open. Why this cable route pulls the
   plug low is not yet diagnosed.

Pickup and transport are unaffected by any of the above: the failed insertion run still
records a clean pickup against the committed spec, and eight pickup cases pass with the
1.0 m cable (`results/table_pickup_v2_longcable`, worst drift 0.063 mm; that report
embeds no workspace spec, so it names no hash). No v2 dataset may be collected until a cable layout passes end to
end against a committed spec.

# Insertion anti-wind-up counted from contact (controller, not physics)

This diagnoses workspace_v2 item 5. I rebuilt `lateral_bias` in `scripts/probe_table_insert.py` from
the saved traces. In both v2 runs (1.5 m abort and 1.0 m pass) the approach starts about 1.0–1.3 mm
low, and the ±1 mm z bound saturated within **5 steps of free-space approach**. The upward
correction was zeroed from then on, and the plug crossed the mouth 150–380 µm low with zero socket
force. In v1 the bias peaked around 0.7 mm and the offset converged to about 0. The 1.0 m pass
shows the same low-plug signature, so it was marginal. The cable only changed how hard the
binding hit.

One change: the bound (and bias accumulation) starts at first socket contact (axial, wall or leaf
force > 0.1 N), with the bias reset to 0 there; the bias is now logged as `lateral_bias_m`. The
file is not in `env.manifest()`.

- `results/full_task_v2_type_o_cable150_fix` (committed workspace_v2, sha256 `7263b394…`):
  **passes**. Offset about 0 µm before contact, seated 19.01 mm, 0.26 s hold, peak wall 22.9 N
  (was 30.84 N abort), peak leaf 50.1 N, grasp drift 0.061 mm / 0.17°. Under the leaf load the
  bias still saturates in contact and the plug droops about 140 µm, within the opening clearance.
- `results/full_task_cable_v1_fix` (workspace_v1 regression): passes, 15.99 mm, peak wall 18.4 N,
  peak leaf 72.2 N (previously 15.98 mm / 17.4 N / 72.2 N).
- `results/full_task_v2_type_o_cable150_fix_halfdt` (0.25 ms): passes, seated 19.01 mm, peak wall 26.6 N, peak leaf 48.0 N,
  grasp drift 0.051 mm. The peak wall is within the 30 N limit at both steps but differs by 3.7 N,
  so its exact value is step-sensitive.

# Held-plug expert with spring-leaf retention (controller, not physics)

`controllers/expert.py` is in `env.manifest()`, so this change alters every manifest source hash;
datasets recorded before it do not replay strictly (none recorded since workspace_v1 did either).

Diagnosis of `results/openarm_cable_v1` (one aligned workspace_v1 episode, traced per command): the
approach is clean (offset < 5 um, no wall contact) until the leaves engage at 9.9 mm. Under the
~13 N leaf load the compliant arm droops ~0.37 mm in Z, the blades graze a slot wall, and the
rigid-socket jam guard (`interval_peak_contact_force_n` > 0.3 N) withdraws. The leaves hold the
plug at 12.5 mm, so the withdrawal cannot extract it; the re-align pulls it back out and the second
retry repeats this until the 8 s limit.

One change: once leaf force exceeds 0.1 N during an attempt (`engaged`), the jam retry is disabled
and the 8 mm/s target advance pauses while the axial socket load is at or above
`push_force_cap_n` (15 N, the probe's cap). Before engagement the expert is unchanged, so the
mouth-jam retry still handles the deliberate 1 mm probe fault.

`results/openarm_cable_v2`: 20/20 aligned (3.46 s, 16.0 mm, peak wall 10.17 N, no retries,
repeatable), 8/8 signed +/-0.5/1 mm offsets, recovery succeeds after one retry, half timestep
succeeds at the same depth; parked arm, grasp slip (max 0.12 mm) and robot contact pass. The fixed
ten-reset collection preflight passes 10/10 (`scripts/collect_varied.py` without `--record`).

Not passing: `half_timestep_force`. The peak wall force is 10.17 N at 0.5 ms and 8.50 N at
0.25 ms (tolerance 0.2 N). The peak is the housing reaching the socket face under the retention
push (axial -24 N vs -12 N at that instant), an impact transient whose size depends on the step.
The rigid-socket peaks were 1.69/1.67 N. A slower 4 mm/s push after engagement was tried and made
it worse (12.74 vs 6.94 N), so it was reverted. The tolerance was not loosened; this check stays open.
