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
