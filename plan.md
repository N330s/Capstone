# OpenArm Plug Insertion: Demonstrations, VLA Fine-Tuning, and RLT

Updated 2026-09-13 after continuous table-to-insertion validation and command replay.

## Goal and scope

Final user scenario: **plug resting on table -> grasp -> lift/transport -> insert**,
using the right arm of the bimanual OpenArm v1. Held-plug insertion is a component
curriculum, not the final task. Reliable physical pickup now precedes scaling
full-task demonstrations. See [TABLE_TO_INSERT.md](docs/TABLE_TO_INSERT.md).

Build a simplified two-flat-blade plug and matching socket inspired by the supplied
photo. The scene should support meaningful precision contact, a practical robot
grasp, repeatable demonstrations, and later vision-language-action (VLA)
fine-tuning followed by reinforcement learning of the insertion phase.

The aim is a mechanically useful training task, not a certified electrical
connector replica. Dimensions below are proposed simulation design values,
not measurements from the photo or claims of compliance with a connector standard.
Use an unpowered fixture for any later physical experiments.

Implementation status: the robot-and-pilot milestone is complete. The scene uses
the official **OpenArm v1 bimanual configuration**, with the **right arm active
and left arm parked**. Both arms remain in the model with collision geometry.
The plug is physically grasped, not welded or moved by an external holder.

| Completed work | Evidence and limits |
| --- | --- |
| Two-blade connector and lead-ins | 49 matched straight/lead-in cases, aligned repeats, and half-timestep checks; improved capture with a remaining failure region. `results/lead_in_comparison/report.json` |
| Robot insertion and recovery | 20 repeatable aligned successes, eight signed offset corrections, forced-jam recovery, and half-timestep agreement. `results/openarm_v1/validation.json` |
| Shared control/observation interface | 50 Hz; eight right-arm joint/gripper targets; 16-element bimanual state, fixed/wrist RGB, and instruction. Privileged diagnostics remain separate. |
| Regression tests | 27 tests passed, including downward rest and scratch-only carried-payload planning. |
| Pilot demonstrations | 20 successful episodes, 3,440 command transitions, split into 16 training and four validation episodes. `data/openarm_v1_pilot/manifest.json` |
| Command replay | All 20 episodes reproduce recorded qpos and qvel exactly in the current pinned simulator configuration. `data/openarm_v1_pilot/replay_report.json` |
| Framework-neutral VLA data adapter | Train-only normalization, disjoint episode splits, action round-trip, and terminal chunk masking pass. `data/openarm_v1_pilot/training_readiness.json` |
| pi0-specific local adapter | Official source pinned; explicit state/action/camera mapping, single-owner normalization, and full-window sampling implemented. 1,968 train / 492 validation windows. `docs/VLA_ADAPTER.md` |
| Float32 transport audit | All 20 episodes pass action conversion; one replay succeeds but fails the strict plug velocity tolerance. Retain `results/vla_adapter_v1/report.json` with `passed=false`; no physics/tolerance change. |
| First varied batch | Ten successful episodes with exact command replay; 1,746 transitions, 8/2 train-validation split, distance/lateral/angular variation. 100 evaluation resets reserved. No recovery episodes. `docs/DATA_COLLECTION.md` |
| Continuous table-to-insertion | Six physical cases pass with the socket rotated 90 degrees about its unchanged insertion axis/location; nominal command replay is exact. Full-task synchronized RGB training data is not yet collected. `docs/FULL_TASK_DEMONSTRATOR.md` |

The pilot is a narrow near-nominal integration dataset, not evidence of learned
policy competence or generalization. No VLA has been fine-tuned, no RLT module has
been trained, and no physical transfer has been validated. Checkpoint-specific
LeRobot export, upstream preprocessing runtime checks, and model inference remain
outstanding. The local adapter is not a complete training integration. See
[VLA_ADAPTER.md](docs/VLA_ADAPTER.md) for the precise implemented boundary and the
retained float32 contact-sensitivity diagnostic.

See [OPENARM_PIPELINE.md](docs/OPENARM_PIPELINE.md) for commands and artifacts,
[SIM2REAL_CONTRACT.md](docs/SIM2REAL_CONTRACT.md) for calibration gaps, and
`PHYSICS_CHANGELOG.md` for contact-model changes. Preserve the straight-slot
baseline in `results/two_blade_v1_straight/` and box prototype in `baselines/box_v0/`.

## Current baseline and remaining geometry work

The archived box_v0 scene used one rectangular block and one broad cavity.
The active scene already uses the two-blade geometry described below.

The new scene has:

- An insulating housing with clear parallel grasp faces.
- Two rigid metal blades protruding from the housing.
- Two separate socket slots, including the solid separator between them.
- A socket face that stops the housing from entering.
- A 1 mm lead-in with 0.3 mm mouth expansion per side in the robot scene;
  the original straight-slot assets remain available for comparisons.
- A rear strain-relief stub; cable dynamics are deferred.
- A mounted socket high enough for the plug and gripper to approach without
  touching the table.

The old table-alignment, roll-detection, and contact-filtering limitations have
been addressed in the active holder benchmark. Existing evidence includes 20
identical aligned successes, 11 regression tests, a 41-case signed sweep, and six
representative half-timestep comparisons. These results validate the configured
holder baseline, not an OpenArm grasp or a learned policy.

The holder CSV traces remain physics diagnostics, not VLA demonstrations.
The separate robot pilot now contains synchronized cameras, state, commands,
episode boundaries, and source/configuration provenance.

## Task progression

```text
validated connector and lead-in comparison
    -> one OpenArm holding the plug, controlled through robot actuators
    -> common observation/action API and replayable pilot demonstrations
    -> supervised VLA fine-tuning and held-out rollouts
    -> learned RL-token representation and frozen inference interface
    -> RLT online learning in simulation on the insertion phase
    -> end-to-end approach/handoff/insertion evaluation
    -> optional physical calibration, pickup, and multitask expansion
```

The held-plug insertion baseline is implemented. Next develop table pickup as a
separate physical gate, then compose pickup and insertion without resets between
them. A stable carried grasp, its measured transform, and a collision-free
transport path are required before collecting full-task SmolVLA demonstrations.

## Phase 1 — Connector geometry and coordinate contract

### Frames

SI units internally; millimetres and degrees may be used in CLI inputs and plots.

Define the socket frame at the midpoint between the two slot entrances:

- +X points into the socket.
- Y separates the two blades.
- Z runs along the broad dimension of each blade.
- Roll, pitch, and yaw rotate about X, Y, and Z respectively.

Define the plug mating frame on the housing front face, midway between the blade
roots, with axes matching the socket at the intended mating orientation.
The blade tips therefore lie ahead of this frame along +X.

Use socket-relative transforms for geometry checks and resets. Rotating or moving
the socket in the world must not change the meaning of depth or alignment.

### Proposed starting dimensions

| Part | Proposed dimensions | Role |
| --- | --- | --- |
| Housing | 28 mm along X, 24 mm along Y, 16 mm along Z | Graspable rigid body |
| Two blades | Each 16 mm along X, 1.5 mm along Y, 6 mm along Z | Actual insertion collision geometry |
| Blade centers | Y = ±6.5 mm | 13 mm center separation |
| Straight slots | Each 2.3 mm along Y, 6.8 mm along Z | 0.4 mm clearance per side |
| Slot tunnel | 18 mm deep from socket face | Blade clearance behind the seated tip |
| Slot lead-in | 1 mm axial length; mouth expanded 0.3 mm per side | Small alignment capture region |
| Strain relief | Short rear cylinder or capsule | Recognizable silhouette, outside grasp region |

Treat clearance as a named parameter. Validate these loose starting values before
tightening. Specify chamfers by axial length and mouth expansion so their geometry
is unambiguous.

Make the housing, blades, and strain relief one rigid body with one free joint.
Use an explicit, documented total mass and inertia; an initial total mass around
30 g is a design assumption to validate or replace with measurements. Do not let
overlapping decorative geoms accidentally add mass.

Start with two equal blades and record the resulting 180-degree mating symmetry.
If a physical polarized connector is selected later, introduce its measured
asymmetry in both plug and socket and update orientation/success logic. Do not
infer precise blade asymmetry from the photograph.

### Collision construction

Use boxes for blades and housing. Build the socket face and tunnels from convex
pieces around two actual empty slots. Preserve the central separator: replacing
the two slots with one common cavity removes the intended constraint.

Use small convex wedges or rotated boxes for lead-ins. Keep the housing face stop
outside the openings and make it the intended seating limit. Blade tips should
not strike an artificial rear wall before the housing seats.

Omit blade holes, molded lettering, fine ribs, and electrical spring detail.
They do not need collision geometry for this stage. Rounded visual housing pieces
may be added without altering the validated grasp/collision envelope.

Rendering geoms must have collision disabled and must not change mass/inertia.
Keep collision geometry available as a viewer overlay.

### Robot-facing sites

Provide stable names for:

- `plug_mating_frame`, `plug_tip_left`, `plug_tip_right`.
- `plug_grasp_frame`: centered between accessible parallel housing faces.
- `socket_entry`, `slot_left_entry`, `slot_right_entry`.
- `socket_seated_frame`: desired plug mating-frame pose when seated.
- `socket_preinsert_frame`: configurable approach pose along negative socket X.

Select the final grasp axis and jaw contact patches against the actual OpenArm
gripper dimensions. Check jaw opening, finger reach, blade visibility, and socket
face clearance before freezing the housing dimensions. Keep fingers behind the
housing front face throughout seating.

### Gate 1

Inspect collision and rendered views. Verify two distinct open slots, dimensions,
mass/inertia, unobstructed grasp faces, and a collision-free aligned path up to
housing seating. No robot or learning is required for this gate.

## Phase 2 — A standalone test that represents a held plug

Keep a free-body dynamics smoke test, but use a force/torque-driven compliant
holder as the main insertion benchmark.

Mount the socket above the table with enough clearance that neither the housing
nor tilted blades touch the table during reset or approach. Reject initial
interpenetration instead of letting the solver eject the plug into a different
starting pose.

The holder applies a bounded Cartesian spring-damper wrench about a commanded
grasp pose. Log translational/rotational stiffness, damping, wrench limits, and
motion speed. This approximates a robot holding the housing while retaining a
dynamic plug and physical contact.

Use a slow axial target ramp or bounded axial force plus damping. Do not overwrite
plug qpos during a trial. Updating an external target is allowed; the plug itself
must move through integrated dynamics. Apply force and torque consistently at the
grasp frame, including the moment arm to the center of mass.

For misalignment trials, preserve the specified lateral and angular target errors.
Do not secretly servo them to zero. Compliance may allow contact-induced
self-alignment; how much is a measured property of the holder and connector
together. Record both commanded error and actual error at first contact.

Include two separately labeled modes:

1. Fixed-error insertion: tests geometry and passive compliance.
2. Active correction: later tests a controller's recovery behavior.

Keep the existing constant-force test as a diagnostic, not the sole acceptance
benchmark. Free acceleration into a hard stop does not represent a controlled
robot approach.

### Solver starting point

Begin from the current timestep 0.0005 s, implicitfast integrator, Newton solver,
100 iterations, friction `0.4 0.005 0.0001`, and condim 3.

The existing prototype changed solref from `0.02 1` to `0.002 1` after observing
approximately 2 mm backstop penetration. Preserve that history, but revalidate
the setting on the thinner blades and new holder. The previous 0.15 N push is
also a baseline setting, not a calibrated real-plug force.

Measure penetration relative to blade thickness and slot clearance. Do not
declare stability merely because qpos and qvel are finite. Check solver warnings,
unexpected resets, contact penetration, velocity spikes, force spikes, and
geometry bypass. Repeat representative contact trials at half timestep to check
whether classifications and force/depth traces materially change.

Change one major physics parameter at a time. Log old/new values, rationale,
expected effect, and measured outcome.

### Gate 2

Aligned insertion seats and holds repeatably under controlled approach speed.
No tunneling, artificial table alignment, or blade-through-separator behavior.
Report the peak penetration and force, not only a success boolean.

## Phase 3 — Success, diagnostics, and a useful difficulty boundary

### Geometry-aware success

Replace the old >25 mm depth rule: the proposed blades are only 16 mm long.

For each blade, compute tip depth in its corresponding slot frame. A proposed
initial success rule is:

- Both blade tips reach at least 15 mm depth.
- Housing front-face gap is between -0.2 mm and +1 mm relative to the socket face.
- Both blades occupy their assigned slot corridors, checked using their
  geometry and relative pose, not only tip center positions.
- Full relative orientation is within 3 degrees of an allowed mating pose,
  accounting explicitly for the chosen connector symmetry.
- No prohibited collision, excessive penetration, or force-limit violation.
- The valid state persists for at least 0.25 s.

Validate these starting tolerances against the final geometry. A blade beside the
socket, one inserted blade, a rotated housing, or a plug beyond the fixture must
never count as success. Report maximum depth separately from final seated state.

Use separate outcomes for success, jam, timeout, initial-state rejection,
force-limit abort, dropped plug, and numerical failure. Define jam as persistent
low insertion progress under commanded advance plus sustained contact over a
configured window; do not label every timeout a jam.

### Diagnostics

Log timestamped traces of:

- Socket-relative plug pose, each blade depth, face gap, lateral and angular error.
- Plug linear/angular velocity and holder target/wrench.
- Plug/socket contact count and force, separated by blade and housing where useful.
- Maximum penetration and solver warning counts.
- First-contact pose, time to seat, hold duration, and termination reason.

Filter plug/socket contacts explicitly. Keep table, gripper, and other collisions
in separate channels. Distinguish summed contact-force magnitudes from net wrench:
opposing slot forces can cancel in the net wrench while still indicating binding.

### Sweeps

Start with signed one-axis sweeps, then sample combinations near the observed
boundary. Proposed values:

- Y and Z offsets: 0, ±0.25, ±0.5, ±1, ±2, ±3 mm.
- Roll, pitch, yaw: 0, ±1, ±2, ±5, ±10 degrees.
- Several documented approach speeds, force limits, and holder stiffnesses.

Run the same cases before and after enabling the lead-in. Do not tune toward a
predetermined angle threshold; report the resulting success map and its dependence
on compliance. Large errors that miss the socket entirely are failures, not jams.

Store versioned results under `results/<scene_version>/<test_config>/`, including
CSV summaries, trajectories, representative videos, and configuration metadata.
Include seeds, model hash, MuJoCo version, tolerances, and actual first-contact
errors. Preserve the old box sweep as a baseline artifact.

### Gate 3 — Connector ready for robot integration

- At least 20 repeated aligned trials succeed with reproducible traces.
- Small-error cases demonstrate contact and occasional passive correction.
- Larger errors produce a measurable failure region without geometry bypass.
- Signed vertical and all three angular sweeps retain meaningful errors at contact.
- Cases for one-blade insertion, roll error, lateral bypass, and excess depth
  explicitly fail success detection.
- Representative success and failure videos agree with automated labels.
- Halving the timestep does not materially change the boundary without explanation.

A compliant-holder pass is necessary but not sufficient for robot validation.

## Phase 4 — One-arm OpenArm environment and control contract

### 4A. Pin the embodiment before collecting data

Use one active arm and an already-held plug. Keep any other arm parked and the
socket fixed to the fixture. This is the first task, not a bimanual benchmark.

Inspect the official [OpenArm MuJoCo assets](https://github.com/enactic/openarm_mujoco),
which provide multiple hardware variants. Record the selected variant, source
commit, license, joint order, actuator types, gripper mapping, and camera setup.
The user confirmed OpenArm v1 with two arms and one active for this task. The
implementation pins upstream commit `161039cd74ea8675fb8197836fe5674659825c75`
and uses `v1/openarm_bimanual.xml` with its Apache-2.0 license retained. The right
arm is active; the left is parked. Robot dimensions are upstream v1 dimensions;
fixture placement, camera mounts, gains, and contact properties remain simulation
assumptions pending measurement on the intended hardware.

Before freezing the scene, verify joint limits, FK, robot base placement, socket
reachability, jaw opening, grasp transform, and finger/socket clearance. Preserve
the connector sites from Phase 1. Asset inspection can proceed while the lead-in
comparison runs; robot contact validation follows Gate 3.

### 4B. Demonstrate insertion through robot control

Compose a separate robot scene. Refactor connector diagnostics so they accept
the robot model/data rather than owning/resetting the whole scene. Keep the
standalone holder harness as a separate regression tool.

The implemented benchmark uses physical finger contacts from the outset; there
is no fixed attachment in the pilot. Representative nominal insertion has about
0.225 mm maximum grasp drift, 1.69 N peak plug/socket contact force, and 0.0043 mm
peak penetration. These are configured simulation results, not calibrated real
connector forces. The robot uses torsional finger contact and an elliptic friction
cone with `impratio=100`; preserve the documented half-timestep validation.

Remove the external plug holder and ideal plug gravity-compensation wrench from
robot trials. Robot actuators, controller, and grasp must generate the motion.
qpos writes are permitted for reset only.

Use a Cartesian expert internally if useful, but select the policy action
interface to match the pinned OpenArm controller. Proposed default: joint-position
targets plus a gripper command, with named joints and explicit units. If the
deployment controller warrants Cartesian actions, decide before collection and
use the same adapter for expert, VLA, and RLT. Do not mix action conventions.

Implement approach, guarded insert, hold, withdraw, and bounded correction/retry.
Log whether an action came from expert, teleoperator, VLA, or RL.

### 4C. Expose the interface all later stages will share

Provide a small environment API:

```python
obs, info = env.reset(seed=seed, options=reset_config)
obs, reward, terminated, truncated, info = env.step(action)
```

Define one step as one robot command interval, not one MuJoCo timestep. Retain
0.5 ms physics substeps initially. Choose command frequency after controller tests;
50 Hz is implemented in simulation, not yet a verified OpenArm deployment rate.
Actions are seven absolute right-arm joint targets in radians plus symmetric
per-finger slider travel in metres, not total jaw aperture. State order is left
seven joints plus mean finger travel, then the corresponding eight right values.

Use a separate chunk executor over this API. Record prediction horizon H,
execution prefix C, timestamps, and actual executed commands. Evaluate inference
latency before choosing C; a nominal large output horizon does not require
executing the entire prediction without observing again.

Actor observations: fixed RGB camera, wrist RGB if supported, robot joint/gripper
state, and instruction text. Privileged fields such as exact blade fit and socket
pose stay in info for evaluation/reset/reward. Do not include simulator contact
force as policy input unless a matching sensing channel is planned on hardware.

State resets must include velocities, controller targets/integrators, grasp state,
and observation buffers. Separate task termination from infrastructure truncation.
For learning trials, a detected jam should allow a bounded recovery period;
the holder benchmark's immediate jam termination would otherwise prevent learning
withdraw-and-realign. Force and numerical aborts still terminate safely.

### Gate 4

- At least 20 aligned robot insertions without grasp slip or prohibited contact.
- A documented signed misalignment benchmark under the robot controller.
- Replay of recorded commands reproduces trajectories within declared tolerances.
- Camera observations and commands are time-aligned; no simulator pose leaks into
  actor inputs.
- Successful reset, insertion, recovery, and controlled retreat on video.

## Phase 5 — Pilot demonstrations, then supervised VLA fine-tuning

### 5A. Choose a VLA with the downstream RLT interface in mind

Primary first baseline is now **SmolVLA**, following the user's model choice.
Pin its LeRobot/checkpoint revisions, implement its own preprocessing and terminal
mask mapping, then profile one training batch on the local 8 GB GPU before an
overfit run. The existing pi0 adapter is preserved as a later comparison.
SmolVLA-based RLT would be a method adaptation, not an exact PI reproduction.
Sources: [SmolVLA guide](https://huggingface.co/docs/lerobot/main/smolvla) and
[architecture overview](https://huggingface.co/blog/smolvla).

The original, now secondary, research path is a flow-based PI model available in
[official openpi](https://github.com/Physical-Intelligence/openpi), initially
assessing pi0/pi0.5. Openpi documents custom-data fine-tuning through LeRobot and
remote policy serving. This is an adaptation choice, not a claim of an official
OpenArm checkpoint or a reproduction using the paper's exact model.

Before a training run, record checkpoint/license, repository commit, backend,
available GPU/VRAM, normalization, action dimensionality, and feature-hook access.
Verify that the selected implementation exposes internal embeddings and sampled
action chunks. An action-only remote endpoint is insufficient for later RLT.

The current workspace is Windows with an 8 GB GPU. Verify SmolVLA's installed
runtime and actual memory footprint; do not inherit openpi's LoRA requirement as
a blanket requirement for all models. For the secondary pi0 path, plan a separate Linux GPU training/inference
environment; openpi documents Ubuntu support. Keep MuJoCo scene development
independent. Check backend-specific fine-tuning support before selecting LoRA or
full updates. Available compute remains unknown; confirm the training host and
resource budget before large checkpoint downloads or training jobs.
Source: [openpi requirements and training guide](https://github.com/Physical-Intelligence/openpi).

If compute requires a smaller VLA, record it as a deliberate alternative and prove
the feature/action interface first. Do not silently replace the PI-oriented path.

### 5B. Build a small, replayable dataset before scaling collection

The original 20 pilot episodes and a separate ten-episode varied batch are
collected. The latter varies approach distance and small angles as well as Y/Z;
it has no recoveries. Visual inspection flags limited slot visibility at contact.
See [DATA_COLLECTION.md](docs/DATA_COLLECTION.md) before scaling. Next consider
100-300 successful and
recovery demonstrations if replay and coverage checks pass. These are project
planning budgets, not established sample requirements or promised success rates.

The collected pilot inserts the already-held plug with seeded Y/Z offsets within
plus/minus 0.3 mm. It does not yet cover angular/grasp variation or genuine recovery
demonstrations; the deliberate fault-injection recovery probe is excluded from BC.
Expand to randomized reachable socket
poses and grasp variations only after the nominal dataset round-trips.

Record:

- RGB frames with camera names, intrinsics/extrinsics, and capture times.
- Joint/gripper observations and commanded robot actions with send/execution times.
- Instruction, task phase, controller/grasp mode, episode ID, and scene/config hash.
- Outcome, intervention/source labels, termination/truncation, and reset seed.
- Privileged physics diagnostics in a separate channel.

Keep primitive-rate observations/actions so future chunk horizons can be formed
without recollection. Preserve both commanded and applied actions when clipping
or safety overrides occur. Do not label an unexecuted action as the transition.

Export using a version-pinned LeRobot schema and a dedicated OpenArm adapter.
Compute normalization using training episodes only. Split by whole episodes and
scene configurations, with an untouched benchmark of at least 100 seeded trials.
Maintain near-nominal, precision-boundary, and out-of-distribution subsets.

Failed autonomous actions are retained for RL/evaluation, not mixed into ordinary
behavioral cloning as successful expert labels. Include expert recoveries as
successful corrective demonstrations. Save handoff/phase labels for later use.

### 5C. Fine-tune incrementally

First overfit a tiny subset to detect action ordering, units, camera, and
normalization bugs. Then train on the pilot/full dataset and evaluate closed-loop
rollouts, not just validation loss.

Compare the unfine-tuned checkpoint, scripted expert, and supervised checkpoint.
The supervised VLA should attempt the entire already-held approach/insertion
task: do not omit precision demonstrations on the assumption that RL will solve
a behavior the VLA never learned.

Start with one instruction and one task. This tests visuomotor adaptation, not
language generalization. Add multiple meaningful instructions/tasks only after
this baseline works, retaining earlier tasks during subsequent training.

### Gate 5

Dataset replay passes; the policy is competent on near-nominal insertion and
reaches the precision phase consistently. Report its actual boundary failures,
latency, forces, and held-out success. A useful initial target is >=80% near-nominal
success; this is a project gate, not a paper result. If it fails due to perception
or action-interface bugs, fix those before RL.

## Phase 6 — Prepare Physical Intelligence's RL Token interface

### Method identity and source

RLT means **RL Token: Bootstrapping Online RL with Vision-Language-Action Models**.
The paper uses pi0.6. It trains an encoder-decoder bottleneck to reconstruct
stop-gradient VLA embeddings, then freezes the VLA and token module for online
RL. Its actor conditions on the token, proprioception, and a sampled VLA reference
chunk; its critic evaluates state/action chunks. TD3-style learning uses a
reference-distance penalty and reference-action dropout. It uses sparse terminal
success rewards. See [paper, Sections IV-V and Algorithm 1](https://arxiv.org/html/2604.23073v1).

The token is a learned representation, not a manually defined insertion-error
vector. RLT is distinct from full-model RL, a generic residual controller, or
RECAP. The PI [technical overview](https://www.pi.website/research/rlt) describes
the compact representation and targeted online adaptation.

### Project preparation and acceptance

Create a small feature-adapter experiment on the selected supervised checkpoint.
Record exactly which embedding layer/tokens are available, their shape, camera
preprocessing, and inference timing. Audit this mapping against paper Section IV-A
before calling the implementation RLT-compatible.

Train and validate the token module on held-out demonstration episodes. Confirm
that reconstruction gradients do not alter the VLA and that token training leaves
reference actions unchanged. Check whether the representation varies across
visually distinct approach/contact states; low reconstruction loss alone does not
establish control usefulness.

Create one versioned inference bundle containing checkpoint, token module,
normalization, camera/action schema, and adapter source revision. Any change to
this bundle invalidates cached features and replay entries.

**Gate 6:** reproducible feature extraction, held-out token checks, unchanged VLA
reference behavior, and an inference latency report. Freeze this bundle before
online RL. Do not claim exact reproduction when using pi0/pi0.5 on OpenArm.

## Phase 7 — RLT in simulation, initially on insertion only

### Algorithm implementation target

Implement the paper's chunked off-policy loop: collect VLA warmup transitions,
then train small actor/critic networks from replay while the VLA/token stay frozen.
The actor outputs action chunks, not additive residuals. Keep the unmasked
reference as the regularization target during reference-input dropout. Optional
human corrections replace the training reference on intervened samples.
Audit against [Algorithm 1](https://arxiv.org/html/2604.23073v1#S4).

PI targets critical phases and describes labeled handoff followed by autonomous
switch prediction. Our simulation will first use explicit preinsert resets, then
VLA-generated approach endpoints; autonomous handoff is a separate acceptance
test. Source: [PI RLT overview](https://www.pi.website/research/rlt).

### Project-specific replay, reward, and execution rules

Use the same controller/action normalization as supervised deployment. Begin with
one environment and a small replay smoke test before parallelizing collection.
Do not introduce a second action decoder inside the RL path.

Store actual chunk duration, executed commands, reference prediction, source,
observations/features at both boundaries, intermediate rewards, and termination
flags. Test terminal chunks, early aborts, and intervention boundaries explicitly.
Compute discounts from actual executed duration; never bootstrap from a reset
state or attach rewards to an unexecuted padded suffix.

Use binary seating success as the first reward, computed by the validated geometry
oracle in simulation. Keep force/penetration limits as aborts and report violations
separately. A dense progress reward or privileged-state critic is a separately
named ablation, not the default comparison.

The reset bank must include states produced by the supervised policy, not only
perfect expert approaches. Track whether failure originates in the approach,
handoff, grasp, or insertion. Handle visible partial jams with recovery; exclude
numerically invalid resets rather than rewarding their exploitation.

Before long training, verify replay alignment, action bounds, gradient isolation,
target updates, and checkpoint resume. Start with a short fixed interaction budget,
inspect actual rollouts, and scale only when the learning loop is correct.

### Gate 7 and comparisons

Evaluate supervised VLA, VLA warmup policy through the RL executor, and trained RLT
with matched reset seeds and action/force limits. Add token/regularizer/dropout
ablations after the main implementation works. Run at least three training seeds
when compute permits and disclose any smaller study.

Report success by difficulty bucket, time to completion, force/penetration,
retries, simulator failures, interaction steps, and wall-clock time including
resets/inference. Do not transfer PI's reported time-to-improvement to this setup.

Accept improvement only when held-out precision performance improves without
regressing near-nominal behavior or increasing unacceptable contacts.

## Phase 8 — End-to-end handoff and eventual physical transfer

Evaluate the entire held-plug approach -> handoff -> insertion trajectory.
An oracle socket-distance switch is a diagnostic baseline only. Train or validate
a handoff predictor using deployable observations and labeled phase transitions.
Define fallback/withdrawal behavior when handoff or insertion fails.

Any final model update requires rechecking token compatibility, cached features,
and earlier success. Keep the simulation benchmark frozen for comparison.

Physical OpenArm trials are a later milestone: confirm actual embodiment,
grasp/camera calibration, controller timing, and an unpowered connector fixture.
Begin with supervised execution and measured simulation discrepancies, then
consider bounded on-hardware adaptation. Simulation RLT results alone do not
establish transfer to hardware.

## Later extensions and calibration

Add axial resistance only after wall contacts and two-slot insertion are validated.
A phenomenological resistance curve must oppose insertion appropriately, apply
only during valid blade engagement, and have documented energy/withdrawal behavior.

Initially omit the long cable or render a non-colliding stub. Add cable mechanics
when testing routing, pulling, or realistic grasp loads. A rigid long cable would
introduce misleading torques.

If real hardware becomes available, measure blade/slot dimensions, lead-ins,
housing grasp geometry, insertion/withdrawal forces, and failure boundaries.
Calibrate against those measurements before making sim-to-real claims.

Add polarized geometry, pickup, unplugging, alternate fixtures, and multiple
connector instances as separate versioned tasks. Preserve comparable benchmarks.

## Repository structure and remaining additions

```text
assets/connector/             shared plug/socket; preserve straight-slot baseline
assets/scenes/                standalone holder scenes
configs/                      connector, robot, controller, cameras, datasets
connector/                    shared geometry-aware diagnostics
envs/                        OpenArm scene composition, reset/step, observations
controllers/                  robot command adapter and chunk execution
scripts/                      connector tests, robot expert, collect/replay/evaluate
data_pipeline/                pilot serialization, validation, VLA data adapter
data/openarm_v1_pilot/         local pilot episodes, splits, replay/readiness reports
third_party/openarm_mujoco/    pinned official assets; fetched separately
learning/vla/                 local pi0 adapter, full-window dataset, optional upstream transforms
learning/rlt/                 FUTURE token, replay, actor/critic, training configs
tests/                        physical regressions and timing/schema contracts
results/<version>/<config>/   metrics, hashes, traces, videos, benchmark manifests
```

Only `learning/rlt/` above is a future addition. The VLA trainer and LeRobot export
are also still future work. Pilot data and fetched
upstream assets are ignored by version control; preserve their manifests and use
the documented collection/fetch workflow. Keep heavy model dependencies separate
from the existing lightweight MuJoCo environment.

## Concrete next implementation milestone

Immediate milestone: **phase-aware full-task environment and synchronized pilot
demonstrations**, followed by the SmolVLA one-batch smoke test. Continuous physical
table pickup -> transport -> insertion now passes six cases using a socket rotated
90 degrees about its unchanged axis/location. Synchronizing joint interpolation
fixed a transport-path collision; the measured acquired grasp is retained through
insertion. No reset or plug teleport bridges phases. Nominal replay is exact after
matching command-boundary physics refreshes. Earlier failures remain preserved.
These are scripted engineering validations, not a generalization result or a
completed full-task learning dataset. See [FULL_TASK_DEMONSTRATOR.md](docs/FULL_TASK_DEMONSTRATOR.md).

Goal: **a checkpoint-specific VLA data/inference round-trip and tiny supervised
smoke test**, using the existing pilot before collecting much more data.

Current priority: SmolVLA, not pi0. The ten-episode varied collection and camera
review are complete. Before broad collection, improve or validate precision-view
observability, implement the SmolVLA adapter, and profile a single training batch
on the local GPU. No SmolVLA install, checkpoint download, or overfit run is
claimed complete. Preserve the pi0 work below as a secondary track.

Progress: pi0 base / JAX LoRA selected as the first candidate; official source is
pinned to `215abfb217dbac7d5f1273282331b9b1866c0479`. The OpenArm numeric/camera
adapter and full-window raw-pilot path are implemented and locally tested.
Checkpoint weights and training dependencies are not downloaded. Local hardware
is an 8 GB RTX 4060 Ti on Windows; a suitable Linux training host is unconfirmed.
Before training, investigate the retained float32 replay velocity failure across
more seeds/timesteps, runtime-test upstream transforms, and connect the filtered
dataset to the trainer. Do not equate this partial progress with milestone exit.

1. Pin the selected PI checkpoint, license, repository/backend revision, and
   Linux GPU host/VRAM budget. Check internal embedding access for later RLT now,
   without implementing the token trainer yet. Record any adaptation from the
   paper's model explicitly.
2. Implement a version-pinned LeRobot/checkpoint adapter. Map both cameras,
   bimanual state, right-only actions, units, image preprocessing, and any padded
   dimensions explicitly. Preserve requested versus applied command semantics;
   never derive actions from the next observed joint position. Prove exported
   samples and decoded inference commands match the existing controller contract.
3. Overfit one or two training episodes. Require finite loss/gradients and useful
   fit improvement, save/reload consistency, valid decoded commands, and zero
   leakage of simulator pose/contact diagnostics into actor inputs. Run a short
   policy-server-to-simulator rollout to catch timing and normalization bugs.
4. Add a seeded closed-loop evaluator comparing expert, base VLA, and fine-tuned
   VLA with identical resets and safety limits. Record latency, chosen chunk
   horizon/execution prefix, success, contacts, and failure phase. Separate smoke
   test rollouts from an untouched benchmark of at least 100 held-out resets.
5. Only after interface checks pass, expand toward 100-300 demonstrations with
   reachable pose/angle variation and genuine expert recoveries. Split by episode
   and scene configuration, refit normalization on training only, and evaluate
   near-nominal versus boundary/generalization performance separately.

Exit criterion for this next milestone: a reproducible training configuration,
checkpoint-specific data validation, a saved tiny-fit checkpoint, and a measured
closed-loop rollout report. Low training loss alone is not Gate 5 acceptance.
If the pilot is too narrow for useful rollouts, document the failure and expand
coverage rather than proceeding directly to RL.

Sim-to-real work runs alongside this sequence: verify the real v1 controller's
gripper convention, limits and latency, measure camera/fixture/grasp transforms,
and compare unpowered insertion forces with simulation. Do not randomize contact
parameters broadly without a measured or explicitly bounded rationale. RLT remains
after the supervised baseline and feature/token gates; no RL trainer or physical
robot execution is part of this documentation update.

## Decisions to resolve at their dependency points

| Decision | Needed before | Proposed starting choice |
| --- | --- | --- |
| OpenArm revision and active side | Resolved for simulation | User-confirmed v1, both arms present, right active and left parked |
| Gripper/control interface | Hardware adapter | Implemented seven joint targets + per-finger travel; real controller mapping still requires audit |
| Cameras and rate | Physical calibration | Implemented fixed/wrist RGB at 50 Hz command boundaries; mounts/rate are simulation assumptions |
| GPU/Linux training environment | VLA smoke test | Local 8 GB GPU is below documented LoRA requirement; separate Linux host unconfirmed |
| VLA checkpoint/backend | Upstream runtime validation | SmolVLA primary; pi0_base / JAX LoRA retained as secondary. SmolVLA revisions/runtime not yet pinned |
| Token feature hook and implementation fidelity | RLT training | Audit selected model against paper Section IV |
| Action horizon/execution prefix | Closed-loop VLA evaluation | Measure latency and contact response first |
| Physical deployment | Real-robot runs | Later calibrated unpowered fixture |

## Research references

Checked 2026-09-12. Pin source/code revisions at implementation time.

- [Physical Intelligence RLT overview, March 19, 2026](https://www.pi.website/research/rlt).
- [RL Token paper, arXiv v1, April 24, 2026](https://arxiv.org/html/2604.23073v1).
- [Official openpi models and fine-tuning guide](https://github.com/Physical-Intelligence/openpi).
- [Official OpenArm MuJoCo assets](https://github.com/enactic/openarm_mujoco).

Community RLT implementations may be inspected later, but their names are not
evidence of paper fidelity or official support. Verify any selected implementation
against the primary paper and record deviations.
