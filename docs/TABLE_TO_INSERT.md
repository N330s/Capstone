# Final task: table -> grasp -> insert

Update 2026-09-13: continuous pickup, transport and insertion now pass in the
rotated-socket variant. See [FULL_TASK_DEMONSTRATOR.md](FULL_TASK_DEMONSTRATOR.md)
for current results. The pickup-development findings below preserve the earlier
grasp and reachability history.

The final user scenario starts with a free plug resting on the work table, not
inside the gripper. The right OpenArm v1 arm must approach, grasp, lift, transport,
align and insert; the left arm stays parked. No live plug teleportation, weld,
hidden holder or privileged pose input to the learned policy is allowed.

The existing held-plug task remains a component benchmark and a possible later
RLT precision-phase curriculum, not a substitute for full-task success.

## Current fix: downward rest and physical pickup v3

The default viewer now starts with **both hands pointing downward**, using the
official v1 zero-joint configuration behind the table. The original forward-facing
pose is retained only in the explicitly selected held-plug insertion benchmark.

```powershell
python scripts/view_openarm.py
python scripts/view_openarm.py --play
python scripts/view_openarm.py --task insertion
```

The first command previews downward rest; `--play` runs reach/close/lift/hold once.
Close and rerun to reset. The GUI path is implemented but was not manually
exercised here; rendered frames, CLI parsing and headless physics were checked.

The actual failure had two parts. The forward start near the socket made direct
joint interpolation collide. Separately, the plug rotated during finger closure;
the old 0.008 m per-finger target then did not retain a loaded pinch on the changed
cross-section. A vertical approach with that old target still fails: only 4 mm
height change, about 79 mm relative drift, and zero final finger contacts.

The corrected candidate uses:

- Downward rest, a raise waypoint behind the table, then an above-table approach.
- Vertical descent to a grasp reference 3 mm above the original housing center
  reference, preserving fingertip/table clearance.
- Slower 0.01 m/s finger closure to 0.006 m per-finger target. This is a motor
  target, not the measured aperture or an assumption that the fingers reach it.
- Contact/numerical checks every physics substep; no suppressed table collisions,
  weld, external plug force, changed friction, or changed collision mesh.
- Relative position and rotation measured after closure, bilateral finger-contact
  checks, and real plug height during lift/hold.

All eight cases in `results/table_pickup_v3/report.json` pass: three identical
nominal repeats, +/-2 mm X/Y placement changes, and a half-timestep trial.
Lift is 73.3-73.6 mm; worst post-closure drift is 0.472 mm and 0.965 degrees.
No unwanted robot contact was detected in these passing trials. All 26 unit
tests pass. The safety threshold is an abort trigger, not a promise that force
cannot overshoot within a physics step; the separate failed ablation demonstrates
this distinction.

See [physical pickup video](../results/table_pickup_v3/pickup.gif) and
[lifted plug](../results/table_pickup_v3/hold.png). The plug rotates approximately
90 degrees during closure, then stays stable in the lifted grasp. This is not
an orientation-preserving pickup. The measured grasp transform is recorded for
transport planning; do not reuse the old nominal insertion grasp transform.

**Open at the pickup-v3 milestone:** controlled set-down, broader pickup coverage, and transport/
insertion with the measured grasp. Initial local IK attempts to seat that grasp
in the existing fixture did not converge, including the allowed half-turn mating
orientation. This is not a proof of unreachability, but it prevents claiming the
current grasp is ready for the complete task. Evaluate alternate IK branches,
grasp orientation or a physically justified fixture layout before full-task data.

The older failures below are retained as diagnostic history, not current pickup
acceptance results.

## Implemented layout/feasibility work

`configs/table_task_v1.json` and `table_task_v2.json` define separate preview
variants. They do not replace the validated insertion scene or its cameras.

- The free plug is initialized 2 mm above the table and settles under gravity.
  Final mating-frame height: 0.32799875 m above world origin (table top 0.32 m).
  Four housing/table contacts, about 1.25 micrometres penetration, negligible
  residual speed, no robot contact and no solver warnings in the tested reset.
- Candidate 640x480 fixed task view covers table plug, active gripper and fixture.
  A separate close oblique socket view provides more detail near the housing.
  The original wrist camera remains in the model. This is a candidate camera
  arrangement, not a frozen three-camera training contract or measured hardware mount.
- Camera probes use the same cameras for the table start and an existing recorded
  seated state. The latter is a **render-only reconstruction**, not a successful
  table-to-insert trajectory. The actual blade/slot interface is still occluded
  at full seating; visibility of the housing/fixture is not proof of precise state
  observability. Inspect approach frames and policy behavior before freezing cameras.

See [table-start view](../results/table_task_v2_checked/table_start_task_rgb.png),
[close socket view](../results/table_task_v2_checked/recorded_seating_camera_probe_socket_rgb.png),
and `results/table_task_v2_checked/report.json`.

## What the grasp checks found

The initial vertical approach put the fingertips about 2.42 mm into the table at
the centered grasp pose. Keeping that tool orientation at insertion also left
only 0.001 rad wrist-limit margin. That candidate is rejected.

An approach pitched 60 degrees with the grasp reference 4 mm higher removes the
static table collision. IK converges for pregrasp, grasp and seated poses; minimum
joint-limit margins are approximately 0.384, 0.506 and 0.187 rad respectively.
These checks cover endpoints only, not a collision-free trajectory or stable grasp.

The physical probe exposed both limitations:

1. Direct joint interpolation from the old insertion home pose encountered a
   prohibited contact during reach and aborted. The 50 Hz monitor observed about
   98.7 N before stopping; this is a failed simulation probe, not a validated
   force-limited or hardware-safe motion controller.
2. Adding an 80 mm retreat waypoint avoided the monitored contact abort, but
   closure tipped the plug and the lift failed. Plug height changed only about
   4 mm, versus a commanded 70 mm lift; relative grasp drift was about 71 mm.

Preserved artifacts: `results/table_pickup_v2/` and
`results/table_pickup_v2_retreat/`. Neither attempt is exported as a BC episode.
No successful physical pickup or end-to-end insertion is claimed.

## Run the probes

Run from the repository root. Output directories must not already exist.

```powershell
python scripts/preview_table_task.py --config configs/table_task_v2.json --output results/table_layout_new
python scripts/probe_table_pickup.py --output results/table_pickup_new
```

The physical probe returns failure unless its lift criteria pass. It is an
engineering diagnostic, not yet an environment reset/step interface or training
collector. Grasp slip must be measured relative to the grasp actually established
after closure; the old held-plug zero-offset assumption does not apply.

## Next acceptance gates

1. Diagnose fingertip/housing contact during closure and choose a physically
   realizable pinch pose with table clearance. Do not fix it by suppressing table
   collision or increasing friction without evidence. Test approaches/withdrawals
   through safe waypoints and add substep contact monitoring.
2. Demonstrate grasp -> lift -> hold -> controlled set-down repeatedly, including
   modest table-pose variations. Require bilateral finger contact, real table
   clearance, bounded relative slip and no prohibited contact. A closed gripper
   or elevated robot wrist alone cannot count as a grasp.
3. Add phase-aware full-task reset/step semantics. Before grasp, table contact is
   expected; after lift, grasp loss aborts/recovery rules apply. Preserve the same
   eight joint/gripper commands and bimanual observations. Record separate
   approach, grasp, lift, transport, align, insert, hold and recovery labels.
4. Carry the measured grasp transform into insertion and validate the transport
   path and seating. Evaluate from table resets; do not count oracle handoff resets
   as end-to-end success. RLT must eventually see real VLA-generated handoffs.
5. Collect successful full-task demos for SmolVLA only after these physical gates.
   Existing held-plug data can remain phase-specific auxiliary data, labeled as
   such. Use the full-task instruction for full-task trajectories. Keep pickup
   and insertion failure categories separate in evaluation.

The current insertion-only reserved reset bank remains an insertion benchmark.
Create a separate held-out table-pose bank when the physical pickup task is fixed.
