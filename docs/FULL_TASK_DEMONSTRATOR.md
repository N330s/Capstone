# Continuous table-to-insertion demonstrator

Validated 2026-09-13: **downward rest -> table pickup -> lift -> transport ->
insert -> hold**, using the active right arm with the left parked. The plug is
free throughout. There is no reset, plug pose overwrite, weld or external holder
between pickup and insertion.

## Changes that made the task connect

1. **Socket orientation:** rotate the socket 90 degrees about its insertion axis.
   Its location, insertion direction, slot dimensions and fixture support remain
   unchanged. This is a new full-task fixture variant, not the original held-plug
   benchmark. No socket motion occurs during a rollout. The optional lower-fixture
   branch exists for experiments but was not used in the passing physical runs.
2. **Measured grasp:** use the plug/tool transform established after physical
   closure, including the plug's closure-induced rotation. The expert accesses
   simulator pose for this; that privileged transform is not a policy observation.
3. **Transport execution:** synchronize joints along a checked joint-space segment.
   The old independently saturated ramps traced a different curve, causing a
   socket collision despite clear endpoints. The corrected smooth interpolation
   stays on the planned command segment; physical contact and grasp monitoring
   remain necessary because dynamics need not track it perfectly.
4. **Payload-aware checking:** a separate scratch model state positions a rigid
   carried-plug proxy for collision queries. It never modifies the live robot or
   plug state. Seeded bidirectional RRT is available if the direct segment fails;
   all six passing cases used direct checked segments, so the RRT fallback is not
   established by these successes. Discrete collision sampling is not a formal
   continuous-collision or hardware safety guarantee.

The multistart audit tried 25 IK seeds per fixture/mating variant. No valid
solutions were found for the original orientation or its half-turn mating pose;
10 seeds found solutions after rotating the socket. This is evidence for choosing
the variant, not proof that the original fixture is globally unreachable.
See `results/transport_audit_v1/report.json`.

## Acceptance evidence

Six complete physical runs pass in `results/table_insert_validation/`: nominal,
+/-2 mm X placement, +/-2 mm Y placement, and a half-timestep run. These are
development cases, not the reserved insertion-only evaluation bank.

| Metric | Observed across six cases |
| --- | --- |
| Final blade depth | 15.994-15.998 mm |
| Valid seating hold | 0.26 s at command-boundary checks |
| Maximum plug/socket contact force | 0.848 N |
| Maximum plug/socket penetration | 0.00231 mm |
| Maximum relative grasp drift | 0.725 mm |
| Unwanted robot contact force | 0 N detected |

Nominal duration is about 39 seconds after one second of reset settling. This is
a conservative scripted demonstration, not an optimized execution-time result.
The success test uses the shared two-blade geometry oracle. Socket force and
penetration, forbidden carried-plug contacts and robot contacts are monitored at
physics substeps; grasp checks and valid-hold accumulation occur at command
boundaries during insertion. The plug remains gripped after seating; releasing
the seated plug and withdrawing the hand are not yet validated.

All 27 unit tests pass, including a test that collision planning does not change
live qpos/qvel. The nominal motor-command replay reproduces every recorded qpos
and qvel exactly without rerunning the expert or planner:
`nominal/replay_report_v2.json`.

The first replay audit failed because its loop omitted the command-boundary
`mj_forward` call used by collection. That refresh affects bias forces in the
next motor-control interval. The replay implementation now matches that timing;
`nominal/replay_report.json` is retained as the failed historical audit. This fix
did not change the recorded rollout or relax replay tolerances.

## Watch or rerun

[Full physical rollout video](../results/table_insert_v3/rollout.gif) and
[insertion frame](../results/table_insert_v3/insert.png). The video predates the
addition of command logging but uses the same successful controller behavior.
The closer socket camera still needs an observability review for policy training.

Run from the repository root, using output paths that do not already exist:

```powershell
python scripts/probe_table_insert.py --render --output results/full_task_new
python scripts/replay_table_insert.py --episode results/full_task_new
python -m unittest discover -s tests -q
```

`view_openarm.py --play` still runs the pickup-only preview. Use the full-task
command above for this rotated-fixture rollout; do not mistake the old insertion
or pickup viewer for the full-task variant.

## Training boundary and next work

This is a full-task **scripted demonstrator**, not a trained policy or completed
learning environment. It uses privileged pose feedback, scratch IK and collision
planning. Command traces record 50 Hz targets, qpos/qvel, phase and timing, including
settling, but are not yet synchronized T+1 RGB/proprioception LeRobot episodes.
Do not train SmolVLA directly on these diagnostic traces or rendered GIFs.

Next:

1. Move the full-task controller into a phase-aware environment/collector using
   the shared eight-dimensional physical action contract. Add reset, timeout,
   grasp-loss, retry and termination tests; keep the command-boundary physics
   refresh identical in recording, policy serving and replay.
2. Freeze a camera/fixture/controller schema, then record a small full-task pilot
   with synchronized RGB, bimanual proprioception, commands and full-task language.
   Keep simulator geometry/planner state strictly outside actor inputs.
3. Reserve a new full-task table-pose evaluation bank; retain the old insertion
   benchmark separately. Replay and inspect the pilot before scaling to 100 demos.
4. Pin SmolVLA/LeRobot, test preprocessing and one-batch memory/backprop on the 8 GB
   GPU, then tiny-overfit and closed-loop evaluation. No VLA/RLT training has run.
5. Add recoveries, broader grasp/pose coverage and calibrated randomization.
   Later RLT handoffs must come from the VLA's actual acquired grasps/approaches.

The rotated unpowered fixture must be reproduced or explicitly adapted on real
hardware. These narrow simulation results do not validate sim-to-real transfer.
