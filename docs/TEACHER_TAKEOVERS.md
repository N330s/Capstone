# Pickup recovery teacher qualification

## Purpose and scope

Test whether the two failed five-parent training trajectories are physically
recoverable before changing VLA training or collecting more nominal episodes.
This follows [the collision diagnosis](FIVE_EPISODE_COLLISION_DIAGNOSIS.md).

Four fixed cases: `train_20003` and `train_22045`, each after 275 commands
(before plug contact) and 350 commands (after one-sided contact). These are
two training resets, not four independent held-out scenes. The VLA is not
re-evaluated in these tests: its saved failed prefix commands are replayed,
followed by a **privileged scripted teacher**.

Initial experiment: `outputs/teacher_takeovers_20261020`: **3/4** pass, all
prefixes reproduce with exactly zero qpos/qvel error. `train_22045` at 350
stops safely because the fully opened hand still has one light pad/housing
contact. The initial attempt, video, observations and failed labels are retained;
it is not a positive corrective segment.

Repaired-teacher requalification: `outputs/teacher_takeovers_20261020_v2`.
Current status: **complete, 4/4 teacher pickups**, independently audited in
`results/teacher_takeovers_20261020_v2`.
Initial independently audited evidence remains at
`results/teacher_takeovers_20261020` (3/4); nothing is overwritten.

## Results

| Training parent | Takeover after commands | Corrective commands | Maximum lift | Continuous carry | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| train_20003 | 275 | 703 | 65.84 mm | 1.00 s | teacher pickup |
| train_20003 | 350 | 721 | 66.67 mm | 1.00 s | teacher pickup |
| train_22045 | 275 | 699 | 63.54 mm | 1.00 s | teacher pickup |
| train_22045 | 350 | 736 | 64.29 mm | 1.00 s | teacher pickup |

All four prefixes have exactly **zero qpos/qvel error**, with identical
requested/applied commands and execution intervals. No case has a drop,
unwanted robot contact or robot/table contact. All three initially successful
teacher motions remain bit-identical after the release repair.

In the repaired fourth case, contact clears in 0.38 simulated seconds; maximum
observed release contact force is 0.261 N (limit 0.5 N). The later planned
grasp/lift passes the original physical gate. The brief roughly 2.6 mm
ungrasped motion during release is not counted as pickup.

Recorded **2,859 qualified candidate commands (57.18 simulated seconds)** and
2,863 corresponding pre-/post-action dual-camera observations across four
segments. These are not yet exported, replayable-as-fresh-reset episodes or
evidence of VLA learning. The original scored VLA remains **3/5 training
pickups**, and held-out validation has not run.

Evidence:

- [Recovery traces](../results/teacher_takeovers_20261020_v2/recovery_traces.png),
  [PDF](../results/teacher_takeovers_20261020_v2/recovery_traces.pdf).
- [Actual video contact sheet](../results/teacher_takeovers_20261020_v2/takeover_frames.png).
- [Repaired late-contact rollout](../outputs/teacher_takeovers_20261020_v2/train_22045_at_350/rollout.mp4).
- [Audited metrics](../results/teacher_takeovers_20261020_v2/metrics.csv),
  [summary and repair comparison](../results/teacher_takeovers_20261020_v2/summary.json).

Initial four-case campaign: 438.36 seconds. Repaired four-case campaign:
518.64 seconds (8 min 39 s), normal completion, no process deadline exceeded.
Independent audit verifies all saved observations/actions and 50 fps video
frame counts. **31 selected unit/regression tests pass**, covering prefix
reproduction, ownership, contact-normal direction/ambiguity, the physical
pickup/grasp gates, sampling, FP32 training and watchdog/disk safety helpers.
Both attempts, geometry diagnosis and evidence occupy about 260 MiB total;
the final free-space check is 28.30 GiB, above the 25 GiB reserve. No cleanup,
optimization, remote API call or model modification was performed. All
experiment processes have exited.

## Execution and safeguards

- Reset the hash-matched `automateData` snapshot normally, using the original
  episode seed/reset options and the existing physical-grasp bookkeeping adapter.
- Execute the original failed VLA's **requested** commands through `env.step`.
  Check qpos/qvel after every command against the saved trajectory within
  `1e-8`; also verify applied commands and executed simulation intervals.
- At takeover, hold measured arm joints and open. Retract vertically 80 mm
  using the existing private-data IK planner, then settle and re-plan the grasp
  using the actual plug pose. A non-flat settled plug is rejected rather than
  reset or teleported. Use pickup-only planning, with no insertion and no
  hidden regrasp retries.
- Teacher v2 handles the initial teacher's release deadlock. A remaining
  **single-finger** contact must be light (at most 0.5 N), the fingers fully
  open and no grasp established. Plan a 6 mm horizontal separation away from
  all measured hand-to-object contact normals before vertical retreat. Reject
  ambiguous normals or any planned robot/table/object penetration. Stop the
  teacher if observed contact force exceeds 0.5 N or contact persists after
  3 mm actual horizontal travel. This release-only teacher check runs at
  command boundaries; the original simulator guards still run at physics
  substeps and are unchanged.
  This is an extra teacher restriction, not a relaxed collision threshold.
- Every live motion goes through the same bounded/rate-limited action interface.
  The teacher's planner changes only its **private** `MjData`, never live qpos,
  qvel, object placement or physics. Robot/table/contact/slip guards stay intact.
- Apply the original gate: two-finger contact, established physical grasp,
  at least 20 mm actual plug lift and one continuous second of carry, with
  zero prior drops and no safety abort. Carry does not mean stationary hold.
- Maximum 55 simulated recovery seconds per case; supervisor enforces a
  360-second process deadline per case and 1,500-second total allowance.
  Keep at least 25 GiB free. No dataset/model deletion, training or API calls.

The original actor overlay remains frozen at SHA-256
`f1518a2141cf382cafa5e893e12fe698470d0b5c2d62fb33500f8774d7546ac3`.
Original rollout and source episode hashes are checked. The original scored
VLA results remain **3/5 training pickups**, with validation not run.

## Recorded artifacts

Each case preserves:

- `prefix_audit.json`: reproduction errors and the actual takeover diagnostics.
- `trajectory.npz`: all T requested/applied commands, ownership/phase labels,
  T execution intervals and T+1 measured states, qpos and qvel.
- `recovery_observations.npz`: synchronized **actual** scene/wrist RGB and
  measured state at the takeover and after every corrective command. Recovery
  observation `k` precedes corrective action `k`; the terminal image/state has
  no outgoing action. Two 320x240 RGB cameras, 50 Hz command intervals.
- `diagnostics.json`, `teacher.json`, `manifest.json`, `report.json`, original
  source hashes and a 50 fps side-by-side video explicitly labeled as teacher
  takeover, not autonomous VLA success.

The video's outcome footer gives the final trial outcome throughout the clip;
the stage label identifies what is actually happening in each frame.

Prefix ownership is `prefix_replay`; only commands beginning at the takeover
boundary are `teacher_recovery`. A segment becomes a positive **candidate**
only if its physical pickup audit passes. Failed VLA prefix commands are never
positive expert targets. No segment is automatically exported into training.
Phase tags describe the teacher stage after action selection and before the
physics step; transition commands can therefore carry the next stage's tag.
They are audit/sampling metadata, never actor inputs.

`scripts/report_teacher_takeovers.py` independently reopens the saved motion,
rechecks prefix commands/states, environment/guard manifests, physical gate,
RGB/action lengths, labels and all video frame counts. It exports a CSV,
summary, actual-video contact sheet and recovery trace plots without simulation.

## Decision after qualification

Update: the user chose to establish an upstream-aligned SmolVLA baseline first.
See [STANDARD_SMOLVLA_BASELINE.md](STANDARD_SMOLVLA_BASELINE.md). The corrective
mixture proposal below is retained for a later controlled experiment, not the
current immediate training step.

Qualification is now passed. The rules below govern the next training-data
step; it has not been launched. If a future teacher test fails, preserve it and
repair the teacher before using its actions as supervision. Do not weaken the
pickup gate or guards.

With all four passed, the useful next experiment is a **small, controlled corrective
data mixture**, not a large nominal-data collection or another long blind run:

1. Verify/export only the qualified recovery segments. Split by original parent;
   both parents remain training-only. Keep both RGB cameras, measured state,
   the same instruction and the same eight absolute action dimensions.
   Teacher pose/contact/phase information is for labeling/audit only and must
   not become an actor input. Torque/depth remain excluded.
   Retain the original parent, fault-prefix command hash and takeover boundary
   as replay-initialization lineage: a detached recovery segment does **not**
   start at a fresh nominal reset. Its physical replay control must first
   reproduce the saved prefix, never teleport into its first observation.
2. Do not splice faulty prefix targets into teacher chunks or bridge unrelated
   episodes. Define terminal chunk masking/padding explicitly. Sample all
   recovery phases, including close/lift, so long open/retreat trajectories do
   not teach the policy to keep reopening indefinitely.
3. Predeclare a short matched experiment from the frozen five-parent actor:
   nominal-only continuation versus nominal plus a bounded recovery quota.
   Match optimization budget, FP32 trainable weights/Adam, BF16 forward,
   LR/schedule, batch size, seeds and parent exposure. Preserve nominal
   open-hold sampling/augmentation and nominal close/lift examples. Save a
   sample-index/phase/source audit. This separates correction content from
   simply doing more optimizer updates.
4. Evaluate both candidates on all five original training resets, including
   the three existing successes as regression controls. Require 5/5 before
   using the fixed held-out pickup bank with fresh expert controls. Report
   validation failures honestly; do not collect or tune on those resets.
5. Only a reproducible held-out pickup improvement supports expanding the
   recovery bank. Insertion, RL/RLT and sim-to-real remain later gates.

Successful teacher recoveries establish a feasible corrective target. They do
not prove that the VLA has learned recovery, uniquely identify the source of
the initial arm drift, prove the training stack bug-free, or demonstrate
generalization or sim-to-real readiness.

## Initial release failure and bounded repair

The first teacher waited for zero plug/finger contacts before moving upward.
For `train_22045` at 350, finger travel reached 43.959 mm for a 44 mm command,
but one right-finger-pad/housing contact remained at 0.1417 N and 1.7 micrometers
penetration. Waiting in place could not release it. This was a teacher logic
deadlock, not a VLA result, gripper actuator failure or simulator abort.

`outputs/teacher_release_geometry_20261020/report.json` preserves a private-data
geometry inspection of the failed end state. A straight upward path exists,
but its first roughly 2 mm still intersects the housing. This motivates the
small horizontal separation rather than ignoring the planner's collision
check. Geometry alone does not prove a dynamically safe recovery; v2 must
requalify all four cases and retain its own results separately. That physical
requalification now passes as detailed above; the failed initial trial remains
retained and excluded from positive corrective labels.

## Reproduction

```powershell
.venv-smolvla\Scripts\python.exe scripts/test_teacher_takeovers.py --output outputs/teacher_takeovers_NEW
.venv-smolvla\Scripts\python.exe scripts/report_teacher_takeovers.py --experiment outputs/teacher_takeovers_NEW --output results/teacher_takeovers_NEW
python scripts/report_teacher_takeovers.py --output results/teacher_takeovers_NEW --plot-only
```

Use a fresh output directory; existing attempts are never overwritten. The
last command uses the system Python with matplotlib; simulation/video auditing
uses the project environment containing MuJoCo and PyAV.
