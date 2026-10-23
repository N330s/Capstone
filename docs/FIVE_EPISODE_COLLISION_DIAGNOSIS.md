# Why two of the five training-reset pickups failed

2026-10-20. Diagnostic-only follow-up to the completed
[five-parent recipe pilot](FIVE_EPISODE_OPENHOLD.md). Original training result
remains **3/5 live training-reset pickups**; validation was not run. No model,
controller, scene, guard, dataset or scored rollout was modified or replaced.

## Measured failure chain

**Off-path approach -> one-finger contact/pushing -> no established grasp ->
continued empty closure -> finger-to-finger collision.**

The terminal `robot_collision` label is not the arm hitting the table. Both
original reports identify contacts between the right left/right finger pads
and collision geoms. Terminal aggregate unwanted forces are 19.40 / 24.82 N,
above the unchanged **5 N** robot-contact abort threshold. Support-contact peak
is zero in both failures. Do not weaken the guard to turn these into passes.

| Measurement | `train_20003` | `train_22045` |
| --- | ---: | ---: |
| Pre-contact finger travel, command 275 | 34.14 mm | 34.92 mm |
| Nearest own expert approach-path distance at 275 | 17.68 mm | 48.99 mm |
| First finger/plug contact | command 294 | command 311 |
| Finger travel at that contact | 33.93 mm | 35.16 mm |
| Hand to plug reference-grasp-site distance at contact | 31.71 mm | 56.36 mm |
| Saved command-end samples with two contacting fingers | 0 | 0 |
| Established grasp | Never | Never |
| Maximum plug lift from pushing | 15.80 mm | 24.16 mm |
| Terminal finger travel | 1.78 mm | 1.77 mm |
| Abort command | 563 | 530 |

Arm deviation exists before the finger leaves its trained open-hold range.
In particular, `train_22045` is already 50.60 mm from its own expert pre-close
Cartesian path at command 200, with a 35.49 mm opening and no contact. This is
not explained solely by executing the same path faster: distance is to the
nearest reference path point, not just the same command index.

Path error alone is not a universal failure classifier: a successful case
also has about 22 mm path deviation at command 275, then reaches a usable
contact geometry. At first contact, the three successful runs are 10.53,
13.06 and 11.75 mm from their plug reference-grasp sites, versus 31.71 and
56.36 mm in the failures. The references are geometric site measurements,
not a claim that zero separation is the unique physically correct pinch pose.

The failed runs push/displace the plug with one finger rather than capturing
it between both fingers. The 24.16 mm lift in `train_22045` is **not** a missed
success: it never establishes a grasp or sustained two-finger carry. Contact
counts above are saved command endpoints, not every physics substep.

![Failure-chain measurements](../results/five_episode_collision_diagnosis_20261020_v3/failure_chain.png)

## The remaining gripper-feedback problem

Audit every one of the 16,000 augmented training anchors against raw state and
logged applied noise. The sampled open-hold finger states span approximately
**32.94-36.99 mm** across all five parents. None of the original pre-close
observations is below 33 mm. The failed runs cross 33 mm at commands 358 / 367,
after one-sided contact begins, then drift far outside that augmented range.

Fresh offline predictions from the saved model use images rendered from the
saved actual qpos/qvel. For `train_22045`, command 500, seed 1000:

| Offline input | Finger-state input | First predicted target | Mean first-five target |
| --- | ---: | ---: | ---: |
| Actual state and actual-pose render | 10.97 mm | 1.85 mm | 1.40 mm |
| Same images, instruction, other 15 state values and cloned noise; replace only finger state | 34.99 mm | 34.05 mm | 34.07 mm |

Thus the finger-state channel can amplify empty closure after the miss. Two
explicit noise seeds show the same qualitative difference. This does not make
state replacement a deployable fix: it is a deliberately counterfactual input,
not a physically consistent recovery or an autonomous success.

At the start of closure, the first action can look less concerning than its
chunk tail. The original `train_22045` chunk at command 350 starts with a 35.02 mm
measured opening and predicts 34.36 mm first, but only 33.17 mm at offset 24.
At command 400, measured travel is 29.80 mm; the same executed chunk predicts
29.15 -> 26.57 mm across that prefix. These are the original saved predictions,
not the explicit-noise diagnostic's different latent stream.

This does **not** reduce the problem to early closing or chunk size. Successful
`train_20042` also goes below 30 mm before its recorded expert does (437 versus
513), then establishes two-finger contact and completes pickup. The problem is
closing without a usable grasp geometry and failing to recover, not matching
the expert's clock at every command.

## What is established, and what remains uncertain

Established for these two saved trials:

- Geometric drift precedes the terminal closure/self-contact event.
- One-sided contact never develops into an established grasp.
- The policy does not recover; late finger-state dependence promotes further
  closure in a controlled same-image/noise counterfactual.
- The retained guard and physical pickup gate reject a genuine failed pinch,
  not a correct grasp mislabeled as failure.

The exact origin of the arm drift is **not uniquely isolated**. Inadequate
visual correction, nominal-only trajectory conditioning and feedback-sensitive
absolute targets remain plausible contributors. Recorded-observation joint
targets are close to their nominal reference in sampled probes, but that does
not prove stable autonomous execution. Changing right-joint state also changes
predicted absolute targets, including on a successful case; this alone is not
evidence of a state-copy software bug. Same-index action discrepancies also
include phase drift. There is no general bug-free claim.

## Most useful next test (proposed at diagnosis completion)

The subsequent implementation, preserved first teacher failure and repaired
qualification are tracked in [TEACHER_TAKEOVERS.md](TEACHER_TAKEOVERS.md).
The proposal below records the decision at the end of this diagnostic stage.

Before another training run, test **four expert takeovers**: the two failed
training resets at a pre-contact anchor (275) and after one-sided contact (350).

1. Reset normally and replay the original VLA's saved **requested commands**
   through the unchanged environment up to the takeover point. Require saved
   qpos/qvel agreement within a declared tolerance. Do not teleport the arm or
   object into a saved configuration during execution.
2. Re-plan from the actual arm/object pose: reopen when needed, retract to a
   safe hover, center the hand around the actual plug, descend, close and lift.
   Do not merely resume a timed waypoint trajectory based on the original pose.
3. Require the unchanged two-finger + 20 mm lift + one-second carry gate, with
   every collision guard intact. If the teacher cannot recover, repair that
   teacher/planner before collecting or training on its output.
4. If qualified, record synchronized actual RGB/state/corrective commands.
   Mix a small, audited amount with nominal close/hold/lift examples, keeping
   validation separate. These target early approach/contact failures in the
   exact current scene, unlike the older near-object, different-scene recovery
   pilot. Do not mix fault commands into expert targets or overtrain reopening.

This tests a specific missing recovery capability without assuming that more
nominal episodes, larger batches, a shorter prefix or more noise will solve it.
Insertion, RL and sim-to-real remain deferred.

## Evidence and verification

- Raw audited FK/contact/coverage reports:
  `outputs/five_episode_collision_diagnosis_20261020_v2`.
- Main offline analysis: 44.08 seconds. GPU counterfactual probe: 106.80 seconds,
  42 observation/noise groups and 168 predicted chunks, within its 300-second
  outer deadline; no timeout. No optimization or physical rollout was run.
- All five original training NPZs and all five original policy trajectory hashes
  are verified; candidate overlay SHA-256 remains
  `f1518a2141cf382cafa5e893e12fe698470d0b5c2d62fb33500f8774d7546ac3`.
- FK uses **separate `MjData`**, never controls a running simulator. Raw hand/
  plug site distance is checked against saved diagnostics before the first
  grasp. After grasp latching, saved `grasp_slip_m` is reference-relative, not
  raw site separation; the diagnostic explicitly distinguishes these meanings.
- A preliminary offline measurement stopped on that slip-definition mismatch.
  Its failure receipt/source archive remain in the unsuffixed output directory;
  only the diagnostic comparison was corrected. No scored rollout was rerun.
- Original-frame extraction and video hashes:
  `frames_manifest.json`; [contact-sheet image](../results/five_episode_collision_diagnosis_20261020_v3/original_video_frames.png).
  The video footer shows the trial's final outcome on every frame, not the
  instantaneous event at that image. Captions identify actual command indices.
- Final PNG/PDF/CSV/summary:
  `results/five_episode_collision_diagnosis_20261020_v3`. Unsuffixed/v2 results
  directories are preserved plotting/layout drafts; use v3 for reviewed evidence.
- Three new diagnostic unit tests cover one-finger motion versus grasp,
  phase lead versus off-path distance and rotation-matrix error accounting.
  **50 selected tests pass** including the existing sampling, FP32, metric and
  watchdog tests. Original campaign implementation hashes and model hashes
  still match their launch receipts. All diagnostic processes finished.

Static image reconstruction is offline and may differ slightly from original
raw renders. Within each state intervention, RGB arrays and explicit noise are
identical. Neither the render nor latent stream is asserted bit-identical to
the original scored inference. These are five training resets and one execution
seed, not a held-out generalization study.
