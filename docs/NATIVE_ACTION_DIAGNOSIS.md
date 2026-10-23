# Why the native SmolVLA pickup failed

2026-10-22. Frozen step-12,243 checkpoint; no training or scene changes.
Both bounded probes completed (202.234 s and 58.031 s). The final selected
unit/regression suite passed **57 tests**. Model hashes remained unchanged.

## Finding

The current actor is not accurate enough even on **correct recorded training
observations**. This is not only a held-out/generalization problem. In live
trials, inaccurate approach/closure is followed by an empty closed gripper and
robot states absent from successful demonstrations. The actor does not recover.
One separate trial stops on early robot collision. This describes observed
failure mechanisms; it does not establish one uniquely responsible code bug.

Evidence:
`results/native_action_diagnosis_20261022_v2/summary.json`, `metrics.csv`,
`action_errors.png` / `.pdf`. Detailed frozen inference:
`outputs/native_action_diagnosis_20261022`; forward-path intervention:
`outputs/native_forward_paths_20261022`.

## What we tested

Thirty offline observations: the median frame in six phases (survey, transit
to pickup, descend, close, lift, insert) on each of the five failed training
resets. Generate all 50 actions but report the **first five**, matching the
executed prefix. Compare against the actual exported target, `action_applied`.
Use correct recorded robot state, both RGB images and instruction, never live
drifted state. Same checkpoint, saved processors and declared BF16 numerical
profile as evaluation. Fixed noise `1000 + source_step`, with a second seed
`2000 + source_step`. Raw-vs-exported-video comparisons use identical state,
instruction and noise. These are selected diagnostic windows, not exhaustive
training/validation statistics or autonomous successes.

### 1. Errors precede closed-loop drift

| Training episode | Closure finger-command RMSE, first 5 (mm) | Same, second seed (mm) | Command-pose discrepancy (mm) |
| --- | ---: | ---: | ---: |
| train_20002 | 4.94 | 4.65 | 5.25 |
| train_20003 | 1.83 | 3.23 | 4.74 |
| train_20042 | 1.43 | 0.46 | 6.96 |
| train_22001 | 2.67 | 2.34 | 6.66 |
| train_22045 | 2.36 | 2.62 | 8.61 |

The primary seed's closure finger biases are negative in all five samples:
the model commands more closure than the demonstrated targets. This persists
with the decoded training-video observations. Arm-command RMSE at these
closure samples is about 0.010-0.019 rad. Static forward kinematics converts
generated-vs-expert joint targets into the final column: robot-grasp-site
translation discrepancy. It is **not actual live tip error or simulated
motion**; it ignores actuation/rate limits/contact and does not prove every
alternative command is invalid. Together with failed live pickup, it shows
the actor has not reproduced demonstration-level motor coordination.

### 2. Live error takes the gripper outside the recorded state range

The full training split's measured right-finger travel spans **18.72-43.99 mm**;
commands span **11-44 mm**. Physical finger travel is not the same as the
command: holding the plug blocks full closure even with a tighter command.

In all four command-limit failures, measured travel goes below the training
minimum within **185-260 commands (3.70-5.20 simulated seconds)**, while the
hand is still **46.6-144.5 mm** from the plug grasp point. It eventually reaches
about **3 mm**, with commands around 3 mm, also below the training-command
minimum. No physical grasp occurs. The collision case stops before this.

This demonstrates an observed distribution shift after missed pickup. The
successful trajectories do not cover those empty-gripper closed states or
corrective recovery actions. It does not prove that extra recovery data alone
will solve the initial approach error. Do not hide it by clamping commands to
the dataset minimum or using privileged plug/phase inputs.

### 3. Low flow loss was insufficient evidence of action accuracy

The pinned native policy averages flow loss over **32 output dimensions**, of
which only eight are real OpenArm commands; the remaining 24 are padding.
This is the installed native behavior, not a newly discovered custom loss bug.
For these 30 training diagnostic windows at flow time 0.5:

- Native 32-dimensional loss: **0.003094** mean.
- Active eight-dimensional velocity MSE: **0.010111** mean (about 3.27x).
- At flow time 0.1, active velocity MSE: **0.087414**, about 8.65x its
  time-0.5 value. Generation traverses multiple flow times, not just 0.5.

These are different samples from the 160-window validation set; do not compare
the absolute values as train/validation gaps. The validation metric deliberately
used fixed time 0.5 and never claimed action or pickup accuracy. The mistake
would be using its reduction as evidence of successful control. Include
generated-command physical errors and physical rollout gates in future model
selection, plus multiple flow times for diagnostic validation.

## Checks that narrow the cause

- **Checkpoint load:** all 500 saved tensors reload exactly in the actual
  `SmolVLAController`, including trained FP32 storage. Model SHA remains
  `0cefdc6ffc0c634f17fe35a94d3940b7f15de2833c8e3142837b5a1a5e7df1de`.
- **Data/processors:** all 30 sampled raw/export state vectors, 50-command
  applied-action targets, masks and instructions agree. Normalization matches
  full training statistics; action postprocessing roundtrips to physical units.
  This is a sampled contract check, not an all-frame dataset certification.
- **Image compression:** paired raw/decoded observations alter first-five
  finger commands by at most **0.273 mm RMSE** in this sample; the closure
  errors remain with training-video input. Compression can still matter near
  contact, but does not account for the observed multi-mm closure errors here.
- **Repeatability:** two fresh-process initial predictions on exact saved live
  inputs reproduce the rollout's saved initial chunks bit-for-bit.
- **Cached versus training forward:** four offline initial/closure cases run
  the same 10-step Euler/noise through a joint training forward rather than
  cached prefix inference. First-five finger changes are **0.004-0.024 mm
  RMSE**, arm changes **0.00035-0.00130 rad**. The 4.94/1.83 mm closure errors
  remain approximately 4.94/1.84 mm. These paths are not bit-exact under BF16,
  but bypassing caching does not remove the sampled failure. Joint forward's
  reconstructed flow input differs by at most 2.39e-7 in FP32.
- **Integration steps:** on four phases of train_20002, 50 instead of 10
  denoising steps does not fix generated actions. Closure finger RMSE worsens
  from **4.94 to 5.58 mm**. This is an offline diagnostic, not a deployed sweep.
- **Simulation:** earlier exact command replay controls picked up in 5/5 cases;
  live VLA picked up in 0/5. The recorded commands are physically executable
  in this scene. This does not certify the scene's real-world contact model.

## Likely contributor and next discriminating experiment

The run processed only **97,944 sampled frames**, about **0.664
sample-equivalent passes** through 147,499 frames. Limited optimization is a
plausible contributor to residual motor errors. It is **not proof** that a
larger batch, a longer run or more successful episodes alone will fix this.
Unobserved phase/history dependence or other data/model issues remain possible.

Next, qualify a **bounded one-episode overfit using the native trainer**, keeping
this checkpoint's observation/action contract, saved training-stat normalization,
scene and inference settings fixed. Track generated approach/closure errors,
not flow loss alone, and require a fresh autonomous pickup with an expert
control. The prior custom single-episode success is not a qualification of this
new native recipe. No such training was launched during this diagnosis.

If the native overfit cannot reproduce one expert pickup, investigate that
small case rather than scaling data. If it passes, expand to a few episodes,
then the full split with an adequate measured optimization budget. Add properly
labeled corrective demonstrations of empty closure/misalignment before making
robustness claims. Insertion and RLT remain gated by reliable autonomous pickup.
