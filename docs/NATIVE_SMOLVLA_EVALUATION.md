# Native SmolVLA online pickup evaluation

2026-10-22. Campaign: `outputs/native_pickup_eval_20261022`.
Completed in 1,557.156 seconds (25m 57s), with all ten planned training-group
trials audited. **Expert controls: 5/5 pickups. Native VLA: 0/5 pickups.**
The conditional validation bank was not run, not scored as failures.
Selected regression checks: 54 passed across the evaluation gate, native
adapters/precision, cadence, physical pickup/grasp, rollout provenance,
resource guards, runtime I/O and source archives. These tests validate
software checks, not autonomous task success.

## Results and next decision

| Training reset | VLA outcome | Simulated seconds | Maximum lift (mm) |
| --- | --- | ---: | ---: |
| train_20002 | Command limit, no grasp | 24.00 | 0.942 |
| train_20003 | Robot collision, safety stop | 1.74 | 0.000 |
| train_20042 | Command limit, no grasp | 24.00 | 0.000 |
| train_22001 | Command limit, no grasp | 24.00 | 0.000 |
| train_22045 | Command limit, no grasp | 24.00 | 0.001 |

Every policy trial had zero established grasps and zero continuous carry.
`train_20002` reached one-finger contact; the other four never contacted the
plug with a finger. The collision trial peaked at 11.555 N unwanted robot
contact, above the unchanged 5 N guard. It is a policy failure, not an
evaluation-process crash. All five initial prediction-repeat differences were
exactly zero. No source-change/resource guard or trial timeout stopped the job.

The first reset gives a concrete coordination failure: the model first
commanded finger travel below 20 mm at command index 175, with hand-to-plug
grasp-point error 30.23 mm. Expert command replay first crossed that same
command threshold at index 572, with 4.41 mm error. Later the VLA physically
closed below 10 mm travel while still 84.90 mm from the grasp point. These are
observed traces, not proof of a particular loss/architecture bug or a proposal
to add a privileged gripper heuristic.

Training was numerically finite but processed 97,944 sampled frames
(12,243 updates x batch 8), about 0.664 passes through the 147,499-frame
training split. This is sample-equivalent exposure, not the number of unique
frames seen or proof that longer training alone will fix pickup.

The recorded-observation/generated-action audit is now completed; see
[NATIVE_ACTION_DIAGNOSIS.md](NATIVE_ACTION_DIAGNOSIS.md). It demonstrates
motor errors even on expert states and subsequent empty-gripper states outside
the recorded training range; checkpoint/processors/caching checks narrow the
cause without declaring the pipeline bug-free. Next, qualify a
bounded one-episode overfit using the same public native trainer and a fresh
live expert-controlled pickup before scaling back to all episodes. Do not
declare the training/inference pipeline bug-free, generalization demonstrated,
or insertion/RLT ready from this result.

Final readable evidence: `results/native_pickup_eval_20261022_v2`.
The original report is preserved in `results/native_pickup_eval_20261022`;
V2 only reserves plot-heading space and repeats the same independent audits,
with identical model/trial outcomes and no additional inference.

## Model and scope

Evaluate the native LeRobot 0.4.4 checkpoint at
`outputs/native_smolvla_20261021_v2/training/checkpoints/012243/pretrained_model`.
Selected before online testing by lowest fixed-window validation loss among
retained checkpoints (0.004507, 160 windows spanning 20 validation episodes).
Weights SHA-256:
`0cefdc6ffc0c634f17fe35a94d3940b7f15de2833c8e3142837b5a1a5e7df1de`.
No overlays, further training, checkpoint search or policy assistance.

The simulator is the unchanged dataset-matched snapshot, not an assertion that
the latest GitHub branch has identical physics/assets. Reset metadata, scene,
robot settings, sources and commands are bound in `plan.json`. OpenArm V1:
right arm active, left parked; two live RGB images, 16-value measured robot
state and instruction. Actions are eight absolute right-arm/finger targets.
No privileged plug pose, phase, torque, depth or recorded observations enter
the actor. Each action goes through live simulation physics.

## Predeclared protocol

1. Fresh command replay controls for `train_20002`, `train_20003`,
   `train_20042`, `train_22001`, `train_22045`. Require exact recorded joint
   positions/velocities and successful physical pickups before policy trials.
   Command replay uses `env.step`, not recorded-state playback.
2. Five fresh live model trials on those same resets. Predict horizon 50,
   execute prefix five at 50 Hz, then reobserve. One configuration only.
3. Only with all five live training-reset pickups passing: five fixed
   validation-reset controls, then five policy trials. The validation bank is
   evenly spaced across sorted original validation parents. Its outcomes did
   not choose the checkpoint, but offline validation did: not a blind test,
   out-of-distribution result or sim-to-real demonstration.

Pickup requires both finger contacts, a physical grasp, at least 20 mm actual
plug lift and at least one uninterrupted second of carry, without a prior
drop or safety abort. Carry need not be stationary. No insertion score is
claimed. Failed training trials do not trigger retries; finish all five for
evidence and skip the conditional validation bank if the gate fails.

## Repeatability and resource controls

Policy uses explicitly controlled numerics: deterministic PyTorch algorithms,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN benchmark off, matmul TF32 off,
BF16 forward with saved FP32 trainable storage. Flow noise is step-keyed
(`1000 + executed_command_index`). Repeat the first prediction once with
identical live inputs and noise before any command; max difference must be
at most 1e-6. Save raw initial actor inputs and per-chunk fingerprints.
This checks within-trial repeatability, not universal cross-render numerical
identity or a resolution of the older actor's cross-context discrepancy.

Hard campaign limit: one hour including setup. Per-trial timeout: 360 seconds;
maximum 1,200 executed commands (24 simulated seconds). Freeze model/source
hashes and stop on changes. Keep disk free >=25 GiB, RAM available >=1 GiB,
GPU free >=512 MiB, temperature <85 C. Kill only owned process trees on
timeout/guard failure. No automatic restart, deletion or budget reset.

## Evidence and interpretation

- `plan.json`: checkpoint, reset/input/source bindings and all planned cases.
- `health.json`, `progress.json`: live resources and completed audited trials.
- Trial folders: raw actor inputs, trajectory, diagnostics, report and actual
  side-by-side camera `rollout.mp4` (50 fps, one frame per simulation step plus
  initial frame). Inference pauses are omitted: videos do not prove real-time
  inference or hardware readiness.
- Terminal `report.json` or `failure.json`; `report_status.json` tracks the
  independent report watcher.
- `results/native_pickup_eval_20261022_v2`: independently re-audited summary/CSV,
  physical lift/gripper traces and initial/middle/final live-frame contact sheet.

Unrun or interrupted cases remain unscored, not zero-success trials. Report
training resets separately from validation resets. Five trials are a pilot,
not a statistically strong reliability claim. Preserve failures and controls;
use failure timing/contact/lift evidence to decide the next experiment.
