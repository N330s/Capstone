# Historical experiment log through 2026-10-23

Superseded as an execution plan by [plan.md](../../plan.md).
Entries below preserve decisions at their original time; a heading saying
"active" or "next" is historical, not a request to rerun that experiment.

## Active: bounded native one-episode pickup test (2026-10-22)

At the user's request, launched `outputs/native_single_episode_20261022_v2`.
See [NATIVE_SINGLE_EPISODE.md](../../docs/NATIVE_SINGLE_EPISODE.md).
Native filter selects only `train_20002` without trimming/re-exporting it;
warm-start native step 12,243, retain global training statistics and public
training components. Eight-update no-weight-copy preflight, then batch 8,
2,000 updates or 30 training minutes, 45 minutes total including expert
control/checkpoint/reload/live test. One final checkpoint, no deletion/sweep.
Track generated commands in physical units; final success requires live
two-finger grasp/lift/carry in the unchanged matched scene. A bounded
independent watcher will save evidence in `results/native_single_episode_20261022_v2`.
This is memorization qualification, not generalization/insertion/RLT success.
The first attempt's monitor rejected the native one-frame history dimension
before any update. Its failure is preserved; V2 fixes only the audit check,
re-audits/reuses the successful expert control and charges prior/repair time
against the same 45-minute cap. The repaired eight-update preflight passed;
main training is active, with finite updates and 62 selected tests passed.

## Latest: native SmolVLA pickup evaluation failed the training-reset gate (2026-10-22)

Diagnosis now completed (no retraining):
[NATIVE_ACTION_DIAGNOSIS.md](../../docs/NATIVE_ACTION_DIAGNOSIS.md).
Even on correct recorded expert observations, closure finger-command errors
are 1.43-4.94 mm in the primary sample, with 4.74-8.61 mm static command-pose
discrepancies. Four live failures close empty fingers to about 3 mm, outside
the full training measured-state minimum of 18.72 mm. All 500 controller-loaded
tensors are exact; sampled normalization/timing agree; decoded-video input
and bypassing cached inference retain the errors. Native flow loss includes
24 padded dimensions and time-0.5 validation misses much larger time-0.1
velocity errors. Residual motor inaccuracy and subsequent state-distribution
shift are demonstrated; limited optimization is plausible, not the proven sole
cause. Next proposed step remains a bounded native single-episode overfit,
with unchanged normalization and generated-action/physical-pickup gates.
No new training, data changes or deployment fix was authorized/launched here.

Evaluation completed in 25m 57s: **expert controls 5/5; autonomous VLA
pickups 0/5**. Four command-limit failures, one robot-collision safety stop;
no established grasps or continuous carry. All five first-prediction repeats
were exact. Independent motion/input/video/model audits passed. Validation
was deliberately **not run**, not scored 0/5. The final checkpoint's lower
offline flow loss has not translated to successful closed-loop pickup.

Evidence: `results/native_pickup_eval_20261022_v2` (summary/CSV, actual live
frames and gripper/lift plots); actual videos remain in the campaign's trial
folders. Original first report is retained. Generated-action accuracy on
recorded observations is now audited per phase in physical units. Next
proposed work, **not launched**: a bounded native single-episode overfit/live replay-controlled gate
before another multi-episode run. Training consumed only about 0.664 sample-
equivalent passes through 147,499 frames; insufficient exposure is plausible,
not an established sole cause. Do not proceed to insertion/RLT from this result.

### Completed evaluation protocol

The clean native run completed 12,243 updates at batch 8, within its four-hour
total allowance (3h 57m 22s including qualification/repair). All logged metrics
were finite. Fixed-window validation decreased from 0.466116 to 0.004507;
this covers 160 windows across 20 episodes, not full-frame validation or robot
success. Curves/CSV and checkpoint selection are saved in
`results/native_smolvla_20261021_v2`.

At the user's request, evaluated the selected final checkpoint (step 12,243)
in `outputs/native_pickup_eval_20261022`. Fixed protocol:

- Unchanged data-matched scene; fresh exact command-replay expert controls
  on the five original training resets, then five live VLA pickup tests.
- One model, execute five actions before reobserving, horizon 50 unchanged.
  Explicit controlled numerical settings and step-keyed noise; audit the first
  repeated prediction and save raw actor inputs/fingerprints.
- Require physical two-finger grasp, at least 20 mm lift and one continuous
  second of carry without drop/safety abort. No recorded-state/image assistance.
- Only if all five training pickups pass: five fixed validation-reset expert
  controls and live policy tests. Otherwise preserve all five training outcomes
  and leave validation unrun, not scored as failures. Validation influenced
  checkpoint selection, so this is not a blind generalization test.
- Hard one-hour evaluation limit, 360 seconds/trial, existing resource guards,
  no automatic retraining, checkpoint/settings sweep, scene edits or deletion.

Saved real dual-camera rollout videos and independently audited CSV/plots in
`results/native_pickup_eval_20261022_v2`. See
[NATIVE_SMOLVLA_EVALUATION.md](../../docs/NATIVE_SMOLVLA_EVALUATION.md).
Do not infer pickup success from offline loss. Reliable pickup still gates
insertion, RLT and eventual sim-to-real work.

### Completed training launch record (2026-10-21)

See [NATIVE_SMOLVLA_TRAINING.md](../../docs/NATIVE_SMOLVLA_TRAINING.md).
The user authorized up to **four hours**, with prior attempts/qualification
charged rather than resetting the clock. Current campaign:
`outputs/native_smolvla_20261021_v2`, training in its `training` folder.
Native LeRobot 0.4.4 training loop/model/loss/sampling/optimizer/scheduler;
original cached pretrained base, 81 existing training episodes, batch 8,
20k-update ceiling with earlier budget stop. No legacy quota/sensor noise/
recovery mix, scene changes or deletion. This PC is a 4060 Ti 8 GiB.

Done: exact two-update loss/clipped-gradient/weight parity; nine sampled
raw/export state/action/mask anchors; two batch compute checks; full native
12-update CLI preflight; fresh reload of all 500 saved tensors and saved
training-stat processors. 43 selected tests pass. The real CLI caught a
recursive-empty-dictionary override retaining SO100 features; explicitly
derive features from dataset metadata before the native factory, requalified
the adapter and preserved the failed preflight. Effective state 16/action 8/
two cameras and FP32 storage/BF16 computation are logged before training.

At launch, the long run was running with finite updates. First validation after step 1:
0.466116 over **160 fixed windows across all 20 validation episodes**, not
all frames or live rollout success. Per-update loss/gradient/LR and periodic
fixed-window validation are logged. Resource/deadline guards and full native
checkpoints are active. A separate local watcher will generate CSV/PNG/PDF
curves and retained-checkpoint selection in `results/native_smolvla_20261021_v2`
when training finishes. Health/report status are in the campaign root.

Next after completion: selected-checkpoint reload, fresh matched-scene expert
controls and five training-reset pickups, then fixed held-out pickup tests.
Reliable pickup still gates insertion/RLT/sim-to-real. The historical frozen
actor's numerical-mode/call-order mismatch is not declared solved; the native
offline update/data path is independently qualified. Do not launch additional
long training or claim bug-free/generalized robot behavior from loss alone.

## Latest direction: use the public SmolVLA path; qualify repeatability and training (2026-10-21)

The user's priority is reliable SmolVLA fine-tuning, not retaining custom
training code. Prefer the official LeRobot trainer/model/processors and keep
only necessary OpenArm data/simulator and explicitly tested compatibility
adapters. Do not silently upgrade the pinned environment.

Latest evidence and next gates: [CADENCE_DIAGNOSIS.md](../../docs/CADENCE_DIAGNOSIS.md).
All 12 cadence conditions completed in 2,364.015 cumulative seconds. Expert
replay 3/3; VLA prefixes 25, 1, 5: 1/3, 0/3, 3/3. The postprocessor rejected
different initial predictions; **the cadence comparison is not qualified**.
These selected training resets do not establish generalization.

New frozen-model tests show identical loaded weights and exact within-process
repeats, but tiny fresh-render RGB differences. A fixed-input A/B test proves
those differences can change actions; it does not fully explain historical
failures. One positive-control cross-context discrepancy remains. Our prefix
execution matches native LeRobot `select_action` in all nine fixtures. Added
exact initial raw-input capture/per-chunk fingerprints without changing actor
inputs/actions; 37 selected tests pass. A one-step logging smoke passed
independent saved-input, motion, scene/command/noise and video audits (not a
pickup score). The same fixed-input/numerical-profile/call-order test repeated
exactly across two fresh processes, all 21 output arrays. Results:
`results/cadence_diagnosis_20261021_final`. No new training or held-out tests.

Next: isolate the remaining numerical-mode versus call-order difference, then a tiny
native loss/gradient/update and raw/export-timing qualification. The existing
native CLI wrapper still needs its FP32 precision adapter qualified before a
clean original-pretrained-base run. No additional corrective-data heuristics
or long run automatically. Preserve every original actor, result and failure.

### Historical storage and cadence-launch record (2026-10-20)

Storage update: at the user's request, removed four hash-identical duplicate
weight files (4.462 GiB), preserving every distinct model and all optimizer
states/data/logs/videos. Free space then was about 36.26 GiB; the earlier disk
blocker below is cleared. See
[CHECKPOINT_STORAGE.md](../../docs/CHECKPOINT_STORAGE.md) for retained-copy mappings
and exact restoration/audit receipts.

See [STANDARD_SMOLVLA_BASELINE.md](../../docs/STANDARD_SMOLVLA_BASELINE.md).
At the user's explicit request to proceed, the bounded continuation was
launched in `outputs/inference_cadence_20261020_resume_v1`. Re-audit/reuse the
first completed expert control; preserve the interrupted second attempt and
restart that condition in the new directory. Charge 218 seconds from the
original run plus current preparation/execution against the same one-hour
allowance. Same model, noise algorithm, cases, guards and pickup metric; only
supervision/monitoring and explicit continuation bookkeeping change. 47
selected tests pass. No training or validation trials are launched.
The independent bounded postprocessor was intended to generate audit/plots
in `results/inference_cadence_20261020_resume_v1`, but rejected the initial
prediction mismatch and did not create that final report. Terminal evaluation
and postprocessing status are in `health.json` and `report_status.json` in the
continuation directory. See the current diagnosis above, not this launch record.

Following the user's decision to use the established SmolVLA path, defer
corrective-mixture training until the execution and native training/data
baseline are qualified. Preserve the teacher bank and all original results.

First, an inference-only campaign was implemented/launched under
`outputs/inference_cadence_20261020`: three fixed training resets (two failures
plus one positive control), fresh exact expert controls, then prefixes 25, 1
and 5 with the same frozen five-parent actor. Same chunk size 50, actual
RGB/state, controller, collision guards and physical pickup metric. Explicit
step-keyed flow noise controls matching prediction anchors. New prefix-25
conditions are not claimed bit-exact reproductions of old sequential-noise
rollouts. One-hour cumulative cap; no automatic retries, validation tests,
training, data deletion or software upgrade. The first exact expert control
passed; a resource guard stopped the process during the second control, before
any model condition. At post-stop inspection C: has 23.27 GiB free, below the
25 GiB reserve (initially 28.30 GiB); this campaign's entire output is only
14.73 MiB. RAM is about 12.64 GiB available and GPU resources are healthy at
that inspection. The cause of the roughly 5 GiB disk drop is not established.
All owned processes have exited; no automatic restart/deletion. **Model
cadence results were not run in that initial attempt**, not failed pickups. The then-required reserve was at least 26 GiB free
to start again (free about 3 GiB or choose another approved storage location).
Preserve completed and interrupted attempts; account for the initial 217.812 s
in any bounded continuation instead of silently resetting the time allowance.
Resource monitoring now records the specific violated threshold and full
disk/RAM/GPU measurements on future stops. 44 selected regression tests pass.

Next qualify raw/export timing and the native LeRobot loss/gradient/update
path, then build a clean pretrained-base fine-tuning baseline without added
phase quotas, sensor perturbations or recovery data. Keep the FP32 optimizer
repair and explicit terminal-mask compatibility. Do not confuse the official
tutorial's 20k/batch-64 example with the paper's larger simulation budget, or
claim a hardware-fitting smaller batch is equivalent without verified
accumulation. Audit existing training data before collecting more. A new long
training run requires its own budget/resource preflight; it is not launched.

## Completed: recovery teacher qualified on all four takeovers (2026-10-20)

See [TEACHER_TAKEOVERS.md](../../docs/TEACHER_TAKEOVERS.md).
The agreed four recovery tests completed normally under
`outputs/teacher_takeovers_20261020_v2`: two failed training resets, anchors 275
and 350. **4/4 privileged teacher pickups**, 63.54-66.67 mm actual lift,
one second of two-finger carry each, no drops or robot/table contact.
All replayed prefixes have exactly zero qpos/qvel error; requested/applied
commands and timing match. The existing private-data planner opens, retreats
and re-plans pickup from the actual settled plug pose. All simulator guards,
the pickup gate, original actor and scored VLA rollouts remain unchanged.
Saved synchronized actual RGB/state/corrective actions with explicit
prefix/teacher ownership: 2,859 qualified candidate commands, 57.18 simulated
seconds. These are two training resets, not four independent held-out scenes.
No VLA inference, optimizer updates, validation rollout, insertion or RL.
This is teacher feasibility, not autonomous VLA success; VLA remains 3/5 on
the five training resets. Nothing was automatically exported into training.
The preserved initial version passed 3/4; the last case stopped on a teacher
open-in-place timeout despite fully opened fingers (one 0.1417 N pad contact).
V2 adds planner-checked horizontal disengagement with a release-only 0.5 N
teacher force ceiling. That contact clears in 0.38 simulated seconds with
0.261 N maximum observed release force. All three earlier successes reproduce
bit-identically after repair. No simulator guard relaxation or live teleport.
Independent saved-motion/labels/manifests/video audit passes; 31 selected
unit/regression tests pass. Final evidence is in
`results/teacher_takeovers_20261020_v2`; all processes finished normally.

The previously proposed next step was to verify/export qualified corrective segments with their original
prefix-replay lineage; exclude failed VLA commands from all target windows.
Predeclare a short nominal-only continuation
versus bounded corrective-mixture comparison, preserving the FP32/open-hold
recipe and nominal close/lift examples. Evaluate all five training resets
including regression controls; only 5/5 permits the fixed held-out pickup bank.
See the linked document for data-label/chunk/phase sampling requirements.
That training proposal is now deferred behind the upstream baseline work above.

## Completed: five-parent collision diagnosis (2026-10-20)

See [FIVE_EPISODE_COLLISION_DIAGNOSIS.md](../../docs/FIVE_EPISODE_COLLISION_DIAGNOSIS.md).
That diagnostic stage used no retraining, new physical rollout, controller/scene
edit or guard relaxation; the subsequent teacher tests are tracked above.
Both failed pickups first touch the plug with one finger, never establish a
grasp, and close to about 1.8 mm until finger-to-finger contact exceeds the
unchanged 5 N robot guard. Their pre-contact path distances at command 275 are
17.68 / 48.99 mm while finger travel is still 34.14 / 34.92 mm. At first contact,
hand-to-reference-plug-grasp-site distances are 31.71 / 56.36 mm versus 10.53-
13.06 mm in the successful runs. A 24.16 mm ungrasped push/lift is not a valid
pickup. Nearest-path and reference-site distances are distinct measurements.

Same-image/explicit-noise probes confirm late non-corrective finger feedback:
`train_22045` at command 500 observes 10.97 mm and predicts a first-five mean
1.40 mm; replacing only that input with 34.99 mm changes the mean to 34.07 mm.
This is offline state-channel evidence, not a deployable observation replacement.
Actual sampled open-hold augmentation covers roughly 32.94-36.99 mm; later
failed states are much farther out. One successful case also closes earlier
than its recorded expert, so early closing/chunk size alone is not the cause.
The exact source of arm drift remains unisolated; no general bug-free claim.

The proposed next implementation was four teacher takeovers on the two failed training
resets at commands 275 (pre-contact) and 350 (one-sided contact). Reproduce each
state by actual prefix-command replay, verify qpos/qvel, then re-plan safe
reopening/hover/recentering/descent from the actual plug/arm pose. Require the
unchanged physical pickup gate before recording a small correction mixture.
Do not teleport, resume a stale timed trajectory, weaken guards, leak validation
or immediately launch more training. Its implementation/status is tracked above.

Audited plots/frames/CSV/summary:
`results/five_episode_collision_diagnosis_20261020_v3`. Main FK/coverage analysis
44.08 s; 168-chunk state probe 106.80 s, normal completion within 300 s. Original
weights/data/scored rollouts remain unchanged. At that stage, 50 selected tests
passed, all diagnostic processes finished and no further training/collection
had started; the subsequent teacher recordings are tracked above.
Insertion/RL remain deferred.

## Completed: scale the successful open-hold recipe to five parents (2026-10-20)

Completed normally: 4,000 updates, equal 800 per parent; training 85 minutes,
full pipeline 114 minutes. Experts 5/5 and saved single-parent control pass.
New live training-reset pickups **3/5**: `train_20002`, `train_20042`, `train_22001`.
The other two stop on finger self-contact; heldout validation is **not run**.
Fresh reload predictions match exactly. The static launch-time notes below are
historical, not an active-job claim. The later diagnosis and teacher tests are
tracked above; no additional VLA training has started.

See [FIVE_EPISODE_OPENHOLD.md](../../docs/FIVE_EPISODE_OPENHOLD.md).
The user's agreed next experiment is launched under
`outputs/five_episode_openhold_20261020`. Fresh exact expert pickup replays and
the saved successful single-parent policy must pass before GPU qualification
and training. This launch note is not a passed qualification or pickup result.

This corrects the earlier comparison: the successful 800-update single-parent
run guaranteed 50% open-hold sampling with +/-2 mm sensor-state augmentation;
the 81-parent run was uniform without that augmentation. Both started from the
original base, not from the successful candidate. Do not attribute the outcome
to episode count alone or conclude new collection is already necessary.

Use five preselected original training parents, including `train_20002`, with
4,000 total updates and equal 800 updates per parent. Same native objective,
LR 1e-5, batch eight, FP32 expert/Adam and BF16 forward, saved normalization,
open-hold quotas and bounded sensor perturbations. Stretch the normalized LR
schedule across the longer run. Only the final inference expert overlay is
retained; no optimizer state or old-output cleanup. Inputs, data, scene, guards
and prefix 25 are unchanged at deployment. Audit every sampled index/offset.

After fresh reload verification, test live pickup on all five training resets.
Only 5/5 permits the fixed validation bank with fresh expert controls, within
the remaining budget. The detached supervisor enforces a three-hour total cap,
100-minute training-stage cap and 25 GiB disk reserve, with thermal/RAM/GPU/
heartbeat guards and no automatic retries. 47 selected pre-launch tests pass.
Live status/receipts are in the campaign; final plots are generated only after
completion under `results/five_episode_openhold_20261020`.

This supersedes the immediate recommendation to collect recovery data. First
test the working recipe at small scale; do not start insertion, RL, broader
collection or another uniform 81-parent run automatically.
Handoff qualification: the first fresh expert replay (`train_20002`) passed.
The remaining expert/control/GPU gates are underway; new training has not yet
started. Read campaign status for subsequent progress, not this static note.

## Completed controlled initial-wait sampling pilot (2026-10-20)

Completed result: both matched arms finished 1,000 updates, with all stages
exiting normally. Training-control expert pickup passed; uniform policy aborted
on robot collision at 903 commands; skip-survey policy timed out at 1,200.
Neither established a grasp. Validation was **not run**, rather than scored 0/5.
Skip-survey fixed validation-probe loss improved 0.004079 -> 0.002849 but did
not establish pickup. Total charged runtime was 59.31 minutes, including the
manual storage-policy restart. Completed plots/audit:
`results/onset_ablation_20261020_v2`; terminal status:
`outputs/onset_ablation_20261020_v2/completion.json`.
The launch-time notes below are historical, not an active-job claim.

Authorized implementation of the next diagnostic experiment is launched under
`outputs/onset_ablation_20261020_v2`. See
[ONSET_SAMPLING_ABLATION.md](../../docs/ONSET_SAMPLING_ABLATION.md).
Compare two matched 1,000-update continuations from the same frozen FP32 final
model: uniform 147,499 anchors versus 143,935 after excluding only the 3,564
leading survey anchors. All 81 training NPZ hashes and metadata are checked;
raw recordings/export/action indices/T+1 alignment and validation stay intact.
Same seed, batch eight, LR 1e-5, native loss, actor inputs and prefix 25.

Each variant must pass a 12-update GPU preflight before full training. The full
runs restart from the common frozen model, then must reproduce those first 12
updates. Log actual-eight-channel evaluation loss and separate wait/active/
approach/pre-close finger/joint errors; do not equate these with pickup success.
Only final-step checkpoints are selected, never validation-loss winners.

After training: exact expert replay, then both live training-reset pickup
controls with unchanged physical guards. Only control-passing variants may
reach the fixed validation bank within the remaining budget; otherwise record
validation as not run. Keep every attempt, including the bank evaluator's
repeated control. No favorable checkpoint/prefix/seed substitution.

The detached supervisor enforces a 2.5-hour total cap, thermal/RAM/heartbeat/
disk checks and 25 GiB disk reserve without AI/API calls or automatic retry.
Retain two new full-run final checkpoints; preflights keep logs/metrics without
model copies. No deletion is planned. Original models/results are unchanged.
38 selected pre-launch tests passed. Check `health.json`, `qualification.json`,
then `completion.json` or `failure.json` for live status; this launch note is
not evidence of completed training or improved pickup. Results/plots are
generated at completion under `results/onset_ablation_20261020_v2`.

Initial qualification: both 12-update GPU preflights passed. A separate CPU
audit checks 243 anchors across all 81
parents (reset/first-active/terminal): exported state, future actions and
terminal masks exactly match the original aligned arrays. Receipt:
`outputs/onset_ablation_20261020/export_alignment_audit.json`.

Manual storage-policy repair: the first uniform attempt was stopped after 124
updates, before any new checkpoint was written, because ordinary two-write
reservation was still applied to final-only mode. Its logs/failure/manual-stop
receipts remain under the original unsuffixed directory. Final-only mode now
reserves one future write, with the same 25 GiB floor; ordinary behavior stays
unchanged. The v2 launch restarts both preflights/arms from the frozen base,
charges 431.39 seconds of prior time against the original cap, and performs no
deletion or automatic retry. 44 selected regression tests pass after repair.
Check the v2 status, not the stopped attempt's historical healthy status.
V2 qualification update: both 12-update GPU preflights passed; the full uniform
baseline stage is underway. Neither completed pickup nor generalization is
claimed from these qualifications.

This executes the sampling-only proposal below. It does not claim initial
waits are the sole cause or start insertion, RL, sim-to-real, or new collection.

## Latest diagnosis: initial forecast error and hidden wait timing (2026-10-20)

See [PICKUP_ONSET_DIAGNOSIS.md](../../docs/PICKUP_ONSET_DIAGNOSIS.md) and the
[diagnostic plot](../../results/fp32_pickup_onset_20261020/onset_diagnosis.png).
No retraining or production-controller change. Autonomous validation remains
**0/5**; 200-command prefix interventions are not full pickup trials.

The first predicted chunk already closes prematurely on recorded expert reset
inputs: offset 24 predicts 38.44 / 38.34 mm on training/validation resets while
the expert still requests 44 mm. Thus failure does not start only after live
state drift. All 81 training parents begin with 44 almost-static survey commands;
their future targets vary with an unobserved wait timer. A single-observation
actor has no explicit phase/time/history to resolve this near-duplicate-input
ambiguity. This is a concrete supervision issue and plausible contributor,
not proof that removing survey anchors alone will fix pickup.

Replanning every command improves finger opening but leaves 43 mm final
off-path TCP error in the 200-command training-control test. Prefix 5 reduces
that final path distance to 7 mm but closes to 29.97 mm. No prefix is promoted.
Exact requested-to-applied filter reconstruction has zero discrepancy; fresh
reset state matches exactly and image differences are tiny (at most one 8-bit
level). Identical input/noise predicts identically. Offline FP16 reduces tiny-
image joint sensitivity but does not remove the early closing.

Native training loss averages 32 channels, including 24 padded ones. Real
channels still contribute 79-98% in the inspected anchors, but aggregate loss
does not establish physical accuracy. This is the pinned upstream convention,
not a newly discovered custom-trainer defect. Add physical/phase-specific
metrics to the next experiment rather than silently changing the objective.

Next proposed implementation: a controlled, sampling-only ablation excluding
initial survey anchors, preserving raw data/export/T+1 alignment/reset metadata,
with a matched baseline and fixed update budget/checkpoint rule. Qualify live
training-control pickup, then the unchanged held-out bank. Keep all guards and
validation separation. Follow with targeted recovery collection for observed
residual errors; do not launch another broad nominal-only collection or long
training run automatically. Insertion/RL remain deferred. This refines the
previous recovery-only next-step proposal below; no experiment has started.

Evidence: `outputs/fp32_pickup_onset_20261020` and
`results/fp32_pickup_onset_20261020`. Both bounded probes completed normally;
34 selected regression tests pass. Original models/data/results are preserved.

## Pickup diagnosis complete: recovery coverage and approach precision (2026-10-20)

See [FP32_PICKUP_DIAGNOSIS.md](../../docs/FP32_PICKUP_DIAGNOSIS.md).
No retraining, controller changes, scene edits or relaxed guards were performed.
Explicit same-noise/same-image counterfactuals confirm that finger-state errors
can amplify closure: at a held-out pre-grasp anchor, replacing only the recorded
35 mm finger state with the live 25.14 mm state changes the target from 35.26 to
19.37 mm rather than reopening. All 81 original training parents were hash-audited:
41,594 pre-close observations, zero below 34 mm. Missing corrective-state
coverage is a concrete problem, not a reason to collect more nominal-only data.

Recorded-input VLA commands physically pick up the training-reset plug (734
commands, 64.40 mm lift, one-second carry) but fail the validation reset on table
contact at 479, before close begins at 523. The assisted validation hand is
8.72 mm from the expert TCP, including 4.52 mm too low, with 44.69 N support
contact. Thus arm approach precision also matters; gripper drift alone does not
explain everything. These are assisted diagnostics, not autonomous successes;
the original full live policy validation remains **0/5**.

Tested normalization/action-index/prompt checks pass; no broad bug-free claim.
Results/plots/CSVs: `results/fp32_pickup_diagnosis_20261020`. Main/probe watchdogs
finish normally (297.83 s / 63.69 s); 31 selected regression tests pass. No job
remains active. Original models/data/live-trial artifacts remain unchanged.

Next proposed step, requiring the next implementation instruction: a small
training-reset recovery pilot with actual synchronized state/RGB, corrective
reopening and safe hover/re-alignment/descent. Keep validation resets separate,
the actor contract unchanged and all contact guards intact. Qualify live pickup
on the training control and fixed validation bank before another long run.
Do not automatically start broader collection, retraining, insertion or RL.

## Training and repaired evaluation complete (2026-10-20)

The corrected multi-parent SFT completed normally: **11,300 finite updates**,
90,400 sampled anchors across all 81 training parents, 3 h 50 min runtime,
4.135 GB peak GPU allocation. The original a-priori final checkpoint survives.
Fixed validation probe loss is effectively flat (0.003552 -> 0.003589); do not
claim pickup generalization or meaningful validation improvement from training.

The first evaluation was incorrectly stopped by an outer supervisor heartbeat
bug. The watchdog repair passes 20 focused tests (73 in the broader selected
suite). Recovery **completed** the fixed bank without retraining or changing
the actor/scene/metrics: **VLA held-out pickup 0/5; expert 5/5**. The separate
training-reset policy control fails too; its expert succeeds (six expert
controls pass in total). No policy trial establishes a grasp. Held-out failures:
one command-budget timeout, one penetration abort, three robot collisions.

All 12 physical conditions and 8,987 video frames are independently audited.
The seven completed trials from before interruption remain byte-for-byte
unchanged. The unfinished eighth model-loading attempt is archived and its
restart explicitly disclosed, not substituted for a scored outcome. Cumulative
active evaluation is 24.39 minutes within the original 45-minute limit; report
generation takes 7.09 seconds within its separate five-minute allowance.
The old failure report stays intact as provenance. No job remains active.

Completed repair status: `outputs/fp32_multiepisode_20261020/evaluation_recovery_001`.
Training curves exist under `results/fp32_multiepisode_20261020/training`;
audited pickup plots, videos' final frames, summary and CSV are under
`results/fp32_multiepisode_20261020/pickup`. Native videos/traces remain in the
original campaign's `pickup_evaluation` tree.
See [EVALUATION_WATCHDOG_FIX.md](../../docs/EVALUATION_WATCHDOG_FIX.md).
Next proposed diagnostic: freeze this checkpoint and compare its actions on
recorded expert observations against live rollout on the same training-reset
control and a failed validation reset. Locate the first approach/alignment and
finger-command divergence, including timing/units/normalization checks. If
recorded-input prediction is wrong, inspect supervision/training; if it is right
but live execution drifts, inspect closed-loop coverage and gripper feedback.
These are hypotheses to test, not a confirmed single cause. Do not launch more
training, insertion or RL as an automatic next action.
This section supersedes the launch-time status and disk notes below.

## Active corrected multi-episode campaign (2026-10-20, 00:34 Bangkok)

The user authorized training toward pickup generalization. The offline,
detached supervisor is launched under `outputs/fp32_multiepisode_20261020`.
The new batch-eight FP32 GPU resume gate **passed**: 12 uninterrupted updates
match 8+4 resumed updates exactly for model, Adam moments, scheduler, RNG,
sampler and update history. Conservative device headroom is 2,636 MiB.
The multi-parent SFT process has now started. Qualification plus training share a four-hour cap;
the a-priori final checkpoint then receives the same paired five-reset pickup
validation screen, with a separate 45-minute evaluation cap. Hourly logs,
disk/thermal/RAM/heartbeat guards, bounded new-run checkpoint retention and
process-scoped sleep prevention do not depend on AI/API calls or the chat.
See [FP32_MULTIEPISODE_TRAINING.md](../../docs/FP32_MULTIEPISODE_TRAINING.md).

With explicit approval, exactly nine old short-run checkpoint bundles were
permanently removed (13.4997 GiB); their logs and all protected models/data/
evaluation results remain. Receipt: `outputs/approved_preflight_cleanup_20261020.json`.
About 38.83 GiB was free before launch. No other old-output cleanup is authorized.
This section supersedes historical "no active process"/disk-status notes below.
Training alone does not establish generalization; insertion/RL stay deferred.

Early handoff checks: the first full 25-update recovery bundle completed;
200+ main updates are finite and all 81 training parents have been sampled.
After unexplained extra free-space consumption, a recorded new-run-only
retention amendment retires disposable qualification/early bootstrap bundles
after verified successors, preserving best/latest/final and all logs/hashes.
No additional old experiment is touched; reserve remains 25 GiB. The main
trainer and frozen physical pickup protocol are unchanged by this amendment.

## Current priority: pickup generalization before insertion (2026-10-20)

See [PICKUP_GENERALIZATION.md](../../docs/PICKUP_GENERALIZATION.md). Freeze the repaired
FP32 expert and prefix 25. First run one memorized positive control and five
deterministically selected held-out validation resets, with separate exact
expert replays. Score physical grasp + 20 mm lift + one second sustained carry,
not insertion success or a momentary grasp flag. Stop at this pickup gate;
do not tune insertion, retrain or sweep prefixes during the pilot. Record all
failures and report the training-reset control separately.

The existing validation split influenced original checkpoint selection; do not
call it an untouched blind test. A passing five-reset pilot must be followed by
the full validation bank and a fresh source-matched tabletop test bank before
claiming broad pickup generalization. Insertion and RL remain downstream gates.
The pilot is complete: **VLA pickup 1/5 (20%); expert pickup 5/5** on the fixed
held-out validation bank. Training-reset positive controls pass separately.
The successful held-out case lifts the plug 50.13 mm and carries it for one
second; three other cases hit the 1,200-command limit without pickup and one
hits a guarded robot collision at 685. The predefined generalization screen
fails. Insertion is not scored and remains deferred, as does RL.

All twelve physical outcomes are independently audited, along with 10,227 video
frames, source/model/split bindings and continuation provenance. One failed
trial's video is explicitly reconstructed from its complete original saved
states after an encoding timeout; no policy rerun or outcome replacement occurs.
The other videos are native recordings. All 56 selected regression tests pass.
Runtime is 32.29 cumulative active minutes within the original 45-minute budget.
Reports/plot/final frames: `results/pickup_generalization_20261019`; original
traces and manifests: `outputs/pickup_generalization_20261019`. Historical
failure files remain as provenance; the completed `report.json` is the current
terminal status. No training or evaluation process remains active.

With explicit approval, only the old `fullwindow_action_probe_preflight_v2`
12-update best checkpoint was deleted (1.50 GiB); its final checkpoint retains
identical model weights. Current VLA models/logs/evaluation traces are untouched.
About 25.56 GiB remains free, above the evaluation reserve but insufficient
headroom for a new full FP32 checkpoint/resume campaign. No further deletion
is authorized.

### Next controlled step

1. Arrange full-checkpoint storage headroom and requalify FP32 GPU optimizer/RNG
   exact resume plus current batch-size/device memory headroom. Do not reuse
   the legacy BF16 batch/resume qualification.
2. Run a bounded corrected **multi-parent** SFT preflight using the existing
   81 nominal training parents, with a new optimizer and fixed held-out split.
   The current 800-update one-parent diagnostic is not a generalization recipe.
   Keep the 16-position/two-RGB/eight-absolute-target contract and train-only
   normalization provenance. A uniform nominal baseline should precede new
   augmentation/sampling changes so its result is interpretable.
3. After qualification, run a separately budgeted broader SFT experiment and
   rerun this exact frozen pickup bank. Track approach/contact/grasp/lift/drops
   alongside loss. Do not order a larger unstructured collection or start
   insertion/RL work as the first response to this failed pickup gate.
4. Only after pickup passes the validation gates: fresh disjoint tabletop test
   bank, then targeted insertion/alignment training. No automatic long training
   campaign has been launched here.

Current repair: [TRAINING_PRECISION_FIX.md](../../docs/TRAINING_PRECISION_FIX.md).
A concrete optimizer-precision defect was found: 96.6M expert parameters were
updated directly in BF16 without FP32 master storage. In the completed 800-update
low-LR control, only 5.75% of those values changed, versus 99.53% of FP32 values.
A real-gradient GPU check confirms 5.48% versus 99.9994% changed for the same
gradient under BF16 versus corrected FP32 storage. Finite losses/gradients were
not sufficient qualification. The previous paired comparison was intentionally
stopped after its augmented condition reached 94 updates; no paired result or
task improvement is claimed. Its artifacts and original code versions survive.
Training entry points now promote trainable weights/optimizer moments to FP32
while retaining BF16 autocast. Checkpoint reload preserves saved parameter dtype;
legacy optimizer resumes fail closed, while legacy weights remain valid warm starts.
The corrected 800-update augmentation run and physical simulation tests are in
`outputs/gripper_control_fix_20261019/precision_repair`. An explicitly versioned
contact-based grasp-settle extension removes exact-target-jitter bookkeeping
failure without changing physics/contact guards; exact expert replay must qualify
it first. No recorded inputs or simulator-state forcing are used in policy trials.
Full training/source details: [GRIPPER_FEEDBACK_FIX.md](../../docs/GRIPPER_FEEDBACK_FIX.md).
The 800-update corrected run has completed; fresh-process predictions reload
exactly and all 6,400 anchors are audited to the one intended training parent.
Prefix five still closes empty and aborts at 546 commands. Prefix 25 grasps at
588, holds through 2,000 commands with zero drops, but does not insert (final
lateral errors about 13/9 mm, orientation error 8.3 degrees). Prefix 50 grasps
but later aborts on slip with three recorded drops. These are live-input trials,
not expert playback. The original-model prefix-25 matched control also completed:
928 commands, no plug contact/grasp, finger self-contact abort. The combined
training repair/augmentation improves pickup/hold at the same execution prefix
on this one training reset; it does not isolate the individual training changes
or establish insertion/generalization. All six videos were fully decoded and
audited; 46 selected regression tests pass. Reviewed evidence is saved under
`results/gripper_control_fix_20261019`. Both supervised jobs finished normally
in 43.49 minutes combined; no training/evaluation job remains active.
No candidate is promoted for broad use or RL. The original data, baseline
weights and validation split are unchanged; old trainer sources are hash-archived.
With explicit approval, only the failed prior single-episode model bundle was
removed (1.12 GiB); its videos, plots, logs and saved verification remain. Older
status paragraphs below are historical and do not describe current disk use or
active jobs. The repaired experiment's expert replay succeeds in 1,673 commands
with zero qpos/qvel error; its fresh baseline fails at 336 commands before grasp.

## Next gates after this repair

1. Completed: same-prefix original-weight control and independent video/model/
   source audit. Review the comparison plot and the prefix-25 policy video in
   [TRAINING_PRECISION_FIX.md](../../docs/TRAINING_PRECISION_FIX.md).
2. Treat prefix 25 as a **candidate**, not a universal default. Pin a small
   deterministic training/held-out reset bank before evaluating it; preserve
   parent split boundaries, source hashes and the declared grasp-metric revision.
   Report approach collision, grasp, lift/carry, drops and insertion separately.
   One memorized training reset cannot establish reliability or sim-to-real.
3. Requalify FP32 GPU optimizer/scheduler/RNG exact resume and benchmark batch
   sizes with actual device headroom. Old BF16-storage qualifications no longer
   qualify the corrected memory footprint. Arrange checkpoint disk headroom
   above the guard before that test; no other deletion is authorized here.
4. Then rerun a bounded, correctly stored SFT baseline using the existing
   nominal multi-parent training data, a new optimizer and unchanged held-out
   parents. Keep clean open/close/hold and joint-error probes alongside flow
   loss. The 800-update one-parent/sensor-augmentation recipe is a diagnostic,
   not a qualified full-dataset training recipe. First determine what corrected
   training learns before ordering another large collection.
5. If pickup is reproducible but alignment still fails, use the automation to
   collect source-matched **alignment/recovery** demonstrations from actual policy
   deviations, mixed with nominal approach/close/hold/lift examples. Synthetic
   sensor jitter alone is not physical recovery coverage. Pin the repaired
   physical-contact grasp bookkeeping in a new collector revision before new
   collection; keep the old collector snapshot immutable for old-data replay.
6. Begin RL/RLT experiments only after a repeatable supervised nominal policy and
   separately qualified reward/termination contracts exist. The current
   inference pauses simulation time: real-time action scheduling, controller
   limits and timing-matched evaluation remain separate sim-to-real gates.

## Historical diagnostic records (superseded where noted above)

Latest diagnostic: [RECORDED_INPUT_CONTROL.md](../../docs/RECORDED_INPUT_CONTROL.md).
Latest completed learning gate: [SINGLE_EPISODE_OVERFIT.md](../../docs/SINGLE_EPISODE_OVERFIT.md).
The deliberately single-training-parent test on `train_20002` completed in 37.4
minutes, with 1,327 finite updates before its 30-minute training cutoff. Expert
replay succeeds in 1,673 commands with zero qpos/qvel error. Original VLA fails
at 322 commands; the new model fails at 292, both on finger self-contact before
grasp. Both VLA conditions used live state/images, unchanged guards and identical
reset/prefix/seed. Fixed training-probe loss improves 0.002001 -> 0.001181, but
joint MAE does not improve and held-out monitor loss deteriorates substantially.
This attempt did not establish memorization; no candidate is promoted. Every
sampled index was audited (10,616 anchors, zero other-parent exposure); selected
raw/export state/action/mask equality, fresh reload, normalizers and full video
decode passed. The original trainer, weights and validation split are unchanged.
Next proposed experiment: lower warm-start LR to 1e-5 and complete a schedule
within the measured budget (e.g. 1,000 updates), with approach/open-hold physical
prediction-error gates followed by the same live rollout. Do not infer that more
data or RL is the fix yet. If accurate fitting still fails, isolate requested vs
applied target semantics before expanding collection. No further run is active;
~25.9 GiB free, so free ~1 GiB more or explicitly approve disposal of this new
diagnostic model before another weights-only run. No files were deleted. This
learning gate supersedes the recorded-gripper-only priority in historical text.

Fresh expert-command replay completes grasp/insertion in 1,703 commands with
zero qpos/qvel error. The original best VLA, supplied recorded numeric state,
avoids early gripper collapse but aborts on table contact at 497 commands;
supplying recorded RGB too aborts at 511, still before grasp. Simulator state
was never forced; all physical motion used ordinary guarded stepping. At the
assisted aborts the hand is 7--9 mm from the recorded pose and 1--2 mm lower.
Reviewed video/plot/CSV/input-readback evidence is saved. This is one held-out
reset, not a training-episode overfit test. Next isolate recorded gripper only
with live joint feedback, then test a known training episode / small overfit
case before more data scaling. No new training or model promotion occurred.

Current implementation: [RECOVERY_SAMPLING_EXPERIMENT.md](../../docs/RECOVERY_SAMPLING_EXPERIMENT.md).
Correction-aware sampling is implemented with per-batch quotas, source-bound
indices and independent raw-chunk exposure auditing. The expanded partial release
contains six training parents and all six fixed validation parents. Its new export
passed 24-segment readback: 24,532 eligible commands and 4,017 excluded faults.
Equal-budget 400-update runs both completed normally (~5.6 minutes each); fresh
reload matched saved losses exactly and preserved normalization. Actual reopening
exposure was 800 correction anchors versus 62 uniform anchors; zero fault targets.
Evaluation is complete: all 975 recorded recovery commands of all six held-out
parents were probed, with three noise seeds plus old/new clean observations
(3,111 predictions per model). Active-opening five-command gripper MAE improves
5.621 -> 4.552 mm (uniform) / 2.664 mm (correction), but original clean-probe
error worsens 0.983 -> 1.673 / 4.271 mm. Fresh paired old-scene rollouts still
fail during transit on finger self-contact: baseline command 322, correction
command 283, no grasp/insertion. Neither new checkpoint is promoted; no job is
running. Reviewed plots/CSV/summary are in `results/recovery_sampling_20261019`.
Next: preserve nominal close/hold/lift supervision through an explicitly reviewed
multi-export replay mixture, test lower corrective exposure with clean-retention
gates, and target early-transit open-hold corrections if drift persists. The
current 25% opening / 25% other-recovery mixture is not a production recipe.
Eighteen training parents remain undownloaded; ~27 GiB is free, below the 29 GiB
launch minimum for another checkpointed experiment. Plan additional storage
before expanding intake/training. Matching new simulator source and recovery
exact-resume qualification remain required before broad/unattended work or RL.

Updated 2026-10-19: recovery-aware data integration is the current step; see
[RECOVERY_TRAINING.md](../../docs/RECOVERY_TRAINING.md). The new 30-episode recovery
release keeps the baseline RGB/position actor, with depth, chest RGB and torque
retained only as raw auxiliary data. A fixed parent-level plan assigns 24 train /
6 validation episodes. The new adapter segments around injected faults so no
fault anchor or future fault target enters BC; the legacy adapter rejects this
label contract. The partial four-parent export passed readback: eight segments,
8,341 eligible commands and 1,054 excluded fault commands. A 16-update GPU pilot
completed normally in 88 seconds; eight probes on one held-out parent improved
0.02785570 -> 0.02470555. Fresh-process checkpoint reload matched the saved loss
exactly, preserved normalization and produced finite 50x8 commands. These are
pipeline qualifications, not autonomous success or full recovery validation.
The old export/checkpoints/trainer are unchanged. Nothing remains training.
The complete release, new-source exact replay and recovery-specific resume
qualification remain gates before unattended training or autonomous claims.

Subsequent evaluation completed; see [RECOVERY_EVALUATION.md](../../docs/RECOVERY_EVALUATION.md).
Fresh baseline/candidate trials on three source-matched original resets both
achieved 0/3 grasps, with finger self-contact before grasping. Full prediction
checks over all 147 recovery commands of one downloaded held-out episode still
show wrong-direction reopening on average. Crucially, the 16-update pilot's
64 sampled windows contained **zero active gripper-reopening targets**, despite
four recovery-labeled anchors; those were insertion correction or a post-open
rise. Next: guarantee correction-window sampling and log its exposure, then run
a bounded meaningful SFT comparison with clean-task retention and fixed holdouts.
The 16-update result does not show that longer recovery training cannot help.
No training was started during this evaluation. Videos, plot, CSV and raw probes
are saved; the original/new simulator distinction remains explicit.

Historical 2026-10-12 status: larger teammate dataset intake and a bounded local RTX 4060 Ti
training run are the current priority while lab access is unavailable. The lab
Docker recipe exists; image build and Linux/4090 qualification remain pending.
Trained-policy simulation rollout and exact expert replay completed earlier.
The pipeline executes; autonomous task success remains unachieved.
Team ownership and university RTX 4090 target retained.

2026-10-13 morning result: both supervised stages completed normally. The original
run reached 11,830 updates; the extension added 7,298, finishing at 19,128 updates
and saving its final report at 07:55 Bangkok. The earlier saved best (step 9,426,
64-probe validation loss 0.00355169) narrowly outscored the continuation best
(step 13,959, 0.00357015). The original final-model automatic rollout failed before
grasp on the robot-collision guard. A three-reset simulation evaluation of the
earlier best is recorded in `outputs/v2c30_best_sim_20261013`; results and assets
are assembled with `scripts/build_training_evidence.py`. This is sampled-loss
and closed-loop evidence, not full-dataset validation or a success claim.
The best-checkpoint evaluation completed on three held-out resets: 0/3 grasps
and 0/3 insertions, all ending at command 322 on `robot_collision`. Dual-camera
H.264 videos, six 16:9 figures, PDF, CSV histories and provenance are available
in `results/v2c30_training_evidence_20261013` and its ZIP. The next training decision
should follow diagnosis of this reproducible pregrasp failure, not simply more
updates or a claim that low offline loss establishes task success.

Latest authorization (2026-10-13): continue training until **08:00 Bangkok today**,
superseding the eight-hour total cap below. The original campaign retains its
existing schedule and deadline. `scripts/continue_smolvla_until.py` waits for it
to finish, requires a successful training report and matching final checkpoint,
then continues weights/optimizer/RNG/sampler in `outputs/v2c30_sft_until_0800_v1`.
This is a separate constant-final-LR tail (no learning-rate warmup restart), with
batch 16, BF16, finite checks, early/15-minute checkpoints, hourly monitoring,
sleep prevention and an independent 08:00 hard cutoff. Five minutes are reserved
for final evaluation/save. Its source dataset and original run remain unchanged.
This new continuation has CPU guard tests; live GPU qualification occurs at
startup and during its first 25 updates, so successful continuation is not yet claimed.

Original local campaign: `outputs/v2c30_campaign_v4`, targeting
`outputs/v2c30_sft_overnight_v1`. The user explicitly authorized **eight hours
total**, including prior runs and qualification; four hours remains the software
default. The first main attempt stopped after 75 updates due to a Windows status
replacement lock. Delete-sharing readers, atomic retries, nonfatal status telemetry,
early/15-minute recovery checkpoints, bounded same-run retention and monitor
shutdown/heartbeat guards are implemented. Exact fresh-process GPU resume passed;
live-monitoring/forced-lock qualification passed 131 updates and 4,721 reads with
zero read errors and one expected nonfatal lock warning. Held-out probe loss
improved 0.458730 -> 0.081399. The unattended main session resumed step 131 at
20:13:17 Bangkok, targeting 12,048 cumulative updates with batch 16 / BF16 and
a 26,602-second remaining supervisor allowance. Hard deadline: approximately
03:36:39 Bangkok on 2026-10-13. Prior runs and new qualification were deducted
from the eight-hour total. At handoff, updates beyond step 190 and monitoring
were healthy; this does not establish eight-hour stability or task success.
No dataset export, scene change or original checkpoint overwrite is required.

## Goal and scope

Controlled input swaps are now complete; see
[GRIPPER_RECOVERY_NEXT_STEP.md](../../docs/GRIPPER_RECOVERY_NEXT_STEP.md). At command 320,
changing only gripper position reduces the mean predicted opening from 34.61 to
3.01 mm; restoring that input under policy-generated observations yields 33.88 mm.
Camera changes alone yield 34.39 mm; joint changes alone yield 34.05 mm. Prefix 25
on one matched reset avoids the early abort but never grasps in 2,000 commands.
Next: gripper-state sensitivity benchmark and bounded state-noise SFT candidate,
plus 10–20 pregrasp corrective demonstrations from the collection owner. Require
actual grasp progress with unchanged guards before a new long run or RL. No new
training was launched during this diagnosis.

Measured failure diagnosis is in [ROLLOUT_DIAGNOSIS.md](../../docs/ROLLOUT_DIAGNOSIS.md).
Saved normalization statistics and round-trip decoding agree within float32
rounding. The same held-out episode replays successfully with zero qpos/qvel
error. On 45 early recorded-observation predictions, gripper MAE is 0.456 mm,
with no prediction below the demonstrated minimum; corresponding closed-loop
actions collapse to about 2 mm before grasp. This supports accumulating control
error; the source of drift (state, visual observations or execution) needs the
next controlled intervention before retraining.

Post-training evaluation is prepared in `scripts/evaluate_smolvla_run.py`; see
[POST_TRAINING_EVALUATION.md](../../docs/POST_TRAINING_EVALUATION.md). After the campaign
finishes, prioritize one simulation-only trial of the best checkpoint (latest
also selectable), with a 15-minute stage ceiling; expand to three resets if useful.
Full validation is optional: run a 16-window GPU smoke check, then all 37,104
held-out windows for best/final checkpoints and paired simulation trials (three
resets by default; 20 optional). Report command-weighted, episode-balanced and
per-episode losses; decoded action errors remain sampled. Preparation does not
launch GPU work or alter the active training. GPU execution remains unverified;
phase-specific offline metrics and untouched test resets remain pending.

Final user scenario: **plug resting on table -> grasp -> lift/transport -> insert**,
using the right arm of the bimanual OpenArm v1. Held-plug insertion is a component
curriculum, not the final task. Reliable physical pickup now precedes scaling
full-task demonstrations. See [TABLE_TO_INSERT.md](../../docs/TABLE_TO_INSERT.md).

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

The earlier two-blade pilot is a narrow near-nominal integration dataset, not
evidence of learned policy competence or generalization. The current training
integration instead uses the teammate's Type O / universal_th cable scene,
source-matched to the new recordings. SmolVLA export/readback, tiny fine-tuning,
save/reload, saved-state resume and guarded policy inference now execute locally;
the learned policy fails before pickup. No RLT module has been trained and no
physical transfer has been validated. See the current milestone below and
[VLA_ADAPTER.md](../../docs/VLA_ADAPTER.md) for the precise implemented boundary and the
retained float32 contact-sensitivity diagnostic.

See [OPENARM_PIPELINE.md](../../docs/OPENARM_PIPELINE.md) for commands and artifacts,
[SIM2REAL_CONTRACT.md](../../docs/SIM2REAL_CONTRACT.md) for calibration gaps, and
`PHYSICS_CHANGELOG.md` for contact-model changes. Preserve the straight-slot
baseline in `results/two_blade_v1_straight/` and box prototype in `baselines/box_v0/`.

## Team ownership and current priority

### Immediate local training sequence (2026-10-12)

1. Use `gdown` in a separate `.venv-download` environment to inspect/download the
   teammate's Drive release into an ignored, new `data/` directory. Do not modify
   the old four-episode source or reuse an existing export/output directory.
   Inspect total size and archive contents first; retain at least 25 GB free disk
   space after download, extraction, export and bounded checkpoints. Initial
   local free disk measured about 106 GB (decimal); this is not the lab quota.
2. Validate manifest completeness, checksums, success outcomes, action/state
   ordering, synchronized cameras and 50 Hz timing. Keep each scene/controller
   release separate. Preserve supplied episode splits; if missing or incomplete,
   create a documented selection without editing the raw source. Reject duplicate
   contents across splits and preserve an untouched evaluation bank.
3. Export with the pinned LeRobot 0.4.4 adapter and verify actual video readback,
   state/actions and terminal padding against raw episodes before training.
4. First qualify the new release with a short BF16 run using the existing
   local base model and frozen VLM. CUDA/BF16 availability has been rechecked on
   the RTX 4060 Ti (8 GB). Previous 24-update qualification peaked at 2.14 GB
   PyTorch-allocated VRAM; this supports feasibility, not a guarantee for a larger
   batch, higher camera resolution or a different training configuration.
5. Move from four spaced diagnostic windows per episode to uniform sampling of
   eligible command windows for a bounded 1,000-update baseline, held-out fixed
   loss probes, periodic checkpoints, finite-gradient checks and measured
   throughput. Compare warmed batches 4/8/16 and select the fastest safe measured
   option, retaining at least 1 GiB conservative device-memory headroom.
   Resume must bind the export/config and restore data/RNG/optimizer/scheduler;
   a larger dataset alone does not qualify the untested upstream training CLI.
6. Compare base and trained policies in the source-matched simulation with
   identical resets and collision guards. Report grasp/transport/insertion
   outcomes separately. Loss improvement is not autonomous task success; RLT
   stays behind the supervised and token-interface gates.

The user initially authorized a token-free unattended training campaign capped at
four hours, and later explicitly extended the total allowance to eight hours,
with hourly checks after preflight. All three archives were downloaded
with `gdown` (14.63 GB total); their manifests list 101 successful demonstrations
and 184,603 commands. All 101 passed strict validation (81 training episodes /
147,499 commands; 20 held-out episodes / 37,104 commands). The first 34 also
passed video readback (102 boundary/midpoint probes) and a visual stage check.
A batch-four, 100-update new-data preflight used 2.53 GB peak allocated VRAM,
with held-out fixed-probe loss 0.488800 -> 0.058982 and decoded gripper MAE
10.81 -> 3.66 mm. These are diagnostic samples, not task-success evidence.
Full-release video export and readback are complete. Batch-16 full-release preflight
completed 199 updates within its bounded qualification window, with held-out probe
loss 0.453493 -> 0.020747. The original campaign stopped on a base-model inference
feature mismatch before starting the main run. That mismatch is fixed via explicit
train-export adaptation for the untouched base comparison; saved trained-model
processors and strict finite 50x8 action checks remain intact. See
[LOCAL_FINETUNING.md](../../docs/LOCAL_FINETUNING.md).

The saved 100-update subset checkpoint reloaded in a fresh process and executed
100 commands / 2 simulated seconds on `train_20004` with no robot collision abort.
No grasp was reached in that short budget; all commands were controller-clipped
and warm prediction took about 0.253 s for a 0.10 s prefix. This is a guarded
interface check, not task success or real-time qualification. The full campaign
completed exporting. The coordinator was handed over to
`outputs/v2c30_campaign_v2/status.json` without stopping/repeating the v1 exporter.
v2 compares batches 4/8/16 on the completed subset, then uses the fastest safe
candidate for full-release preflight and training. Measurements and selection are
saved in `outputs/v2c30_campaign_v2/batch_benchmark/report.json`. Benchmark process
time is deducted from the four-hour training allowance; the learning rate stays
unchanged. GPU headroom uses peak reserved memory and device snapshots, not just
the previously measured 2.53 GB PyTorch allocation.
The completed batch comparison selected **16**: 8.19 / 10.76 / 11.18 samples/s
for batches 4 / 8 / 16 respectively. Batch 16 used 4.91 GB peak allocation and
left 1,595 MiB conservative device headroom. All three completed 40 finite-gradient
updates; 214.67 s of benchmarking is deducted from the total training budget.
This is the fastest tested safe option, not an exhaustive maximum-batch search.
The full-release preflight and long training will use batch 16 with unchanged LR.
After explicit user authorization to fix/restart, v3 reused the completed export,
readback, benchmark, 199-update checkpoint and exact expert replay. Fresh base and
preflight-model inference checks passed, and the main training launched at 18:41:36
Bangkok time, controlled by `outputs/v2c30_campaign_v3/status.json`.
The main supervisor allows 13,179 seconds (3 h 39 min 39 s; hard deadline about
22:21:15) after conservatively debiting prior training/qualification, not a fresh
four-hour allowance. Batch 16, BF16, frozen VLM, LR 1e-4, 1,000-update evaluations,
hourly health/checkpoints and a 5,709-update cosine horizon sized to measured
throughput are configured. Main output: `outputs/v2c30_sft_4h_v1`.
That v3 attempt later stopped at update 75 at 18:45:36 due to a Windows status-file
replacement lock, before its first checkpoint. It is preserved as a failed run,
not reported as running or completed. v4 fixes and qualifies concurrent monitoring,
then resumes a new run's own matching checkpoint with the extended eight-hour
total allowance. Early and 15-minute checkpoints bound progress loss on interruption.
All failed artifacts remain. These checks establish runtime integration and
offline learning, not autonomous table-to-insertion success or sim-to-real readiness.
A temporary process-scoped Windows sleep-prevention request is active until
that campaign ends (six-hour outer guard including CPU preparation); global power
settings are unchanged. That earlier four-hour limit was subsequently extended
by explicit user authorization to eight hours total for v4, as described above.

Full-window training, model/processor save, finite losses/gradients and a separate
process resume comparison were exercised on the old four-episode export. Four
resumed updates exactly matched uninterrupted weights, optimizer, scheduler,
RNG and sampler. New physical-unit action probes also execute. These are runtime
checks, not evidence that the larger-release policy can grasp/insert. The final
campaign will record full-release readback, short new-release learning, fresh-process
guarded model inference, hourly loss/health/checkpoints and a post-training
simulation rollout. No AI/API token is used by the local Python worker.

| Owner | Deliverable | Integration boundary |
| --- | --- | --- |
| Data-collection teammate | Automated, synchronized full-task demonstrations and coverage reports | Versioned episodes, timestamps, commanded/applied actions, outcomes and splits |
| Scene teammate | Improved scene, camera visibility, reset distributions and contact validation | Versioned assets/config, reproducible reset/step API, success and safety diagnostics |
| User / training workstream | Dataset validation/export, SmolVLA training and deployment, evaluation, token training and RLT | One shared observation/action decoder and reproducible experiment bundles |

The teammate's `automateData` branch now provides the reusable reset/step environment
used for the integration rollout; the older local full-task script remains separate.
Do not independently rebuild their collector or scene. Require a versioned integration
release before either team scales work. Keep the older scene as a frozen baseline;
new geometry/camera/control versions must not silently replace training assets.

Primary training target is the university **RTX 4090 (24 GB VRAM)**, not the local
8 GB development GPU. Linux is the preferred deployment target; university OS,
RAM, driver, persistent disk quota, job limits and installation permissions still
need confirmation. GPU memory is separate from RAM and disk capacity.
Source: [NVIDIA specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/).

Reserve **500 GB free SSD space** for a bounded compressed-data SFT + RLT project;
**1 TB free** gives more room for versions and experiments, but is not enough for
unlimited raw RGB archives. A **150-200 GB** allocation can support an initial
small fine-tuning experiment, not the whole planned campaign. These are planning
estimates, not measured usage. See [TRAINING_AND_STORAGE.md](../../docs/TRAINING_AND_STORAGE.md)
for assumptions, per-category budgets and retention rules.

Fine-tuning should reuse upstream LeRobot rather than reimplement its optimizer.
The substantial project work is correct data/control integration and closed-loop
evaluation. SmolVLA-based RLT is a separate research adaptation, not a trainer flag.
The next user-owned milestone is a validated dataset-to-training-to-rollout path;
RL begins only after supervised policy and token-interface gates pass.

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
mask mapping, then profile one training batch on the university RTX 4090 before an
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

The local Windows/8 GB machine remains a development host. Use an isolated,
version-pinned SmolVLA environment on the university 4090; do not inherit the old
openpi dependency pins or its LoRA requirements. Start with the selected SmolVLA
recipe's frozen backbone/action-expert training configuration, explicitly record
trainable parameters, and test mixed precision and microbatches of 1-4 before
scaling. Use gradient accumulation when needed; do not assume an upstream A100
batch size or training time applies to this machine. Profile decoding, GPU memory,
training throughput and inference with rendering enabled. Separate training and
evaluation initially to avoid unnecessary GPU contention. Confirm host access and
disk quota before downloads. The secondary pi0 path remains deferred.
Source: [openpi requirements and training guide](https://github.com/Physical-Intelligence/openpi).

Use SmolVLA as the deliberate first baseline; do not switch to pi0 merely to match
the RLT paper before this pipeline is working.

### 5B. Build a small, replayable dataset before scaling collection

The collection teammate owns new demonstrations. Request 5-10 synchronized
full-task episodes first, then approximately 50 for the first learning baseline,
expanding toward 100-300 only after coverage and policy evaluation. These counts
are experiment budgets, not precision-insertion sample-efficiency guarantees.
The old held-plug pilots are adapter fixtures/component benchmarks only; they
cannot teach table pickup. Freeze the data contract in
[TRAINING_AND_STORAGE.md](../../docs/TRAINING_AND_STORAGE.md) before bulk export.

The original 20 pilot episodes and a separate ten-episode varied batch are
collected. The latter varies approach distance and small angles as well as Y/Z;
it has no recoveries. Visual inspection flags limited slot visibility at contact.
See [DATA_COLLECTION.md](../../docs/DATA_COLLECTION.md) before scaling. Next consider
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
The supervised VLA should attempt the entire table -> grasp -> transport -> insert
task; report held-plug insertion separately as a component benchmark. Do not omit
precision demonstrations on the assumption that RL will solve
a behavior the VLA never learned.

Start with one instruction and one task. This tests visuomotor adaptation, not
language generalization. Add multiple meaningful instructions/tasks only after
this baseline works, retaining earlier tasks during subsequent training.

### Gate 5

Dataset replay passes; the policy reaches the precision phase consistently from
table starts. Report full-task success, pickup success, handoff frequency and
insertion success conditional on handoff, plus boundary failures, latency and
forces. A useful initial target is >=80% near-nominal full-task success; this is
a project gate, not a paper result. If it fails due to perception
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

Train the token module on training episodes and validate on separate validation
episodes; do not train on the untouched evaluation set. Confirm
that reconstruction gradients do not alter the VLA and that token training leaves
reference actions unchanged. Check whether the representation varies across
visually distinct approach/contact states; low reconstruction loss alone does not
establish control usefulness.

Create one versioned inference bundle containing checkpoint, token module,
normalization, camera/action schema, and adapter source revision. Any change to
this bundle invalidates cached features and replay entries.

**Gate 6:** reproducible feature extraction, held-out token checks, unchanged VLA
reference behavior, and an inference latency report. Freeze this bundle before
online RL. SmolVLA feature selection and OpenArm control make this an adaptation,
not an exact reproduction. Pin the paper version and document deviations.

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

Evaluate the entire table -> grasp -> transport -> handoff -> insertion trajectory.
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

Add polarized geometry, unplugging, alternate fixtures, and multiple
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

2026-10-12 local campaign preparation: the full-window SFT trainer and a bounded
token-free supervisor are implemented. Data intake rejects unsafe archives,
preserves raw manifests, creates an explicit episode holdout, and checks the
existing dataset contract before export. The supervisor covers child startup,
training, evaluation and checkpoint writes; a separate watchdog terminates that
child at the deadline. Hourly monitoring is performed locally, not by automatic
ChatGPT messages. The campaign is staged while video export/preflight are pending;
completion and task-success claims require actual reports.

Historical milestones below retain their original evidence and limits.

2026-10-08 lab setup preparation: Dockerfile, GPU Compose service, critical-package
constraints, non-root/read-only runtime, persistent storage mounts and an offline
infrastructure preflight are implemented. Compose configuration parses locally.
A Windows-only diagnostic passes version/import/dependency, CUDA BF16 and PyAV
checks; TorchCodec, Linux EGL and container mounts were skipped, so it is NOT a
full container pass. The local Docker Linux engine is not running: the image has
not been built or GPU-tested. See [LAB_SETUP.md](../../docs/LAB_SETUP.md) for the host
checklist, bundle transfer, build, no-skips preflight, training qualification and
detached-job commands. Qualify the Linux image and short training run on the lab
host before full-window training or accepting any 4090 throughput estimate.

2026-10-04 closed-loop integration completed:

- Loaded the saved update-24 model and upstream processors, using only cameras,
  bimanual state and instruction; decoded eight physical-unit right-arm targets.
- Used a separate `automateData` checkout at
  `b7297ba82701fe7e7814c32ea617a85f832b74a4`, matching all ten recorded source
  hashes, workspace, robot XML and runtime scene/control contract. Reset state
  matches the held-out recording exactly. The teammate's scene is not modified.
- With a five-command / 0.10 s prefix, the learned policy executed 12 commands
  (0.235 s) before a finger/socket-frame collision triggered the existing guard.
  Peak unwanted robot contact: 65.13 N; no grasp. All 12 targets were clipped by
  existing bounds/rate limits. This is an engineering-path pass, NOT task success.
- A 100-command expert replay from the same reset matches recorded qpos/qvel
  exactly, with zero unwanted robot contact. It is a two-second control check,
  not a new full-episode success run or an untouched policy benchmark.
- Warm inference averaged 0.304 s, exceeding the 0.10 s prefix. Simulation pauses
  during prediction; real-time serving is not qualified. No hardware execution.
- 42 regression/provenance tests pass. Reports, commands and video:
  [SIM_POLICY_ROLLOUT.md](../../docs/SIM_POLICY_ROLLOUT.md).

**Next user-owned implementation: a configurable full-window supervised trainer
and repeatable evaluation, before RLT.** Exercise the mask-compatible upstream
trainer over all training frames, including the initial stationary/table-start
phase (the tiny diagnostic only used 12 spaced windows). Add process-level resume
qualification, regular held-out loss checks and bounded checkpoint retention;
run a short budgeted training experiment before an extended university run.
Keep `train_00003` validation-only and train-only normalization. Re-evaluate the
same smoke reset first; then compare base/trained/expert policies on a separately
reserved reset list. Profile the 4090 and design the policy-worker/controller
handoff for measured latency without hiding it by blindly extending contact-phase
prefixes. Obtain approximately 50 varied full-task demonstrations before claiming
a useful baseline. Only after supervised task competence and the feature/token
gates pass, begin insertion-only SmolVLA RLT adaptation. Sim-to-real remains later.

The entries below are historical progress notes; pending claims describe their
respective dates, not the latest status above.

2026-10-04 qualification completed: 24 stochastic training updates across 12
windows from all three training episodes, evaluated on four windows from the
held-out episode. Mean flow loss improves 0.3235 -> 0.2456 (train) and
0.3819 -> 0.2922 (validation). After restoring update-12 weights/processors,
optimizer, scheduler, RNG and sample position, three updates match the
uninterrupted branch exactly. Evidence: `results/smolvla_training/qualification_v2.json`.
The sampler now batches Hx8 action samples correctly before preprocessing; this
fix is also applied to the original smoke script. This verifies saved-state
reconstruction on the tested runtime; a separate-process resume launcher and
closed-loop task performance remain to be tested. Next implementation target:
policy inference and short-prefix execution in the teammate's matching v3 scene,
with phase/safety/latency reporting. No RLT or physical execution has started.

2026-10-04 training progress: real pretrained SmolVLA passes a three-update,
batch-one GPU smoke test on the local RTX 4060 Ti with BF16 autocast. Fixed-sample
loss falls 0.2312 -> 0.1252; model weights reload exactly, numeric processors
reload consistently, and postprocessed inference returns finite 50x8 commands.
Peak PyTorch-allocated GPU memory is 2.09 GB. Evidence:
`results/smolvla_training/smoke_v3.json`. Added the explicit LeRobot 0.4.4 padding
mask spelling shim; no installed upstream source was edited. This qualifies the
small diagnostic loop, not full trainer/resume, held-out competence or deployment.
Next: multi-sample tiny-fit + validation + resume qualification, then a rollout
through the teammate's matching v3 scene/controller. RLT remains later.

2026-10-03 runtime progress: real LeRobot 0.4.4 export and readback now pass in
isolated `.venv-smolvla` (CPU PyTorch; original simulator environment unchanged).
Fixed upstream codec identifier to `h264`. Successful export:
`data/collection_smolvla_smoke_v2`, 7,251 rows, approximately 38 MB. Twelve actual
readback samples cover first/middle/terminal commands in all four episodes;
states/actions, task text and padding masks pass, both camera streams decode.
See `results/smolvla_export/readback.json` for package versions and compression
errors; lossy image quality still needs task-specific visual review. Next is the
CUDA model/processor forward-backward and save/reload test, not another data-only
adapter test. No model training, resume qualification or policy rollout is claimed.

2026-10-03 integration update: teammate `automateData` source inspected; v3
data-order checks and explicit smoke split selection implemented. Four local
full-task recordings pass source validation (3 train / 5,501 steps; 1 validation /
1,750 steps). The manifest's missing `train_00005` is explicitly excluded only in
the smoke selection, not silently skipped or removed from the original manifest.
Ten adapter tests pass. Real LeRobot export/readback, model training and save/reload,
and a matching-v3-environment rollout remain the next acceptance steps; do not
claim the complete pipeline works from source validation alone.

First training-side slice implemented 2026-09-27: strict raw SmolVLA data adapter,
split-separated optional LeRobot export, and local-only dry-run/smoke launcher.
LeRobot 0.4.4 is the release target; university dependency and checkpoint hashes
are not yet locked. Existing varied pilot passes source validation. Export writer
and launcher have local unit coverage, but real LeRobot export/model runtime,
resume and closed-loop evaluator remain unvalidated/unimplemented respectively.
See [SMOLVLA_TRAINING.md](../../docs/SMOLVLA_TRAINING.md) for commands and exact boundaries.

Immediate user-owned milestone: **SmolVLA dataset validation, university training
smoke test, and closed-loop evaluator**. The teammates' phase-aware full-task
environment and synchronized pilot are integration dependencies. Continuous physical
table pickup -> transport -> insertion now passes six cases using a socket rotated
90 degrees about its unchanged axis/location. Synchronizing joint interpolation
fixed a transport-path collision; the measured acquired grasp is retained through
insertion. No reset or plug teleport bridges phases. Nominal replay is exact after
matching command-boundary physics refreshes. Earlier failures remain preserved.
These are scripted engineering validations, not a generalization result or a
completed full-task learning dataset. See [FULL_TASK_DEMONSTRATOR.md](../../docs/FULL_TASK_DEMONSTRATOR.md).

Goal: **a checkpoint-specific VLA data/inference round-trip and tiny supervised
smoke test**. Use existing held-plug data for adapter tests while waiting for the
first full-task release; do not present those tests as full-task policy training.

Current priority: SmolVLA, not pi0. The ten-episode varied collection and camera
review are complete. Before broad collection, improve or validate precision-view
observability with the scene teammate, implement the SmolVLA adapter, and profile
a single training batch on the university 4090. No SmolVLA install, checkpoint
download, or overfit run is
claimed complete. Preserve the pi0 work below as a secondary track.

Historical secondary work: pi0 base / JAX LoRA was the first candidate; source is
pinned to `215abfb217dbac7d5f1273282331b9b1866c0479`. The OpenArm numeric/camera
adapter and full-window raw-pilot path are implemented and locally tested.
Checkpoint weights and training dependencies are not downloaded. Local hardware
is an 8 GB RTX 4060 Ti on Windows; the new training target is the university 4090,
with its OS still unconfirmed. Preserve the retained float32 replay diagnostic;
test SmolVLA's own runtime transforms and decoded-command physics rather than
assuming the pi0 adapter validates a different model. This partial work is not
milestone completion.

1. Agree the teammate data/environment contract and obtain a 5-10 episode sample.
   Pin SmolVLA checkpoint/license and LeRobot/PyTorch revisions; record university
   host access, OS, VRAM, RAM, disk quota and cache paths. Check embedding access for later RLT now,
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
   Add a launcher with configurable data/output roots, train/validation splits,
   precision, microbatch/accumulation, seed and checkpoint retention. Prove
   interrupted-job resume including optimizer, scheduler, RNG and data position.
4. Add a seeded closed-loop evaluator comparing expert, base VLA, and fine-tuned
   VLA with identical resets and safety limits. Record latency, chosen chunk
   horizon/execution prefix, success, contacts, and failure phase. Separate smoke
   test rollouts from an untouched benchmark of at least 100 held-out resets.
5. Train a first full-task baseline on approximately 50 demonstrations from the
   collection teammate. Only after interface checks pass, request expansion toward
   100-300 demonstrations with
   reachable pose/angle variation and genuine expert recoveries. Split by episode
   and scene configuration, refit normalization on training only, and evaluate
   near-nominal versus boundary/generalization performance separately.
6. After Gate 5, implement feature extraction and token reconstruction (Gate 6),
   then bounded insertion-only chunk replay and actor/critic training (Gate 7).
   Test sparse rewards, duration-aware targets, frozen gradients, safety aborts
   and resume before long RL runs. Start with one rollout environment sharing a
   frozen policy service; profile throughput before adding workers. Evaluate
   final table-to-insertion handoff before claiming end-to-end improvement.

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
| University training environment | VLA smoke test | RTX 4090 / 24 GB VRAM; OS, RAM, access and persistent storage quota unconfirmed; Linux preferred |
| Storage and retention | Dataset transfer / training | 500 GB free SSD working allocation, 1 TB preferred; compressed video and bounded replay/checkpoints |
| Teammate integration release | Full-task training | Versioned scene/reset-step API and 5-10 synchronized episodes; then roughly 50-episode baseline |
| VLA checkpoint/backend | University baseline | SmolVLA primary; local revisions/runtime pinned and tested. University dependency lock remains to qualify; pi0/JAX secondary |
| Token feature hook and implementation fidelity | RLT training | Audit selected model against paper Section IV |
| Action horizon/execution prefix | Closed-loop VLA evaluation | Measure latency and contact response first |
| Physical deployment | Real-robot runs | Later calibrated unpowered fixture |

## Research references

SmolVLA, RTX 4090 and RLT abstract rechecked 2026-09-27; existing method details
reference paper v1. Paper v2 is available: audit version differences before
implementation and pin source/code revisions, rather than following moving main.

- [SmolVLA training guide](https://huggingface.co/docs/lerobot/smolvla).
- [RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/).
- [RL Token paper version history](https://arxiv.org/abs/2604.23073).

- [Physical Intelligence RLT overview, March 19, 2026](https://www.pi.website/research/rlt).
- [RL Token paper, arXiv v1, April 24, 2026](https://arxiv.org/html/2604.23073v1).
- [Official openpi models and fine-tuning guide](https://github.com/Physical-Intelligence/openpi).
- [Official OpenArm MuJoCo assets](https://github.com/enactic/openarm_mujoco).

Community RLT implementations may be inspected later, but their names are not
evidence of paper fidelity or official support. Verify any selected implementation
against the primary paper and record deviations.
