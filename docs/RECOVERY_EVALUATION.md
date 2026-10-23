# Recovery pilot: paired rollout and correction-coverage evidence

Follow-up completed: [RECOVERY_SAMPLING_EXPERIMENT.md](RECOVERY_SAMPLING_EXPERIMENT.md)
records expanded intake, equal-budget 400-update uniform/correction runs, all-six-parent
recovery-section probes and the persistent early-transit failure. The decision
below describes the earlier 16-update qualification, not the latest result.

2026-10-19. No training, scene edits, relaxed guards or robot execution occurred
during this evaluation. Baseline: original best `step_0009426_best`. Candidate:
the 16-update recovery qualification checkpoint `step_0000016_final`.

## Decision

The immediate next step is **guaranteed sampling of actual gripper-reopening
corrections**, followed by a bounded, meaningful recovery training experiment.
The tiny pilot sampled some recovery labels but no active reopening targets.
Its failure is therefore not a fair test of whether adequately trained recovery
behavior can work. It also does not establish that more training cannot help.

## Paired closed-loop comparison

Both policies were freshly evaluated on three original held-out resets in the
original source-hash-matched simulator. The recovery candidate was trained on
the newer recordings; this is an **old-domain transfer/regression diagnostic**,
not evaluation in the missing new environment. Policy seed 1000, execution
prefix five, predicted horizon 50 and all collector safety guards were fixed.
The test capped each rollout at 2,000 commands and used owned-process deadlines.

| Reset | Baseline stop command | Recovery-pilot stop command | Result for both |
| --- | ---: | ---: | --- |
| `train_20004` | 322 | 267 | Finger self-contact; no plug contact or grasp |
| `train_21010` | 322 | 396 | Finger self-contact; no plug contact or grasp |
| `train_21036` | 322 | 322 | Finger self-contact; no plug contact or grasp |

Both achieved **0/3 grasps and 0/3 insertions**. Surviving longer on one reset
did not constitute task progress. Baseline reference demonstrations were still
in `transit_pick` at command 322; intended closing begins at commands 523, 559
and 533 respectively. All model rollouts prematurely closed the empty gripper.

Fresh baseline trajectories were not bitwise identical to the October 13 saved
trajectories, although the failure/outcome/stop commands reproduced. Only fresh
paired results are used here. Same-seed stochastic inference across separate
executions is not asserted to be numerically exact. One sampling seed and three
selected resets do not establish statistical generalization.

Evidence: `outputs/recovery_transfer_eval_20261019/report.json`, per-rollout
reports/trajectories/logs, and six decoded-valid H.264 videos. All video frame
counts match executed commands plus one initial observation.

## Complete recovery-section prediction check on one held-out recording

Tested all 147 recovery commands in `train_09009`, including all 57 `reopen`
commands, using three fixed noise seeds per input and checkpoint. That gives
441 recovery predictions per checkpoint, of which 171 are reopening predictions.
Additional old/new clean and old counterfactual gripper-offset probes bring the
total to 549 predictions per checkpoint. No fault actions are used as targets.
These are recorded observations, not reconstructed new-scene rollouts and not
the complete six-parent recovery validation split.

| Metric | Baseline | Recovery pilot |
| --- | ---: | ---: |
| Five-command gripper MAE, all 147 recovery frames | 2.434 mm | 2.692 mm |
| Five-command gripper MAE, 57 reopening frames | 5.695 mm | 5.590 mm |
| Mean predicted first-target change from observed position, reopening frames | -4.598 mm | -4.513 mm |
| Expert first-target change, same reopening frames | +0.291 mm | +0.291 mm |

Negative target changes mean further closing. The candidate's small reduction
in reopening error did not correct the average wrong direction, and aggregate
recovery error worsened. This is consistent with the lack of relevant training
exposure found below; it is not evidence that the recovery dataset is useless.

On original expert images at commands 200/320, changing only measured gripper
travel by +/-3 mm yielded first-target sensitivity gains 1.022 (baseline) and
1.042 (candidate), averaged over 18 matched triplets. A state-independent 35 mm
hold target would have gain zero in this counterfactual. With the observed
opening reduced by 3 mm, the desired correction was approximately +3 mm, but
the mean predictions changed it by -0.577 and -0.276 mm instead. The state-copy
behavior persists. This is a local diagnostic, not a linear stability proof or
a physically consistent intervention.

Raw predictions and matched source/checkpoint hashes:
`outputs/recovery_transfer_eval_20261019/baseline_probes/report.json` and
`candidate_probes/report.json`.

## Actual optimizer exposure: the concrete sampling gap

Traced every logged sample in the completed 16-update pilot to its parent
recording and its entire valid future action chunk:

| Exposure | Count |
| --- | ---: |
| Sampled training windows | 64 |
| Expert anchors | 60 |
| Recovery anchors | 4 |
| Active gripper-reopening anchors | **0** |
| Windows containing active gripper-reopening targets | **0** |
| Injected-fault targets | 0 |

The four recovery anchors were insertion `align`, two insertion `withdraw`
steps, and a `rise` step after reopening had finished. The available training
subset contains 118 active reopening commands among 6,036 eligible commands;
uniform sampling in a 64-window smoke run missed them entirely.

The audit defines active reopening as a `recovery` label and an applied finger
target more than 0.2 mm above the preceding observed travel. This separates
deliberate opening from small servo settling; it is not a safety threshold.
The complete traced rows are in `sampling_audit.json` under the evaluation
output. The general loss decrease of the pilot must not be interpreted as
evidence that it was trained to reopen the gripper.

## Next controlled training experiment

1. Build an explicit correction-anchor index from training parents only.
   Guarantee at least 25% active reopening windows per batch; retain nominal and
   other valid task windows in the remainder. Keep fault-boundary segmentation.
   Labels guide sampling only; they never enter actor observations.
2. Log actual opening-target exposure, phase coverage, gripper/joint errors and
   separate held-out reopening metrics. Assert nonzero correction exposure early
   in the run. Preserve the checkpoint's normalization and the two-camera,
   16-position input contract; no torque, depth or chest input.
3. Start a new bounded run from the original best checkpoint for a few hundred
   updates, rather than interpreting a 16-update qualification as recovery SFT.
   Compare against an equal-budget uniform-sampling control if resources allow.
   Keep parent holdouts fixed and monitor clean-task retention.
4. Require clear held-out reopening improvement without unacceptable clean-task
   degradation, then repeat the same guarded old-scene comparison. The original
   evaluation failure occurs during transit, so collect or augment small early
   deviations there if that failure persists; later recovery examples alone do
   not establish coverage of those policy-generated observations.
5. Obtain the matching new source revision before claiming same-scene evaluation
   of the recovery dataset. Finish the full parent-level data/validation intake
   before a broad run. RL and physical deployment remain later gates.

No new training was started in this evaluation. A larger batch or more updates
may increase rare-example exposure, but explicit sampling makes that exposure
measurable and guaranteed.

## Artifacts and checks

Reviewed comparison plot, CSV and combined JSON:
`results/recovery_transfer_eval_20261019_v2/`. The common-prefix trajectory errors
use equal comparison lengths within each reset pair, not unequal rollout lengths.
The earlier unsuffixed plot/report is preserved as a draft.

New helpers: `evaluate_recovery_transfer.py`, `probe_recovery_behavior.py`,
`audit_recovery_sampling.py`, and `report_recovery_transfer.py` under `scripts/`.
Four new evaluation/audit unit tests and six existing rollout-contract tests
passed; modified scripts compile and `git diff --check` passes. Both checkpoint
files and all source recordings remain unchanged.
