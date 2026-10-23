# Evaluation watchdog repair (2026-10-20)

## Completed result

The repair and resumed evaluation completed successfully, without retraining.
All **12 conditions / 8,987 video frames** passed the independent physical-trace,
video and provenance audit. The seven original completed conditions' file hashes
are unchanged. Active evaluation totals **24.39 minutes**, below the original
45-minute cap; recovery/reporting took 9.36 minutes, with only 7.09 seconds of
separately budgeted report generation. No training/evaluation process remains.

The software-recovery result passes; the **policy task gate fails**:
held-out VLA pickup **0/5**, held-out expert pickup **5/5**. The separate
training-reset policy control also fails, while its expert succeeds (all six
expert controls pass). None of the six completed policy trials establishes a
grasp. Held-out failures are one command-budget timeout, one penetration abort
and three robot collisions. Insertion is not evaluated and RL stays deferred.

Audited summary/CSV/plots: `results/fp32_multiepisode_20261020/pickup`.
Training curves: `results/fp32_multiepisode_20261020/training`.
Current terminal recovery status:
`outputs/fp32_multiepisode_20261020/evaluation_recovery_001/completion.json`.
The original outer failure report remains preserved as historical evidence.

## Cause and fix

The 11,300-update FP32 multi-parent trainer completed normally. Its outer
supervisor then killed a progressing evaluation after about 15 minutes.
The supervisor initialized heartbeat age from total stage duration even when
no heartbeat source was configured. The evaluation stage did not provide a
source, so its duration inevitably tripped the 900-second stale-heartbeat guard.
This was a supervisor bug, not evidence of a training/model crash.

`learning/vla/job_monitoring.py` separates heartbeat freshness from wall-clock
deadlines. Evaluation watches `progress.json` and native per-condition logs.
Unmonitored short report stages use their hard deadline without a fake heartbeat
timeout. Missing configured heartbeat output still gets a finite startup grace;
a resumed job does not inherit hours of heartbeat age from its prior session.
Actual stale output still triggers the guard. Owned-process-tree termination,
hard deadlines and thermal/RAM/disk guards remain enabled.

`scripts/run_fp32_multiepisode.py` uses this helper for future campaigns.
The completed training is **not** restarted to exercise the fix.

## Recovery of the existing evaluation

`scripts/resume_fp32_evaluation.py` is scoped to
`outputs/fp32_multiepisode_20261020`. The original failed campaign completion,
status and launch plan stay unchanged. Read
`evaluation_recovery_001/health.json` and its terminal `completion.json` for
the repair's current state; the old outer completion remains historical evidence.

Seven completed conditions (four expert, three policy) are retained exactly,
including the policy failures. Their physical traces and all video frames were
audited before recovery, then SHA-256 bound in `external_interruption_001.json`.
Continuation re-audits them and does not rerun them. The source scene, final
11,300-update checkpoint, reset bank, seed, execution prefix, metrics and guards
are unchanged. No training or checkpoint sweep occurs.

The unfinished eighth condition, `train_21022_policy`, had an empty output
folder and a model-loading log, with no saved terminal trace or outcome. Its
original folder/log were moved (not deleted) into
`pickup_evaluation/interrupted_attempt_001`. The restart is explicitly disclosed
and separately preserved; it is not a replaced completed outcome. This narrow
recovery refuses to restart a condition with saved physical/terminal artifacts.

The original external interruption is conservatively charged **910 seconds**,
computed from the immutable supervisor time ledger plus timestamp slack.
Recovery preparation adds one second: prior debit **911 seconds**, leaving
**1,789 seconds** of the original 2,700-second active evaluation allowance.
Idle time between sessions is not active evaluation time. The child and outer
recovery supervisor both enforce the remaining allowance. Report generation
has a separate cumulative five-minute cap. No extra training allowance exists.

`learning/vla/evaluation_recovery.py` and the independent report auditor verify
the original time sources, completed artifact hashes, exact outcome prefix and
interruption/continuation provenance. Original source versions are hash-archived.

## Evidence outputs and limits

Training plots are now generated **before** resuming simulation, so a future
simulation failure cannot hide already completed training results:

- `results/fp32_multiepisode_20261020/training/training_validation.png` and PDF.
- `.../training/training_summary.json`: finite training summary and parent counts.
- After all trials and independent audits: `.../pickup` holds the pickup plot,
  condition CSV, summary and final policy frames.
- Native videos and full traces remain in the original `pickup_evaluation` tree.

The initial recovery check passes 20 focused tests; the broader selected suite
also passes **73 tests**. Cases include fresh progress
after more than 15 minutes, truly stale progress, resumed startup with old files,
no configured heartbeat, conservative cumulative budget, and tampered source/
completed-artifact rejection. These verify the supervisor repair, not pickup
success. The final result above uses the complete original validation
denominator, never a selected subset.
