# Initial-wait sampling ablation

## Completed result

Both 1,000-update arms completed normally, with identical initial fixed probes.
The expert training-reset control picked up. Uniform policy collided at 903
commands without a grasp; skip-survey policy reached 1,200 commands without a
grasp. Both failed the training-control gate, so validation was **not run**.
Skip-survey's fixed validation-probe loss improved 0.004079 -> 0.002849, but
physical pickup did not follow. Charged runtime, including the preserved manual
restart, was 59.31 minutes. Final audited plots/summary are under
`results/onset_ablation_20261020_v2`. Historical launch notes below are superseded.
The next authorized recipe-scaling test is [FIVE_EPISODE_OPENHOLD.md](FIVE_EPISODE_OPENHOLD.md),
not automatic recovery collection or another large uniform run.

2026-10-20. Authorized follow-up to [the onset diagnosis](PICKUP_ONSET_DIAGNOSIS.md).
This is a small controlled continuation experiment, not an overnight retrain or
a change to the robot controller. Original results/models/data remain intact.

## Question and comparison

Does excluding ambiguous initial waiting anchors improve autonomous pickup?
Both arms start with the frozen FP32 final model from the previous campaign,
with a fresh, identically configured AdamW optimizer/scheduler. Only the eligible
training-anchor pool changes:

| Setting | Uniform baseline | Skip-initial-survey ablation |
| --- | --- | --- |
| Training parents | Same 81 | Same 81 |
| Eligible command anchors | 147,499 | 143,935 |
| Initial wait anchors excluded | 0 | 3,564 |
| Optimizer updates | 1,000 | 1,000 |
| Batch / learning rate / seed | 8 / 1e-5 / 1000 | Identical |
| Forward / trainable storage | BF16 / FP32 | Identical |
| Checkpoint selection | Final step 1,000 | Final step 1,000 |
| Live execution prefix / policy seed | 25 / 1000 | Identical |

Matching seeds does not mean identical training anchors: each pool maps the
sampler's draws differently. This is a one-seed, modest-budget pilot, not a
powered study or evidence that other causes are absent.

Before the two full arms, each variant must complete a 12-update GPU preflight.
It retains metrics/logs without another large model copy. Each full arm starts
fresh from the same frozen base, not the updated preflight weights. Its first
12 updates must reproduce the preflight's sample/loss/gradient/LR history.
This is a fresh-run reproducibility check, not a new optimizer-resume test.

## Data and actor contract

The source-bound index independently checks all 81 training NPZ hashes and
metadata against the original export receipt. It excludes only the leading
`survey` commands; a later survey command would remain eligible. Global
dataset indices, all future action chunks, T+1 observations, terminal masks,
normalization statistics and the train/validation split are unchanged. No
new videos or alternate trimmed dataset are generated.

Index-field naming caveat for this frozen version:
`first_active_requested_target` stores the source **applied** target, despite
its name. It is audit metadata only and never used for sampling or execution;
all supervision remains the original `action_applied` arrays.

All 81 leading waits are 44 commands. The maximum reset-to-first-active state
change is 3.55e-11 in stored units. Thus the first active state is practically
the reset state, but its intended future differs: the robot may start moving
immediately rather than reproducing an invisible wait timer. This is an
intended behavior of the ablation, not a reason to rebase action indices.

The actor still receives only 16 joint/finger positions, two RGB images and
the instruction. Phase labels are used only for offline index construction,
metrics and plots, never forwarded to the policy. Torque/depth remain unused.

## Metrics and evaluation

- Keep the native loss unchanged. Report both its 32-channel mean and the
  corresponding actual-8-channel evaluation mean. No loss reweighting is mixed
  into this sampling experiment.
- Use identical fixed train/validation timeline probes (eight each, not full
  validation), plus 16 training-only phase probes on four preselected parents:
  original wait reset, first active, approach, and pre-close. Log physical
  finger MAE / per-joint MAE at prefixes 1/25/50, and predicted/target finger
  profiles. Original-wait error and first-active error are separate metrics;
  neither alone establishes autonomous success.
- After both training arms finish, replay the exact expert on `train_20002`,
  then run both live VLA controls on that reset. Use actual rendered images
  and measured state, no teacher forcing or recording-state injection.
- Audit saved physical traces and video frame counts. Pickup still requires
  both finger contacts, a physical grasp, at least 20 mm actual plug lift,
  one second continuous carry and no prior drop/unsafe abort.
- Measure early TCP deviation by offline FK on separate `MjData`, never by
  forcing the executing simulator state. Nearest Cartesian approach-path
  distance distinguishes some timing drift from off-path error; it does not
  measure orientation or guarantee grasp alignment.
- Only a variant that passes its training control may enter the unchanged
  five-reset validation bank, and only if the remaining total time permits.
  The existing bank evaluator repeats its own training control; preserve
  both attempts and disclose the repetition, never substitute a favorable
  outcome. If the gate fails, record validation as **not run**, not 0/5.

The fixed held-out bank was already inspected in previous diagnostics. It is
same-scene validation evidence, not a blind test or scene/object OOD study.
No checkpoint/prefix/latent-seed selection on validation outcomes.

## Runtime, storage and safety

The detached local supervisor uses no AI/API calls, Hub upload or network.
It has a **9,000-second (2.5-hour) total hard allowance**, 600-second preflights,
2,400-second training-stage caps, and 420-second physical-control caps. A full
validation bank has its existing independent 45-minute limit and must fit
within the remaining campaign allowance. No automatic retry.

Thermal guard: stop at 85 C. Available RAM must stay above 1 GiB; the first-12
GPU check requires at least 1 GiB device headroom. Heartbeats come from actual
training status/event or stage log writes. A process-scoped helper prevents
sleep only while the supervisor is alive; it does not change permanent power
settings. Closing the chat does not provide or remove monitoring.

Preserve at least **25 GiB disk reserve**. Only two new final checkpoints are
retained, with preflight logs/metrics but no model copies. No old or new
checkpoint deletion is part of this campaign. Low space or another failed
qualification stops the comparison rather than silently reducing its budget.

## Artifacts and status

The first launch's two GPU preflights completed normally (about 78 / 77 seconds).
During its preliminary uniform continuation, free space fell enough to expose
an over-conservative budget: the trainer reserved two future checkpoint writes
even with all periodic/best snapshots disabled. The owned trainer was manually
stopped after 124 updates, before any new model/checkpoint file was created.
Its logs and original failure/manual-stop receipts remain intact. This is not
a scored training arm, nor a replaced pickup trial.

The repaired launch is **`outputs/onset_ablation_20261020_v2`**. It uses the same
model/optimizer/sampling/evaluation recipes, but final-only mode reserves one
future write; ordinary multi-checkpoint runs still reserve two. The 25 GiB floor
is unchanged. Both preflights and both full arms restart fresh. A source-bound
manual-restart ledger charges 431.39 seconds of previous launch/preparation
time against the original 9,000-second allowance, rather than granting a fresh
allowance. No automatic retry and no deletion.

This is launch/qualification status, not evidence that either full arm or
physical evaluation is complete. Read the **v2** `health.json` for current state.
The repaired v2 launch has now passed **both 12-update GPU preflights** and
entered the 1,000-update uniform baseline stage, with no reported error.
An independent CPU boundary audit passes **243 anchors across all 81 parents**:
reset, first active command and terminal command. Exported state/future actions
and terminal masks match the raw arrays at their original indices exactly.
Receipt: `outputs/onset_ablation_20261020/export_alignment_audit.json`.

- Supervisor: `scripts/run_onset_ablation.py`.
- Sampler/index audit: `learning/vla/onset_sampling.py`.
- Trainer options are opt-in in `scripts/finetune_smolvla.py`; default sampling,
  objective, checkpoint behavior and native actor contract are unchanged.
- Active campaign: `outputs/onset_ablation_20261020_v2`.
- Preserved stopped attempt: `outputs/onset_ablation_20261020`.
- Current stage/health: `health.json`; launch binding: `plan.json`;
  preflight gate: `qualification.json`; outcomes: `controls.json` and
  `completion.json`, or `failure.json` if interrupted/failed.
- Final plots/summary, generated only after completion:
  `results/onset_ablation_20261020_v2/sampling_comparison.png`, PDF and summary JSON.
- Source versions are archived by hash before launch. Historical campaigns
  continue to point to their original archived trainer, not this updated file.
- 44 selected tests pass after the disk-policy correction, including explicit
  tests that final-only mode keeps the original 25 GiB floor. Source/split
  binding, prefix masks, loss accounting, FP32 state and watchdog tests pass.
- `scripts/audit_onset_alignment.py` reproduces the CPU boundary audit, including
  instructions, without starting CUDA or training. Its source-bound receipt is
  `outputs/onset_ablation_20261020/export_alignment_reproduced.json`.

Do not call the pipeline successful just because training is finite. If pickup
does not improve, report the negative result: this small continuation did not
establish that removing waits alone fixes the task. The next data pilot should
target actual premature closure and arm misalignment with synchronized
corrective reopening/hover/re-alignment examples. Insertion, RL and sim-to-real
remain behind repeatable autonomous pickup.
