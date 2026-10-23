# Recovery-aware SmolVLA training

2026-10-19. The actor remains two RGB views plus 16 joint/gripper positions,
predicting 50 absolute eight-dimensional right-arm targets. Chest RGB, all depth
streams, all torque arrays, phases, fault labels and simulator diagnostics are
excluded from actor observations. Raw recordings are retained unchanged.

Follow-up: [RECOVERY_EVALUATION.md](RECOVERY_EVALUATION.md) records paired old-scene
rollouts, full held-out recovery-section probes, and the finding that the tiny
pilot sampled no active reopening targets. Correction-aware sampling is the
next training priority; four generic recovery anchors did not exercise reopening.

Latest follow-up: [RECOVERY_SAMPLING_EXPERIMENT.md](RECOVERY_SAMPLING_EXPERIMENT.md).
Selective intake now includes six training and **all six planned validation**
parents. A new export passed 24-segment readback (24,532 eligible commands;
4,017 faults excluded). Equal-budget 400-update uniform/correction runs both
completed, and both passed fresh-process checkpoint reload with unchanged
normalization. Their raw-chunk audits counted 62 versus 800 active reopening
anchors and zero fault targets. Eighteen training parents remain undownloaded;
the four-parent qualification and the 26-missing count below are historical.
Full recorded recovery-section probes and transfer interpretation are documented
with the new experiment, not inferred from sampled flow loss.

## Data boundary

`learning/vla/recovery_data.py` validates parent episodes through the existing
strict state/action/timing/scene adapter, then validates every action label
against both manifest and episode metadata. Only `expert` and `recovery` commands
are eligible. The legacy reader now fails closed on recovery-labeled recordings.

Each uninterrupted eligible interval becomes a virtual training episode. For
example, in `train_09000`:

```text
raw commands:  [0,484) expert | [484,616) FAULT | [616,2472) recovery + expert
BC segments:   [0,484)                         [616,2472)
```

No rows are spliced together across the fault interval. A horizon-50 sample at
raw command 483 has one valid target, followed by 49 repetitions of that valid
target with padding masks enabled. The next BC segment begins with the actual
recorded observation at raw command 616, not an interpolated or reset state.
Segment boundaries are loss/memory boundaries, not claimed task successes or RL
terminal events. Keep complete original trajectories for later RL work.

The split plan is deterministic and fixed before selecting local samples:
`outputs/recovery_pipeline_20261019/parent_splits.json`. It assigns 24 parents to
training and six to validation, with two validation parents per recovery type.
Every segment inherits its parent's split. The old validation set is untouched.

## Local qualification subset

The intentionally partial local view contains training parents `train_09000`,
`train_09100`, `train_09200` and held-out validation parent `train_09009`.
These four parents contain 9,395 commands. Excluding 1,054 fault commands leaves
8,341 eligible commands: 6,036 training and 2,305 validation, in eight segments.
The 573 recovery commands remain eligible. This is pipeline qualification, not
coverage of the full recovery validation split or an autonomous success result.

Files:

- Immutable hard-linked view: `data/recovery_pipeline_20261019/qualification_view`.
- Segment-separated LeRobot export: `data/recovery_pipeline_20261019/qualification_export`.
- The export receipt uses the distinct contract
  `openarm-v1-smolvla-recovery-segments-v1`; the old trainer rejects that contract.
- `scripts/verify_recovery_export.py` compares real decoded export samples with
  source values, checks segment-end masks and all eligible source intervals, and
  verifies that no auxiliary channel was exported as an actor feature.

`scripts/prepare_recovery_data.py` makes new split plans/views/exports only. Full
release preparation refuses missing parents. `--allow-partial` is explicit and
records all excluded missing parents in the view. Source manifests are never
edited. Further selective downloads can use
`scripts/inspect_drive_recovery.py --quick-open --episodes <parent-name>`.

## Training qualification

`scripts/finetune_recovery_smolvla.py` is a separate recovery-contract entry point
based on the existing stable trainer. The original source-bound trainer and
its checkpoints are unchanged. The recovery trainer inherits the warm-start
checkpoint's input/output normalization, rather than silently changing its
coordinate system or using validation-derived statistics. It starts a new
optimizer and schedule; this is not an optimizer resume of the old dataset run.

Checkpoint bindings include the new export receipt, warm-start weights/config,
processor configs/statistics, adapter/trainer hashes, seed and sampling plan.
The existing finite checks, frozen-backbone checks, BF16, gradient clipping,
atomic status/checkpoint completion, optional retention and exact state/RNG
resume mechanics are retained. Recovery-specific fresh-process exact resume
still needs independent qualification before an unattended long run.

`scripts/run_recovery_pilot.py` supervises a small test with an independent
180--900 second hard deadline and owned-descendant cleanup. Defaults: 16 updates,
batch four, learning rate 1e-5, eight timeline probes per split, original best
checkpoint `step_0009426_best`, unchanged images/state/action dimensions. These
defaults test mechanics; they are not a tuned recovery training recipe.

```powershell
.venv-smolvla\Scripts\python.exe scripts\verify_recovery_export.py `
  --view data/recovery_pipeline_20261019/qualification_view `
  --export data/recovery_pipeline_20261019/qualification_export `
  --report outputs/recovery_pipeline_20261019/export_verification.json

.venv-smolvla\Scripts\python.exe scripts\run_recovery_pilot.py `
  --export data/recovery_pipeline_20261019/qualification_export `
  --checkpoint outputs/v2c30_sft_overnight_v1/checkpoints/step_0009426_best/pretrained_model `
  --backbone data/models/smolvlm_backbone `
  --output outputs/recovery_pipeline_20261019/pilot_v1
```

## Gate before a meaningful long run

Finish downloading/validating the selected release and all six held-out parents;
obtain the exact new collector/environment source and qualify expert replay.
Retain clean-task supervision or a deliberately reviewed mixture so recovery
training does not erase nominal behavior. Compare recovery-aware loss and
gripper/joint errors separately, then evaluate identical resets with unchanged
guards. Actual plug contact, grasp, lift and insertion determine progress.

Do not proceed to RL merely because a smoke run completes. State-noise
augmentation remains a separate controlled option; this first integration uses
real corrective demonstrations without synthetic observation changes.

## Completed local qualification

`outputs/recovery_pipeline_20261019/qualification_summary.json` records the
combined result. The real LeRobot export passed numerical/camera readback and
fault-boundary checks. The supervised pilot completed all 16 optimizer updates
normally in 88.14 seconds, including evaluation and checkpoint writes. All logged
losses and gradient norms were finite; sampled anchors were 60 expert and four
recovery actions, with zero faults. Peak PyTorch allocation was 2.52 GB.

Eight fixed validation probes on the **one downloaded held-out parent** changed
from mean flow loss 0.02785570 to 0.02470555. This is a small engineering test,
not full validation, a tuned model, statistical significance or evidence that
the gripper-feedback failure has been solved.

Fresh-process checkpoint reload passed: the matched flow-loss probe was exactly
equal (difference zero), all inherited normalization tensors were unchanged,
and decoded output was finite with shape `(1, 50, 8)`. The baseline model checksum
is unchanged. Final candidate:
`outputs/recovery_pipeline_20261019/pilot_v1/training/checkpoints/step_0000016_final/pretrained_model`.

Detailed artifacts: `export_verification.json`, `checkpoint_verification.json`,
`pilot_v1/train.log`, `pilot_v1/training/events.jsonl`, and the saved training and
pilot reports. No training process remains running after qualification.
Selected regression suites passed 37 tests across data boundaries, recovery
segmentation, archive inspection, export helpers, budgets and Windows I/O.
At qualification time, no autonomous simulation or hardware test of this
candidate had been performed. The subsequent old-scene diagnostic is linked above;
evaluation in the new recovery environment still requires its matching source.
The remaining 26 recovery parents still require download/validation for a
complete-release run. The source-revision/replay and exact optimizer-resume gates
above remain unresolved.
