# Post-training SmolVLA evaluation

Prepared for `v2c30_sft_overnight_v1`. The overnight controller already attempts
one 2,000-command simulation rollout. The workflow below adds full validation and
paired simulation resets. It is prepared for manual execution, not scheduled.

## Before running

Use `.venv-smolvla` and retain the exported dataset, merged raw collection, local
backbone and matching `third_party/capstone_automateData` checkout. The evaluator
requires a completed training report and a campaign completion file (a failed
campaign is acceptable if training completed but its smoke rollout failed).
Incomplete training is refused. Checkpoint completion and dataset bindings are checked.

CPU-only inventory is safe while training runs:

```powershell
.venv-smolvla/Scripts/python.exe scripts/evaluate_smolvla_run.py --output outputs/v2c30_eval_inventory --prepare-only
```

## Quick simulation first (current preferred workflow)

The user subsequently extended training to 08:00 on October 13. For that
continuation use the following after its `completion.json` reports completion:

```powershell
.venv-smolvla/Scripts/python.exe -u scripts/evaluate_smolvla_run.py --run outputs/v2c30_sft_until_0800_v1 --campaign outputs/v2c30_sft_until_0800_v1 --output outputs/v2c30_extended_best_sim_v1 --simulation-only --checkpoint-choice best --rollout-episodes 1 --stage-timeout-seconds 900
```

The continuation's best includes its initial inherited checkpoint as a fallback.
If continuation failed, inspect its logs and use the completed original run below.

After training and its automatic smoke rollout finish, test only the saved best
checkpoint on one held-out reset, skipping all offline validation:

```powershell
.venv-smolvla/Scripts/python.exe -u scripts/evaluate_smolvla_run.py --output outputs/v2c30_best_sim_v1 --simulation-only --checkpoint-choice best --rollout-episodes 1 --stage-timeout-seconds 900
```

This has a 15-minute subprocess ceiling, not a promised runtime. It saves an
animation and task/grasp/collision metrics. Use `--checkpoint-choice latest` to
test the latest complete checkpoint instead, with a new output directory.
Best means lowest saved probe loss; it may not yield the best task success.
For a less anecdotal check, repeat with `--rollout-episodes 3`. Full validation
below remains optional. Simulation-only reports explicitly record that validation
was skipped. No new evaluation is automatically launched tonight.

## Optional full validation workflow

First qualify the evaluator on 16 windows of each retained best/final model:

```powershell
.venv-smolvla/Scripts/python.exe -u scripts/evaluate_smolvla_run.py --output outputs/v2c30_eval_smoke_v1 --max-windows 16 --rollout-episodes 0
```

Verify `report.json` says `completed: true`. This check explicitly records
`full_validation: false`. Then evaluate all validation windows and three identical
held-out simulation resets per model:

```powershell
.venv-smolvla/Scripts/python.exe -u scripts/evaluate_smolvla_run.py --output outputs/v2c30_eval_full_v1
```

Use new output directories. Batch size defaults to 8; `--batch-size 4` reduces
memory demand. GPU execution is not yet qualified by tonight's CPU checks.
Each subprocess has a four-hour timeout, separate from the training allowance.
No optimizer updates, downloads, uploads or hardware commands occur.
Runtime is not benchmarked; each model's `status.json` estimates remaining
validation time from completed batches. Keep the computer awake during evaluation.

Use `--rollout-episodes 20` to cover all held-out resets. Trials use 2,000-command
horizons, prefix 5 and seed 1000, with the matching scene and existing guards.
Models run sequentially in separate processes to release GPU memory.
Identical checkpoint paths are evaluated once with both labels.

## Outputs and interpretation

- Top-level `report.json`: completion, validation comparisons and rollout reports.
- `best/report.json` and `final/report.json` (or `best_final`): all 37,104 validation
  windows across 20 episodes, command-weighted and episode-balanced loss,
  per-episode means, hashes and timing. `windows.csv` supports later plots.
- Joint MAE (radians) and gripper MAE (mm) are sampled at eight timeline locations
  per episode; these decoded-action errors do not cover every window.
- Rollout folders: animations, trajectories, task success, grasp establishment,
  safety aborts, failure phase, requested/applied commands and inference latency.
- `failure.json` and stage logs: failures; partial results are not marked complete.

Loss uses fixed flow time 0.5 and one seeded noise draw per window, identical for
both models and independent of batch grouping. It preserves upstream reduction:
terminal padding contributes zeros to a full-chunk average. Different sample/noise
selection means values need not equal the 64-probe curve. This covers every window,
not every possible flow time/noise draw.

The validation set participated in checkpoint selection; it is not an untouched
test set. Simulation uses paused-time inference and finite horizons. Low loss
does not establish grasp-and-insert success or hardware readiness. Phase-specific
offline aggregation, unseen test resets and scene contact calibration remain pending.
