# Correction-aware recovery SFT

2026-10-19. This experiment tests training exposure before requesting more data.
The original 16-update qualification contained no active reopening target.

## Protocol

- Preserve the immutable 24-train / 6-validation parent split plan.
- Local disk is limited (~37 GiB free at start). Selectively retrieve all six
  planned validation parents and six training parents, two per recovery type,
  rather than duplicate 26 GiB of archives or exhaust checkpoint space. This is
  explicitly a partial training release; eighteen training parents remain absent.
- Validate checksums, timing, action/state ordering, labels and auxiliary arrays;
  export contiguous non-fault segments to a new view/export. Keep all old files.
- Build a checksum-bound sampling index from physical source state/actions.
  Active reopening means a recovery-labeled applied target exceeds measured
  gripper travel at that command by >0.2 mm. This is a sample category, not a
  command clamp or a physical safety limit.
- Correction batches: at least 25% active reopening anchors, approximately 25%
  other recovery anchors, and the remainder nominal expert anchors. Sampling
  is with replacement; quotas are enforced on every batch and logged.
- Equal-budget uniform control: same release, warm-start weights, normalization,
  learning rate, batch size, seed, update budget and evaluation probes. Only the
  selection rule changes. A single-seed comparison is diagnostic, not a general
  statistical conclusion about an optimal sampling mixture.
- Planned budget: 400 updates per condition, batch 8, BF16, LR 1e-5, fixed
  evaluation every 100 updates. Each process has a 30-minute independent hard
  ceiling including imports, evaluation and checkpoint writes. No overnight run.
- Evaluate held-out active reopening separately from aggregate timeline loss;
  retain first-target, five-command and whole-chunk gripper MAE and joint MAE.
  Probe all recorded recovery commands of all six held-out parents after training
  with three fixed noise seeds. Also check the original three clean recordings
  and their counterfactual gripper-state perturbations for retention.
- Audit actual sampled chunks against raw labels, including every future target;
  no fault anchors, fault targets, cross-fault stitching or validation samples.
- Reload each saved checkpoint in a fresh process and verify fixed-probe loss,
  inherited normalization tensors and finite physical-unit output.

Input contract stays two RGB cameras plus 16 positions, output 50x8 absolute
targets. Depth, chest RGB, torque, phases and fault labels are not actor inputs.
The original training code and baseline checkpoint are preserved. These new
runs start new optimizers; they do not resume either the original dataset run
or the 16-update qualification. Changed trainer bindings make that distinction
explicit. Exact optimizer resume of this new recovery version remains unqualified.

## Files

`learning/vla/recovery_sampling.py` constructs/verifies the offline index and
implements deterministic quota sampling with the existing saved sampler RNG.
`scripts/prepare_recovery_sampling.py` writes a new index.
`scripts/finetune_recovery_smolvla.py` supports both selection modes and separate
held-out correction probes. `scripts/run_recovery_pilot.py --experiment` allows
up to 600 updates / 30 minutes with owned-process cleanup; qualification defaults
remain bounded to 64 updates / 15 minutes. At least 29 GiB free is required
before launching a checkpointed experiment, preserving the intake reserve.
`scripts/report_recovery_comparison.py` verifies equal experimental bindings
and produces a JSON summary, CSV and comparison plot.

## Interpretation and remaining gates

Offline correction improvement does not establish recovery, grasp or insertion
success. The matching new simulator source is still missing. Old-environment
rollouts can only measure transfer/regression, not exact new-scene performance.
If the early transit collapse persists despite learning later reopening, target
early transit corrective coverage or a separately controlled state-noise study.
Do not introduce torque/depth, change action chunking, weaken collision guards,
collect large quantities of nominal data, or start RL based on mean loss alone.

## Completed offline results

The twelve downloaded parents contain 28,549 raw commands: 24,532 eligible
commands and 4,017 excluded faults. Export/readback passed all 24 segments.
Training: 12,073 commands in six parents; validation: 12,459 in all six planned
parents. The index has 236 active opening commands in each split. The training
set has 892 total recovery commands; active opening is only 1.95% of its eligible
anchors. Adjacent 50 Hz frames are correlated, not independent demonstrations.

Both runs completed 400 updates normally, batch eight, ~5.6 minutes each, with
3.32 GB peak PyTorch allocation. All 800 logged update losses/gradient norms
were finite. Fresh-process reload matched the final saved probe exactly for
both models, preserved both normalization tensors, and decoded finite 50x8
physical-unit actions. Audits found zero fault targets in either run. Original
baseline weights and `scripts/finetune_smolvla.py` are unchanged.

Actual reopening anchors: uniform 62/3,200 (1.94%); correction 800/3,200 (25%).
The two runs have identical initial validation measurements and checked matching
data, checkpoint, schedule, batch, seed, trainer and probe bindings.

Each model was probed on all 975 recovery commands of all six held-out parents,
plus selected old/new clean and old counterfactual-state observations. There
were 1,037 observation cases and three fixed noise seeds: 3,111 predictions per
model. Active-opening comparison uses all 236 eligible held-out opening frames,
or 708 predictions. This is complete **recovery-section** coverage, not every
one of the 12,459 eligible validation commands or an autonomous evaluation.

| Recorded-observation metric | Original baseline | Uniform 400 | Correction 400 |
| --- | ---: | ---: | ---: |
| Active opening: five-command gripper MAE | 5.621 mm | 4.552 mm | 2.664 mm |
| Active opening: first prediction commands opening | 2.54% | 12.99% | 57.34% |
| Active opening: mean first target delta from measured travel | -4.516 mm | -3.510 mm | +0.286 mm |
| Expert opening delta on those same observations | +0.291 mm | +0.291 mm | +0.291 mm |
| Original clean probes: five-command gripper MAE | 0.983 mm | 1.673 mm | 4.271 mm |
| Original +/-3 mm state probes: local target/state sensitivity gain | 1.022 | 1.036 | 1.016 |

The correction model learns opening on existing data, but not reliably, and
regresses substantially on nominal behavior. On original recorded `close`
probes its gripper MAE rises 0.880 -> 4.867 mm; on `lift`, 2.641 -> 9.799 mm.
At lift observations, mean expert target delta is -7.999 mm (maintain closure
against an object), whereas the correction model predicts +1.531 mm (opening).
That is not an acceptable task policy. The near-one conditioning sensitivity
also persists; a positive opening bias is not the same as learning a restoring
feedback response. These are local offline diagnostics, not stability proofs.

Sampled timeline flow loss improves 0.042670 -> 0.035108 uniform and -> 0.024899
correction. Lower aggregate flow loss does not negate the clean-behavior
regression. The 25% reopening / 25% other-recovery mixture is an experimental
contrast, **not a recommended production recipe**. Neither checkpoint is promoted.

## Guarded closed-loop transfer check

Fresh paired rollouts used original held-out reset `train_20004`, seed 1000,
chunk 50, execution prefix five, unchanged source/robot/workspace hashes and
collision guards. The comparison finished in 145.52 seconds:

| Policy | Commands before abort | Outcome | Grasp / insertion |
| --- | ---: | --- | --- |
| Original best | 322 | Finger self-contact, `robot_collision` | No / no |
| Correction 400 final | 283 | Finger self-contact, `robot_collision` | No / no |

Both stop while the recorded expert is still in `transit_pick`. Measured gripper
travel reaches about 1.78 mm in both runs. The correction model's learned
near-object reopening has **not** fixed early empty-hand closure. This is one
old-scene reset and one policy seed, not evidence of new-scene performance or
proof that every additional recovery-training recipe would fail. The uniform
candidate was compared offline; it was not given a new closed-loop rollout here.

Videos decoded fully with 323 baseline / 284 correction frames, matching the
command count plus initial observation. Raw trajectories, diagnostics, source
bindings, logs and MP4s are in
`outputs/recovery_sampling_20261019/old_scene_transfer/`.

Final figures/PDFs, validation-metric CSV, behavior-metric CSV, checkpoint
checksums, equal-budget assertions and transfer summary are in
`results/recovery_sampling_20261019/`. The earlier `report_preview` under
`outputs` is a draft; the `results` version has the reviewed layout and transfer
result. Original baseline/trainer SHA256 values match their pre-experiment
values. Fifty-six selected recovery, legacy-data, archive, sampler, Windows I/O,
deadline and rollout-contract tests passed; modified scripts compile and
`git diff --check` passes. No training/evaluation process remains active.

## Evidence-based next experiment

1. Preserve nominal closing/holding/lift supervision explicitly. Review a
   multi-export replay mixture of the original **training** parents and new
   recovery training parents; keep both validation sets separate and retain
   source-version receipts rather than bypass the mixed-scene rejection.
2. Test a lower corrective exposure (for example 5--10% active opening) against
   the uniform control, with a logged batch-cycle quota, matched update budgets,
   and predeclared clean-retention/recovery criteria. Do not tune only mean loss.
   Change one factor at a time when measuring the effect of the quota.
3. Use the existing eighteen undownloaded training parents to increase independent
   trajectory diversity before requesting a large new nominal collection.
   About 27 GiB remains free after both runs; allow roughly another 10 GiB of
   storage headroom or choose another data drive before full intake. Do not
   delete original archives/checkpoints automatically. Even another bounded
   checkpointed experiment requires restoring the launch guard's 29 GiB minimum.
4. If guarded early-transit failures persist, request targeted early-transit
   open-hold corrections and matched nominal close/hold examples. Later
   near-object reopening does not establish coverage of early empty-hand drift.
5. Obtain the exact new environment source and qualify expert replay before
   claiming same-scene new-data policy evaluation, unattended long training,
   RL readiness or sim-to-real readiness. Recovery exact optimizer resume still
   needs its own GPU qualification. No further training is automatically launched.

Reproduce the reviewed report (choose a new output directory):

```powershell
python scripts/report_recovery_comparison.py `
  --uniform outputs/recovery_sampling_20261019/uniform_400 `
  --correction outputs/recovery_sampling_20261019/correction_400 `
  --baseline-behavior outputs/recovery_sampling_20261019/baseline_behavior/report.json `
  --transfer outputs/recovery_sampling_20261019/old_scene_transfer/report.json `
  --output results/recovery_sampling_reproduced
```

Use the system Python with matplotlib for report rendering; the pinned training
environment intentionally does not require matplotlib. Exact training arguments
are saved in each supervised run's `command.json`. Selective download provenance
is in each expansion's `archive_inventory.json`.
