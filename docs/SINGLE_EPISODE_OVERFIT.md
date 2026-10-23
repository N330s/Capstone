# One-demonstration memorization diagnostic

## Subsequent repair and approved artifact cleanup (2026-10-19)

This document records the historical, failed mixed-storage experiment. Its
trainer has since been repaired for FP32 trainable weights and optimizer moments;
see [TRAINING_PRECISION_FIX.md](TRAINING_PRECISION_FIX.md). References below to
an unchanged trainer and no active job describe the earlier experiment, not the
current workspace. Exact historical trainer source is hash-archived.

With explicit user approval, only
`outputs/single_episode_overfit_20261019/pretrained_model` was permanently removed
to recover 1,197,810,221 bytes (about 1.12 GiB). Its saved model SHA256 was
`c26af72641fb4c99c92746c9a7a779bd8d859d1a17af60cc1e6324efe9dd1504`.
The logs, saved verification results, trajectories, videos, plots and original
best model remain. The deleted candidate cannot now be reloaded without
recreating it; historical reports are retained unchanged as records of prior
verification, not a claim that the model bundle is still present.

Question: can the existing SmolVLA setup reproduce one known successful training
demonstration when it must use its own live camera images and measured joints?
This separates a basic memorization/control problem from a broad data-coverage
problem. Passing would not establish generalization; failing would not by itself
prove a particular bug or that SmolVLA cannot learn the task.

## Fixed protocol

- Original scene and source-matched collector checkout; `train_20002`, 1,673
  commands, original training split. Validation parents remain untouched.
- Require fresh expert-command replay to complete grasp and insertion with
  qpos/qvel maximum error at most 1e-8 before any fine-tuning.
- Run the original best checkpoint once on the same initial reset.
- Warm-start that checkpoint; train only on this episode's command anchors,
  sampled uniformly with replacement. Preserve 50-command chunks, terminal
  padding masks, 16 measured positions, two RGB views and eight absolute targets.
- Batch eight, BF16, peak learning rate 1e-4, maximum 2,000 updates / 30 minutes.
  Vision/language remains frozen under the inherited upstream configuration.
  Existing normalization is inherited, never fitted to validation.
- Fixed-noise probes at baseline/every 250 updates/final: 32 timeline anchors
  from the memorized episode and eight monitor-only original validation anchors.
  Report flow loss and physical first-five-command gripper/joint error.
- Save one final inference-only model. No validation-based selection, no optimizer
  checkpoint or exact-resume claim; no existing checkpoints are removed.
- Reload in a fresh process: require matching saved flow loss and exactly
  unchanged normalization tensors. Then run live closed-loop simulation with
  execution prefix five and noise seed 1000 from the same initial reset.
- Keep all robot/table/contact guards. Never substitute recorded actor inputs or
  force robot/object state after initialization. Total supervisor limit one hour.
- Require 26.5 GiB free initially and retain at least 25 GiB for this bounded
  weights-only diagnostic. This differs from the larger recovery trainer's
  29 GiB guard, which reserves optimizer and multiple checkpoint copies.

The first useful gate is approach and grasp without collision; insertion is
reported separately. Do not choose more training, more data, different targets or
RL until the actual paired result is reviewed. A single failure cannot rule out
other learning rates, budgets or model architectures.

## Implementation and artifacts

`scripts/overfit_smolvla_episode.py` owns source/split checks, hard per-stage and
total deadlines, owned descendant cleanup and three physical rollouts. It leaves
`scripts/finetune_smolvla.py`, original data and original weights unchanged.

`scripts/report_single_episode_overfit.py` independently audits every update's
sampled indices and finite loss/gradients, fully decodes videos, and produces
JSON/CSV and plots. The training environment has PyAV; system Python has
matplotlib, so verification and plotting are separate commands.

```powershell
.venv-smolvla\Scripts\python.exe -u scripts/overfit_smolvla_episode.py --output outputs/single_episode_overfit_20261019
.venv-smolvla\Scripts\python.exe scripts/report_single_episode_overfit.py --evaluation outputs/single_episode_overfit_20261019 --output results/single_episode_overfit_20261019 --verify-only
python scripts/report_single_episode_overfit.py --evaluation outputs/single_episode_overfit_20261019 --output results/single_episode_overfit_20261019 --plot-only
```

## Completed result (2026-10-19)

The protocol completed in 37.4 minutes. Training reached **1,327 updates** before
its preset wall-time cutoff (28.3 minutes including final probes/save). All
10,616 sampled anchors belonged to this one training parent; 1,672 of its 1,673
possible anchors were visited, averaging 6.35 visits per anchor. All update losses
and gradient norms were finite. The checkpoint reloaded exactly; normalization
tensors and the original baseline weights/trainer remained unchanged.

| Physical condition | Commands | Grasp | Insert | Stop |
| --- | ---: | --- | --- | --- |
| Fresh expert-command replay | 1,673 | Yes | Yes | Success; qpos/qvel error exactly zero |
| Original best VLA, live inputs | 322 | No | No | Finger self-contact |
| Single-episode fine-tune, live inputs | 292 | No | No | Finger self-contact |

The last model closed the measured gripper from 43.99 mm to 1.77 mm during
`transit_pick`. At command 292, the reference is still approaching; its applied
target first falls below 30 mm at command 522. This is premature closure, not a
failed insertion or a recovery-scene mismatch. No state/image assistance or
simulator-state forcing was used. Guards were not relaxed.

On 32 fixed training-episode probes, flow loss improved **0.002001 -> 0.001181**
and first-five-command gripper MAE improved **0.349 -> 0.269 mm**, but joint MAE
did not improve (**0.006217 -> 0.006388 rad**). Eight monitor-only held-out probes
deteriorated markedly: flow loss **0.003633 -> 0.057077**. These are fixed probe
subsets, not full-dataset validation. The candidate is not promoted.

This attempt **did not establish memorization**. It does not prove that the
architecture cannot learn, that the task needs more data, or that all training
code is bug-free. It verifies source replay, sampled-parent isolation, selected
raw/export state/action/terminal-mask equality, finite optimization, saved-model
reload and unchanged normalization. It does not identify the sole cause of the
control failure.

## Next controlled move

First repeat the one-parent fitting test with a **lower warm-start peak learning
rate (1e-5)** and a schedule that finishes inside the measured runtime budget
(for example 1,000 updates). This is a proposed experiment, not an established
fix. The current 1e-4 restart worsened the early probes, and its 2,000-update
schedule was cut off at 1,327 updates, with LR still 2.73e-5. Require meaningful
improvement in physical joint/gripper errors, especially approach/open-hold
anchors, rather than accepting aggregate flow-loss improvement as memorization.
Then repeat the identical live rollout. Do not expand collection or begin RL
based on this failed gate alone.

If good recorded-input fitting still fails in live control, the next isolated
ablation is requested intent targets versus the current rate-limited applied
targets, using the existing raw arrays and exact expert replay. That is a
hypothesis about state-copying/closed-loop sensitivity, not a proven label bug.
Policy-induced early-transit correction data is a later option, not a conclusion
that another large collection is necessary now.

No further run was launched. Approximately 25.9 GiB remains free; free about
1 GiB more before another saved-model run, or explicitly approve removal of
this failed diagnostic's model bundle while preserving its logs/evidence. No
archives or checkpoints were deleted.

Reviewed artifacts:

- [Diagnostic plot](../results/single_episode_overfit_20261019/overfit_diagnostic.png),
  [PDF](../results/single_episode_overfit_20261019/overfit_diagnostic.pdf),
  [audited summary](../results/single_episode_overfit_20261019/summary.json),
  [conditions CSV](../results/single_episode_overfit_20261019/conditions.csv).
- VLA video: `outputs/single_episode_overfit_20261019/overfit/rollout.mp4`.
- Expert control: `outputs/single_episode_overfit_20261019/expert_replay/rollout.mp4`.
- Original VLA: `outputs/single_episode_overfit_20261019/baseline/rollout.mp4`.

All videos were fully decoded and have exactly commands + 1 frames. The report
independently read back six raw/export anchors including the final padded chunk.
Raw logs, source hashes, model and physical trajectories are retained under
`outputs/single_episode_overfit_20261019`. No training or keep-awake job remains.
