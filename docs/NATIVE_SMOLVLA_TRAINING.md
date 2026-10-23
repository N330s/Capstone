# Native LeRobot SmolVLA fine-tuning

Completed 2026-10-22: **12,243 updates**, batch 8, in **3h 57m 22s** total
including qualification/repair. All logged metrics finite; stopped at the
authorized time budget. Fixed-window validation: **0.466116 -> 0.004507**
over 160 windows spanning all 20 validation episodes (not rollout success).
Audited curves/CSV and retained-checkpoint selection:
`results/native_smolvla_20261021_v2`. Selected checkpoint: step 12,243.
Online evaluation completed: expert controls **5/5**, VLA training-reset
pickups **0/5** (four command-limit failures, one collision safety stop).
Conditional validation was not run. See the evidence/next gate in
[NATIVE_SMOLVLA_EVALUATION.md](NATIVE_SMOLVLA_EVALUATION.md).

2026-10-21 training launch record. Clean run:
`outputs/native_smolvla_20261021_v2/training`.
Supervision/receipts: its parent directory. User-authorized maximum is **four
hours total**, including qualification, batch checks, failed preflight and
intervening repair time. No timer reset, checkpoint deletion, Hub upload or
model/API token use. This PC is an RTX 4060 Ti 8 GiB, not the lab RTX 4090.

## What changed and what did not

The training loop is `lerobot.scripts.lerobot_train.train` from pinned LeRobot
0.4.4. Dataset, shuffle sampling, processors, policy, flow objective, optimizer,
scheduler and checkpoint serialization are public LeRobot implementations.
`scripts/train_native_smolvla.py` installs small process-local adapters; no
site-packages edits or dependency upgrades:

- Explicitly derive input features from dataset metadata. An empty CLI feature
  dictionary is recursively merged with the pretrained config and does **not**
  clear its old six-value state/third camera. Clear it programmatically before
  the native factory, then assert state 16, action 8, two RGB cameras, horizon 50.
- Alias dataset `action_is_pad` to this release's `actions_id_pad` policy key;
  require a boolean B x H mask and reject conflicting aliases.
- Restore checkpoint trainable precision and use FP32 trainable weights/Adam
  moments with an explicit native BF16 Accelerator. Check precision each update.
- Use staged local tokenizer files. Preserve native pretrained processor settings
  and explicitly overridden **training-only** statistics.
- Resource guards, local metric writes, offline fixed-window validation and
  graceful budget-stop checkpoint callbacks. Windows uses an explicit JSON last
  checkpoint pointer instead of requiring symlink privileges.

No custom loss, phase quotas, gripper sensor perturbations, image augmentation,
recovery mixture, actuator heuristic, scene repair or simulator state forcing.
The immutable initializer is original `data/models/smolvla_base`, SHA-256
`7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`, not an older
local fine-tuned actor. Both preflight and long run start independently from it.

## Completed qualification

- `outputs/native_qualification_20261021_v2/report.json`: two real-model updates
  match a controlled native-forward/reference calculation **exactly** in loss,
  clipped gradient hashes, LR and resulting trainable-weight hashes. Frozen
  weights remain unchanged. All 99,880,992 trainable parameters are FP32.
- Nine raw/export anchors (start/middle/terminal for three training episodes)
  match state, 50-command targets, instruction and terminal masks. Terminal
  anchors have 49 padded targets. Decoded RGB compression MAE is reported
  separately, about 1.59-4.85 uint8 levels in these samples. This is not an
  all-frame image/timing audit or a certification of dataset task quality.
- Compute checks at batches 4 and 8 passed. Batch 8: about 17.36 samples/s of
  fixed-batch native compute, 4,016 MiB peak reserved PyTorch memory, about
  2.5 GiB conservative GPU headroom. Selected batch 8; sizes above 8 are untested.
  End-to-end dataloader performance is slower; do not present compute-only time
  as total training throughput.
- Full native CLI preflight: 12 finite random-data updates, native full
  model/optimizer checkpoint saved. Fresh-process reload checks all **500 saved
  tensors** bit-exactly and saved processors against training statistics.
- 43 selected unit/regression tests pass, including six native-adapter tests.

The first CLI preflight failed **before training** because the empty-dict
override retained SO100 features. Its failed log and completed compute checks
remain in `outputs/native_smolvla_20261021_v1`. The repaired adapter was
requalified against the reference before continuing in V2. Its receipt charges
419 seconds at continuation start, counting the failed attempt and intervening
repair/requalification wall time. Reused compute benchmarks already used the
correct programmatically derived feature contract. This new-CLI bug is not
evidence that earlier programmatically configured training used six-value state.

## Active recipe and evidence

- Existing export: `data/drive_v2c30_20261012/export_all_v1` (81 train / 20
  validation episodes). Native uniform frame sampling and physical-unit
  absolute right-arm joint/finger actions, bimanual proprioception, two RGB
  cameras and instruction. Torque/depth remain unused. Full-task episodes are
  retained; this is not a silently trimmed pickup-only dataset.
- 20,000-update ceiling; actual run stops sooner if the total four-hour wall
  budget expires. Batch 8 is **not** equivalent to the tutorial's batch 64;
  no unqualified accumulation flag or effective-batch claim.
- Original pretrained optimizer/scheduler presets (peak LR 1e-4, native
  warmup/cosine schedule). Native Windows dataloader uses zero workers for
  this qualified baseline; throughput tuning is separate from model validity.
- Train loss, pre-clipping gradient norm, LR and update time: every update in
  `training/native_events.jsonl`. Effective feature shapes and precision are
  logged before the first update.
- Validation: 160 fixed flow-loss windows spanning **all 20 validation
  episodes**, at step 1/every 1,000/final budget stop, using training processors,
  fixed per-window noise and flow time 0.5. Plan in
  `training/native_validation_plan.json`. This is **not every validation frame**,
  stochastic training loss, or rollout success. RNG/model training mode are
  restored afterward. First measurement after one tiny update: 0.466116.
- Full checkpoints at step 1,000, then native 5,000-step intervals/final, plus
  a graceful time-budget checkpoint. Complete receipts verify FP32 storage and
  bind model hashes. Every existing checkpoint remains intact.
- Guards retain 25 GiB disk reserve, 1 GiB available RAM, 512 MiB GPU free and
  temperature below 85 C. Source edits abort the frozen campaign. Parent
  deadlines kill the complete owned process tree, including Windows launchers.
  Checkpoint writes reserve their estimated size above the disk floor.

The entrypoint stops with time reserved for validation/checkpoint serialization;
the supervisor enforces the original hard deadline. A resource failure stops
without automatically retrying or deleting work. Checkpoints use native
RNG/optimizer/scheduler state, but exact dataloader-cursor replay is not claimed.

## Reporting and online evaluation

`scripts/report_native_smolvla.py` completed locally (no model/optimizer calls).
It verified saved artifacts, wrote CSVs and PNG/PDF training
and fixed-window validation curves to `results/native_smolvla_20261021_v2`,
and selects the lowest fixed-window validation loss **among retained
checkpoints**, with a latest-checkpoint fallback. Watch `report_status.json`.
Missing/failed training produces an explicit partial report, not task-success
evidence. Full-validation evaluation is a separate measured job, not silently
claimed from these 160 windows.

Now evaluate a freshly reloaded selected checkpoint in the unchanged matched
simulator: expert controls, original five training resets, physical two-finger
pickup/lift/carry/no-drop gate, then the fixed held-out bank. Check new raw-input
fingerprints and declare numerical/execution settings. Preserve prefix five as
a candidate from the earlier diagnostic, not a proved generalization fix.
Insertion and RLT wait for reliable pickup.

The earlier frozen-actor numerical-mode/call-order discrepancy remains open;
do not claim it was solved by the native training migration. Fixed-input
same-profile repeatability and native offline update/data parity are qualified
independently, sufficient to start this clean SFT baseline without making a
closed-loop reliability or bug-free claim.

References:
[official SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla),
[pinned public trainer](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/scripts/lerobot_train.py).
