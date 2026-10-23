# Native one-episode pickup qualification

Completed 2026-10-22: `outputs/native_single_episode_20261022_v2`.
**Live pickup passed on one training reset / one inference seed**, after 1,386
additional native updates. Grasp at command 335, pickup gate at command 415;
68.794 mm maximum lift, one continuous second carry, no drops or safety abort.
All saved model tensors reloaded exactly; normalization remained unchanged.
Final model SHA-256: `6873ab243639b6b8df563dcb92fc8e29071d4d4abe4050627bb51341c9ef927f`.
See [current status](TRAINING_STATUS.md) and [lab handoff](LAB_HANDOFF.md).
This is a bounded memorization diagnostic, not generalization or insertion.

## Fixed experiment

- Existing training parent `train_20002`, all 1,673 command rows retained,
  selected with native `dataset.episodes`; no data copy/re-export/phase quota.
- Warm-start from native step 12,243, SHA-256
  `0cefdc6ffc0c634f17fe35a94d3940b7f15de2833c8e3142837b5a1a5e7df1de`.
  New native optimizer/scheduler, not an optimizer-state resume. Fresh native
  preflight and main run both independently start from this same initializer.
- Native LeRobot 0.4.4 training loop, uniform shuffle, model/loss, AdamW and
  original warmup/cosine preset. Batch 8, existing FP32 trainable storage/
  Adam moments and BF16 Accelerator. No new loss, augmentation or controller
  heuristic. Full-training-split normalization remains unchanged.
- 2,000-update ceiling or 30 minutes of training, whichever comes first.
  45-minute hard campaign cap includes fresh expert replay, eight-update
  preflight, checkpoint save/reload and final live rollout. No auto-retry.
- One retained final native checkpoint. No preflight weight copy, early step
  1,000 checkpoint, checkpoint sweep or deletion. Disk reserve remains 25 GiB;
  RAM >=1 GiB, GPU free >=512 MiB, temperature <85 C. Sources and immutable
  initializer/data hashes are checked; owned process trees stop at deadlines.

## Qualification and evidence

The real native episode-filter view must contain exactly this parent and its
full length. Nine raw/export anchors check states, 50-command applied-action
targets and padding. The initial native model must reload all saved tensors
exactly. At update 0/1/every 250/final, probe eight fixed recorded observations:
first frame, six phase medians and just before closure. Record first-five
arm/finger RMSE and native flow loss at times 0.1/0.5/0.9. These are training
probes, not validation. Probe RNG and numerical flags are restored afterward;
probes never update weights or alter native sampling/optimization.

Saved pre/postprocessor statistic tensors must equal the initializer's.
The final checkpoint is freshly reloaded and all tensors checked before live
testing. No success claim from generated-command error or loss alone.

Final evaluation uses fresh simulator physics and live proprioception/two RGBs,
same matched scene, prefix five/horizon 50, seed 1000 with step-keyed noise and
controlled BF16 numerics. Repeat the first prediction once with identical
inputs/noise before commands. Compare against the fresh exact expert control.
Require actual two-finger grasp, >=20 mm plug lift and >=1 continuous second
carry, without a prior drop or safety abort. Maximum 1,200 commands and 360
seconds per trial. No recorded-state/image assistance or physical overrides.

Campaign `health.json`/logs show live status. Terminal `report.json` or
`failure.json` preserves outcomes. A bounded independent watcher writes
`results/native_single_episode_20261022_v2`: audited summary, update/probe CSVs,
and PNG/PDF curves. Actual video lives in the policy trial's `rollout.mp4`.
Video timing is simulated time with inference pauses omitted, not real-time
hardware readiness. Unfinished trials are not counted as policy failures.

The original campaign's fresh expert control passed. Its first preflight
stopped before any optimizer update because the new audit assumed state shape
16 rather than native history shape 1 x 16. V2 accepts only those two documented
audit shapes without changing actual dataset/model inputs. The original failed
log/report is preserved; V2 charges 275.596 seconds of preceding execution and
repair rather than resetting the 45-minute clock. The unchanged expert control
was independently re-audited and hash-checked when copied into V2; it was not
silently rescored or counted twice. Five selection/history unit tests pass.
The repaired real CLI preflight completed all eight finite updates without
writing weights; the main run independently reloaded the same initializer.
The selected regression suite passed 62 tests. The native scheduler preserves
its public automatic scaling of warmup/decay to the 2,000-update ceiling;
the preset's nominal 1,000 warmup steps are not used literally in this short run.
At the first scheduled main-run probe (update 250), closure-command RMSE on
the fixed recorded observation was 5.41 mm versus 4.88 mm initially; this early
window had not improved yet. The completed final live result is reported above;
this paragraph preserves the interim observation rather than treating it as final.

## Decision

If one fresh live pickup succeeds, the native recipe has a memorization
positive control; repeatability/generalization are still unproven. Expand
gradually only after that. If it fails, keep the case small and use changes
in approach/closure accuracy and live divergence to choose the next diagnosis.
Do not launch another full-dataset run or claim a fix automatically.
