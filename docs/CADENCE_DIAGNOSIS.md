# Cadence audit and public SmolVLA path

2026-10-21. The objective is reliable SmolVLA fine-tuning for OpenArm, followed
by held-out pickup, insertion and eventually RL/sim-to-real. Custom training
code is not an objective: prefer the maintained public implementation once
its data, precision and checkpoint interfaces are qualified for this robot.

## Completed campaign, but not an isolated cadence result

`outputs/inference_cadence_20261020_resume_v1/completion.json` records all 12
conditions complete in 2,364.015 cumulative seconds, including the charged
218 seconds from the interrupted attempt. Expert replay: 3/3. Frozen VLA:
prefix 25: 1/3; prefix 1: 0/3; prefix 5: 3/3. These are three deliberately
selected **training resets**, not a held-out generalization estimate.

Physical traces/chunk schedules/video hashes were checked before the reporter
rejected different initial predictions. The strict same-initial-prediction
gate remains failed. No gate was weakened; the promised final cadence plots
were not generated. Preserve the original `report.log`, `report_status.json`,
videos and all failed attempts. Prefix five is a candidate, not a proven fix.

## New evidence

- Four fresh workers (two default numerical settings, two deterministic) in
  `outputs/cadence_repeatability_20261021`: same actor, three resets, three
  identical-observation/noise predictions each, no physics steps or updates.
- Full loaded-state SHA-256 is identical in all four workers:
  `9676ba9f2b2285d9a8669a6c1f2cd6a6ce5851abab541d9e42acc3d255b01e1a`.
  Initial qpos/qvel/proprioception match. Every within-worker repeat is exact;
  the global CUDA RNG is not advanced by the explicit-noise calls.
- Raw camera images can differ across fresh render contexts even with equal
  physics. In the two default workers, changed RGB channel values are:
  `train_20003`: scene 7 / wrist 0; `train_22045`: scene 20 / wrist 39;
  `train_20002`: scene 3 / wrist 0. Each change is only **one uint8 intensity
  level**, out of 230,400 channel values per camera.
- PyTorch deterministic mode does not make fresh camera renders identical.
  It is not a rendering repair or evidence of real-time deployment readiness.

The controlled A/B test in `outputs/cadence_fixed_inputs_20261021_v2` feeds
those captured images into one frozen actor, holding state, instruction and
latent noise fixed. A/B/A/B repeats are exact for each input. Changing only
the RGB changes the full predicted chunk by:

| Training reset | Maximum joint difference (rad) | Maximum finger difference (mm) |
|---|---:|---:|
| train_20003 | 0.006846 | 0.0888 |
| train_22045 | 0.009128 | 0.2330 |
| train_20002 | 0.009889 | 0.4660 |

This proves an RGB sensitivity contribution. It does **not** prove it is the
only source of the historical mismatch or why a whole rollout fails. The
original campaign did not save raw initial RGB. Also, `train_20002` has a
remaining nonzero comparison against the earlier default-process predictions
when replayed in the new deterministic/call-order context (maximum 0.009889).
The first two parents reproduce those earlier predictions exactly. Qualify
same-fixture, same-numerical-mode cross-process/call-order behavior before
claiming the inference repeatability issue completely closed. Do not describe
this as a missing-weight, dropout, RNG or GPU-kernel bug without further evidence.

A second fresh process, `outputs/cadence_fixed_inputs_20261021_repeat`, now
repeats **all 21 saved arrays exactly** (A/B/A/B chunks and three native queue
prefixes for each parent), under the same deterministic settings and call
order as V2. This qualifies that fixed-input path across fresh processes.
It does not isolate which change in numerical mode/call order caused the
earlier positive-control discrepancy.

The initial A/B attempt is preserved in `outputs/cadence_fixed_inputs_20261021`.
It stopped after its first parent because its fixture validator incorrectly
assumed that only the scene camera differed. V2 explicitly measures both
cameras while requiring identical state. This was a diagnostic assumption
repair, not a policy, guard or physics change.

## Public inference API equivalence

Compared our selected prefixes with the installed LeRobot 0.4.4
`SmolVLAPolicy.select_action`, with `n_action_steps` set to 1, 5 and 25,
the same pre/postprocessors and explicit flow noise. All **nine** comparisons
(three parents times three prefixes) are bit-exact, including every queued
command. This rules out a prefix-slicing discrepancy in these fixtures; it
does not qualify the entire training pipeline or long closed-loop behavior.

The model itself already uses public `predict_action_chunk`; do not confuse
the OpenArm/safety/evidence bridge with a separately implemented SmolVLA model.

## Implemented observability repair

Future `scripts/rollout_smolvla.py` policy runs save exact raw initial state,
instruction and both RGB arrays to `first_actor_input.npz`, hash-bind that file
in the report, and add raw actor-input fingerprints to every predicted chunk.
`learning/vla/inference_inputs.py` validates and hashes; it never transforms
inputs, changes actions or forwards privileged simulator state. The previous
rollout source is archived for historical evidence. No weights, normalization,
sampling, guard, controller or simulator changes were made.

37 selected unit/regression tests pass, including eight repeatability/input
audit tests. Real-model checks are the separate nine native-queue comparisons.
The added recording also passed a one-step simulator smoke in
`outputs/cadence_input_logging_smoke_20261021`. Independently reloaded raw-input
fingerprints/file hashes, exact scene/model/command/noise bindings, physical
trace and the two-frame 50 Hz MP4 all audit correctly. This is one physics
step for logging qualification, **not a pickup trial**. No optimizer update,
new scored cadence campaign or held-out validation run occurred.

## Next: qualify the public training path, then train

1. Same numerical settings/saved inputs/call-order fresh-process replay and
   the logging smoke are complete. Still isolate **differing call order and
   numerical mode** one at a time. Record processed tensors/noise hashes as well as raw
   inputs; diagnose the remaining positive-control discrepancy, not just its
   effect on pickup. Do not replace live images with recorded images in a scored
   autonomous rollout. Do not blur/quantize images to force an audit pass.
2. Use the native LeRobot dataset, processors, policy, optimizer/scheduler and
   `lerobot.scripts.lerobot_train.update_policy`/training loop. Begin from the
   original cached `data/models/smolvla_base`, not a previously fine-tuned actor.
   Compare a tiny deterministic loss/gradient/update fixture and raw/exported
   timing before a long run. No new phase quota, sensor noise or recovery mix.
3. Keep only explicitly qualified compatibility changes: terminal-mask alias
   and FP32 trainable/master weights/Adam moments with BF16 forward compute.
   The existing `scripts/lerobot_smolvla_train.py` currently handles only the
   mask alias; it is **not yet** a qualified FP32 native-baseline launcher.
   Test the minimal precision adapter before recommending that command.
4. A clean nominal-data run uses the audited existing training split, native
   sampling and pretrained optimizer/scheduler presets. Choose a real
   hardware-fitting batch; smaller batches are not automatically batch 64.
   Obtain a new runtime/storage budget before a long run. Require reliable
   training-reset pickup and then held-out pickup before insertion/RL.

Prefer these public sources rather than expanding the custom trainer:

- [Official SmolVLA fine-tuning guide](https://huggingface.co/docs/lerobot/smolvla)
- [Pinned native trainer](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/scripts/lerobot_train.py)
- [Pinned native policy](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py)
- [Current official OpenArm integration](https://huggingface.co/docs/lerobot/openarm)

The current OpenArm guide is useful for later hardware integration, not proof
that its current version/calibration/units match our pinned simulator contract.
No in-place dependency upgrade or replacement of the existing scene/dataset.

Final machine-readable evidence:
`results/cadence_diagnosis_20261021_final/summary.json` and
`rgb_and_native_queue.csv`. These audit the saved fresh-process and causal
fixtures; they are not a replacement pass for the rejected cadence report.
