# Pickup failure: where the first error starts

2026-10-20. Follow-up to [the state/coverage diagnosis](FP32_PICKUP_DIAGNOSIS.md).
No training, production-controller edits, source-scene edits, weaker safety
guards or physical robot commands. The original autonomous validation result
remains **0/5**. Short-prefix interventions below are not pickup evaluations.

## Main finding

The model's first forecast already closes the fingers too early. This happens
on recorded expert reset observations, before accumulated live-state drift.
Subsequent state feedback can amplify that initial error, as the previous
same-image/same-noise state intervention demonstrated. Arm-path error also
remains; changing execution prefix alone is not a complete fix.

There is a concrete supervision ambiguity at the start of the demonstrations:
an unobserved waiting timer determines different future chunks for almost the
same current observation. This is a plausible contributor to the initial
forecast error, **not proof that it is the sole cause**. No retraining ablation
has yet established how much removing that ambiguity helps.

![Initial forecast, live finger travel and arm-path deviation](../results/fp32_pickup_onset_20261020/onset_diagnosis.png)

## 1. Error exists in the first predicted chunk

Use the frozen final model, recorded reset observations and explicit latent
noise seed 1000. The demonstrator requests 44 mm throughout the first 25
commands. The model predicts only **38.44 mm** at chunk offset 24 on the training
reset (`train_20002`), and **38.34 mm** on the validation reset (`train_20004`).
First-25-command finger-target MAE is respectively **2.88 / 2.94 mm**; the first
command alone has just **0.064 / 0.175 mm** error. The forecast tail, not simply
the very first action, starts the premature closing. The original live first
chunks show the same direction of error.

These are individual noise-conditioned predictions, not average performance
over all latent draws. Seed 1001 changes severity. No seed is selected for
deployment or used to replace the original scored trials.

## 2. Near-identical observations have timer-dependent future targets

All **81 training parents** begin with **44 `survey` commands**: 3,564 initial
wait anchors in total. For the two inspected resets, state changes over this
period are at most about 3.4e-11 in stored units. Scene frame 0 versus frame 25
has mean absolute difference only 0.060 / 0.148 in 8-bit pixel units; wrist
images are even closer. These images are near-identical, not bit-identical.

Yet the future finger target at offset 24 differs with the anchor's position
inside that wait:

| Current survey anchor | Future target at offset 24 |
| --- | ---: |
| 0 | 44.0 mm |
| 25 | 41.6 mm |
| 43 | 35.0 mm |

The actor receives one observation (`n_obs_steps=1`), two RGB images, joint/
finger positions and the instruction. It receives no phase, timestamp or
history. Thus nearly indistinguishable inputs carry differing future labels
because the demonstration advances on a hidden timer. A conditional generative
model can represent different futures, but cannot reliably infer the correct
wait position from an unobserved timer. This supports testing the training
conditioning/sampling before collecting more of the same nominal trajectories.

The previous audit hash-verified all 81 source NPZs. This follow-up reads their
phase/state arrays to count initial waits; it does not claim another full hash
verification or an all-frame visual-equivalence audit.

## 3. Shorter execution prefixes do not solve both errors

Run only the first **200 commands** on the training reset, with actual live
state/RGB and unchanged environment/guards. All three tests complete those
200 commands without abort. They stop well before the full pickup gate.

| Commands executed before replanning | Final finger travel | Final TCP distance to nearest expert approach-path point |
| --- | ---: | ---: |
| 1 | 34.39 mm | 43.02 mm |
| 5 | 29.97 mm | 7.02 mm |
| 25 | 32.33 mm | 29.13 mm |

The nominal approach opening is 35 mm. Prefix 1 improves opening but still
deviates substantially in arm position. Prefix 5 follows the Cartesian path
more closely at the final sample but closes too far. This is not evidence for
promoting any prefix as a successful controller.

Explicit noise is shared at common command anchors (`seed=1000+anchor`), but
changing prefix changes intervening observations and noise draws. This is a
diagnostic feedback-frequency intervention, not identical full random streams
or a statistically powered success-rate comparison. No held-out prefix tuning.
Nearest-path distance measures Cartesian TCP position only, not orientation,
grasp alignment or eventual success. It separates some timing differences from
actual off-path drift; matching-index errors alone conflate both.

## 4. Boundary, repeatability and precision checks

- Reconstruct the unchanged bounds/slew filter on requested expert and original
  VLA commands: **maximum applied-action discrepancy is zero** on both resets.
  No extra/double-filter arithmetic error was found in these samples. Commands
  run at 50 Hz; a prefix of 25 spans 0.5 simulated seconds.
- Fresh reset state equals recorded state exactly on both resets. Instructions
  match. Source/fresh-render pixel differences are at most one 8-bit level,
  with mean differences below 0.00017. A gross reset/camera mismatch does not
  explain the initial error in these cases.
- Repeating identical preprocessed observations and explicit latent noise
  produces **bit-exact predictions**, with input tensor hashes checked. There
  is no observed hidden-state or fixed-noise nondeterminism in this probe.
- Tiny image changes can alter BF16 joint predictions by up to about 0.00685
  rad in these samples. Offline FP16 reduces this sensitivity to about
  0.00143-0.00175 rad, but **does not remove early closing**: seed-1000 initial
  first-25 finger MAE remains 2.60 / 2.88 mm. Precision may matter for fine
  alignment; it is not a demonstrated complete fix. FP16 is not deployed.

These checks narrow the suspects. They do not certify the entire software
stack as bug-free or exclude subtler visual/learning problems elsewhere.

## 5. Why a small reported loss is not enough

The pinned native SmolVLA loss averages **32 action channels**, whereas this
task has **8 real channels** and 24 padded channels. The diagnostic captures
per-element losses under `no_grad`, reconstructing the native mean to within
2.4e-9 on the inspected batches. This is the existing upstream convention,
not a newly introduced trainer bug or a training-loss change.

At the training reset, native loss is 0.01938 versus 0.07596 averaged over only
the 8 real channels. At an approach anchor it is 0.00162 versus 0.00547.
Across six inspected anchors, real channels contribute about 79-98% of total
loss: the objective is not simply dominated by learning padded zeros. But a
small aggregate number is not directly a small physical finger/pose error.
Phase-specific, physical-unit metrics should accompany future training.

## Recommended next experiment (not started)

1. Prepare a reproducible sampling-only ablation excluding initial `survey`
   anchors, leaving raw episodes/export/reset metadata and T+1 action alignment
   untouched. Check its exact reset-to-first-action behavior; do not simply
   delete frames and silently change the actor's time contract. Keep validation
   parents separate and the fixed final rollout bank unchanged.
2. Compare baseline and ablation from the same initial model, optimizer recipe,
   explicit training seed and fixed update budget. Log actual-8-channel loss,
   early forecast finger error, approach pose error and the existing native
   loss; do not rewrite historical logs or change the objective simultaneously.
3. Use training-control live pickup as a qualification gate, then the frozen
   held-out bank with expert controls and all original contact guards. Choose
   the checkpoint rule and execution prefix before evaluating; do not select
   a favorable held-out prefix or latent seed.
4. Use observed residual failures to collect a small training-only recovery
   pilot: synchronized actual RGB/state, safe reopening, hover/re-alignment
   and descent. The previously measured missing corrective-state coverage
   remains real. Broader nominal-only collection is not the first remedy.

The ablation tests a specific data-conditioning explanation. It is not a
guarantee that dropping waits alone fixes grasping. Arm alignment and corrective
coverage remain separate requirements. Insertion, RL and sim-to-real stay
behind repeatable autonomous pickup.

## Evidence and limits

- Raw probes: `outputs/fp32_pickup_onset_20261020`; repeatability subdirectory.
- Audited summary/PNG/PDF: `results/fp32_pickup_onset_20261020`.
- Scripts: `diagnose_pickup_onset.py`, `probe_pickup_repeatability.py`,
  `report_pickup_onset.py` under `scripts/`; helper `learning/vla/pickup_onset.py`.
- Main probe completed in 182.67 s; repeatability in 35.67 s, within separate
  900 / 300 s watchdogs. No long-running training was launched.
- 34 selected regression tests pass, including three new onset tests. This is
  not a claim that every repository test ran.
- Model: `outputs/fp32_multiepisode_20261020/training/checkpoints/step_0011300_final/pretrained_model`.
  Weights SHA-256: `9a6341162b9df3e8735a1e15d8bbde4acef4e11c79955d9a3b4a80cdfe5316f3`.
- Scene remains the pinned `third_party/capstone_automateData` snapshot at
  `b7297ba82701fe7e7814c32ea617a85f832b74a4`, not a claim of latest GitHub state.

Original datasets, checkpoints and scored live-trial artifacts are preserved.
