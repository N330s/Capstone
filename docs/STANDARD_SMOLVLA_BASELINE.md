# Establish an upstream-aligned SmolVLA baseline

Updated 2026-10-21. This replaces corrective-data training as the immediate next step.

Current progress supersedes the earlier pending stages below:
[NATIVE_SMOLVLA_TRAINING.md](NATIVE_SMOLVLA_TRAINING.md). Native update/data/
checkpoint qualification passed and a clean original-pretrained-base run is
running at batch 8, capped by the user's four-hour total allowance. The repaired
feature/mask/FP32 adapters delegate learning to the public LeRobot loop.
The prior frozen-actor inference-context discrepancy is still open; no online
pickup/generalization result is inferred from offline training loss.
Keep the qualified recovery bank, but do not use it to explain away a baseline
that has not yet achieved reliable training-reset pickup.

The user explicitly prefers usable public code over maintaining a custom
trainer. Adopt the official LeRobot training loop with only qualified robot
data/precision/mask adapters. See [CADENCE_DIAGNOSIS.md](CADENCE_DIAGNOSIS.md)
for completed evidence, a remaining cross-context mismatch and the next gates.

## References and what we will reproduce

- [SmolVLA paper, section 4.3](https://arxiv.org/html/2506.01844v1#S4.SS3):
  50-action prediction chunks; in simulation, obtain a new observation and
  predict again after **each executed action**; ten flow integration steps;
  train the action expert with the VLM frozen.
- [Official LeRobot SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla):
  start from `lerobot/smolvla_base`, use the upstream training path, and begin
  with around 50 task demonstrations, repeating each variation. Its example
  uses 50 episodes across five object positions, 20,000 updates and batch 64;
  smaller hardware-fitting batches are explicitly allowed.

The tutorial is not the exact paper training budget: section 4.3 reports
100,000 simulation fine-tuning steps at batch 64. Neither recipe guarantees
success on OpenArm. Our custom robot, contact-rich cabled plug, action bridge
and 50 Hz dataset are not the paper's LIBERO/MetaWorld environments. Do not
call any resulting number a reproduction of their benchmark score.

## Stage 1: isolate execution cadence without retraining

Implementation: `scripts/evaluate_inference_cadence.py`.
Campaign: `outputs/inference_cadence_20261020`; status is in `health.json`,
completed audited conditions in `conditions.json`, terminal status in
`completion.json` or `failure.json`.

**Current status: all 12 conditions completed**, under
`outputs/inference_cadence_20261020_resume_v1`. Its receipt re-audits/reuses the
completed control, preserves every interrupted file and restarts only the
uncompleted condition in the new directory. Prior runtime is conservatively
charged as 218 seconds; new preparation/execution shares the original one-hour
cap. Core actor/observation/scene sources remain identical. AST comparisons
verify the noise/prefix-audit algorithms are unchanged by monitoring and
continuation helpers. It finished in 2,364.015 cumulative seconds. No new training.

The bounded postprocessor checked saved motion, noise schedules, prior-time
ledger and preserved artifacts, then **rejected the initial prediction
mismatch** before generating final plots. `report_status.json`/`report.log`
retain that failure. No missing-trial scores: all planned conditions ran.
Expert replay passed 3/3; policy prefix 25: 1/3, prefix 1: 0/3, prefix 5: 3/3.
Do not promote these descriptive selected-training-reset counts to a controlled
cadence conclusion or generalization claim. The strict gate remains failed.
Fresh diagnosis and native queue parity are in
`results/cadence_diagnosis_20261021_final`; historical raw RGB was not captured.
Future rollouts save initial raw inputs/per-chunk hashes without modifying
actor behavior. 37 current selected tests pass; nine real-model native queue
comparisons are exact. A fresh-process fixed-input replication is exact across
all 21 arrays under the same numerical settings/call order; one earlier
cross-context discrepancy still needs mode/call-order isolation. One-step
logging smoke passes independent input/motion/video audits, not a pickup score.
No new training or software upgrade.

The preserved initial attempt was safely resource-stopped. The first
expert replay (`train_20003`, pickup at command 757) passed all physical,
exact-state, manifest and video audits. The second expert attempt was stopped
before its completion report; its trace/video files remain as interrupted
artifacts, not a scored failure or qualified control. All nine model conditions
were **not run** in that original attempt. Do not confuse its status with the
new continuation or infer an outcome from a pending condition.

Storage update: [checkpoint cleanup](CHECKPOINT_STORAGE.md) subsequently freed
4.462 GiB from four hash-identical weight copies; free space is about 36.26 GiB.
Every distinct model and optimizer state remains. The disk blocker is cleared,
and evaluation restarts only on the user's subsequent explicit instruction.
The readings and stop below
describe the preserved initial attempt.

The initial supervisor reported a generic resource guard at 217.812 seconds.
It did not record which individual threshold triggered. Immediate post-stop
inspection established a disk blocker: C: free space is 23.27 GiB, below the
25 GiB floor, versus 28.30 GiB at launch. The campaign wrote only 14.73 MiB;
the source of the roughly 5 GiB drop elsewhere is not established. At inspection
RAM was 12.64 GiB available, GPU free memory about 4.6 GiB and temperature 65 C.
Do not attribute the stop to model behavior or weaken the reserve. No user apps,
datasets, checkpoints or unrelated files were stopped/deleted/changed.

All owned processes exited. Need at least 26 GiB free at start (roughly 3 GiB
more free space) or an explicitly approved alternative storage target before
continuing. Preserve the completed control and interrupted attempt. A bounded
continuation must record the prior 217.812 seconds and preparation time; do not
silently reset the one-hour allowance or automatically retry after a stop.
The supervisor now writes named violations plus all resource measurements in
`safety_stop.json` for future stops; this monitoring-only repair is not a new
model attempt. Historical source versions are archived. 44 selected tests pass.

Frozen actor: five-parent expert overlay SHA-256
`f1518a2141cf382cafa5e893e12fe698470d0b5c2d62fb33500f8774d7546ac3`,
with its original base checkpoint/processors. **This actor is a diagnostic
subject trained with the earlier custom recipe, not a new standard baseline.**

Three fixed training parents: the two failures `train_20003`, `train_22045`,
and known positive control `train_20002`. Qualify fresh exact expert-command
pickups first on all three, then run all nine combinations of parent and
execution prefix **25, 1, 5**. Maximum 1,200 commands (24 simulated seconds)
per condition, same immutable weights, live state/RGB, command limits,
physical-grasp adapter and two-finger/20 mm/one-second/no-drop pickup gate.

The 50 Hz actuator loop is unchanged. Prefix 25 refreshes observations every
0.5 simulated seconds, prefix 5 every 0.1 s, prefix 1 every 0.02 s. **Do not
change training chunk size from 50 to 1**: prediction horizon and execution
prefix are different settings. No teacher takeover or state replacement.

Use explicit initial Gaussian flow noise from a CUDA generator seeded by
`1000 + executed_command_index`. This gives identical latent samples at common
prediction anchors, rather than allowing different numbers of inference calls
to consume unrelated portions of a global RNG stream. It does not force equal
subsequent physical observations, eliminate all stochastic effects or constitute
a multi-seed statistical study. Store and audit each chunk's seed and actual
executed prefix. The historical sequential-RNG 3/5 result remains unchanged;
fresh prefix-25 controls are **new conditions**, not claimed exact reruns.

The noise option is opt-in; ordinary rollout/controller default sampling is
unchanged. Pre-edit source versions are archived for historical reproducibility.

One-hour cumulative cap, hard per-process deadlines, 25 GiB disk reserve,
at least 1 GiB available RAM, at least 512 MiB GPU free and temperature below
85 C. No automatic retries or deletion. Preserve every failure; a process
timeout/incomplete condition is not a task failure score or an omitted success.
Do not extend the run automatically or proceed into training.

Independent checks after each trial: exact scene/guard manifests, reset and
source/weight bindings, saved physical pickup gate, requested command equality
to the selected chunk prefix, per-anchor noise schedule and every MP4 frame.
This remains paused-time simulation. Inference takes approximately 0.4 seconds
per call on this PC; prefix 1 is **not** a demonstrated real-time 50 Hz robot
controller. Later deployment requires measured async/RTC behavior, not pretending
simulation pauses are available on hardware.

Decision: if shorter prefixes improve reach/grasp, fix/qualify execution first.
If they do not, retain the negative evidence and inspect the learning/data path.
Do not pick the best prefix on held-out validation or infer that cadence alone
is the unique cause. All three parents remain training-only.

## Stage 2: native training/data equivalence qualification

Before another multi-hour run:

1. Start from the original cached `data/models/smolvla_base`, not an old local
   checkpoint that already went through custom sampling or legacy BF16 updates.
   Preserve all successful/failed historical actors as controls.
2. Use the pinned LeRobot training implementation. Compare its model forward,
   pre/postprocessing, loss, gradients and optimizer update with our boundary
   on identical observations, action chunks, flow noise and flow time. Check
   raw NPZ versus exported RGB/state/command timing, camera ordering, units,
   train-only stats, action names and episode-boundary masks.
3. Retain only the known, explicit compatibility fixes until tested otherwise:
   LeRobot 0.4.4 emits `action_is_pad` but this policy reads `actions_id_pad`;
   trainable expert/state projection and Adam moments must remain FP32 while
   BF16 is used for forward compute. Do not remove these repairs merely to
   claim an unmodified upstream command. No in-place site-packages edits.
4. Verify all intended state/action projections are trainable after adapting
   OpenArm's dimensions. The original state contains parked-left channels
   before active-right channels; do not assume its feature semantics match
   the pretrained SO100. Keep the current contract for a parity comparison.
   A minimal active-arm-only feature contract is a separately declared later
   ablation, not an inference-only channel reorder.
5. Do not silently upgrade the installed stack to the current website version.
   Record exact package/code/container versions. A version migration needs its
   own fixture, checkpoint reload and expert/policy execution qualification.

## Stage 3: clean task fine-tuning baseline

- Audit the existing 81 training / 20 validation parents before requesting
  new data. A five-episode fitting pilot is not a full task-training dataset;
  81 unsuccessful nominal episodes also do not establish that quantity is
  irrelevant. Assess repetitions and variation coverage, not frame count alone.
- Use ordinary upstream sampling/objective first, with **no** added gripper
  sensor noise, open-hold quota, recovery mix or deployment heuristic. Keep
  camera/instruction/action representation stable and measure training-reset
  approach accuracy as well as loss. Full-task insertion data remain separate
  from the pickup success metric; do not silently relabel or trim episodes.
- Prefer the pretrained optimizer/scheduler presets and a declared tutorial-
  scale budget (20k updates initially), with preflight and early execution
  checks. Do not launch it automatically from the cadence experiment.
- Fit actual VRAM first. The lab RTX 4090 has a different capacity from this
  PC's RTX 4060 Ti 8 GiB. The pinned CLI does not expose a training-config
  gradient-accumulation field in the inspected code: do not invent a flag or
  claim batch eight equals batch 64. Any accumulation adapter must be qualified
  for averaged loss, optimizer/scheduler steps and checkpoint resume. Otherwise
  use a fitting native batch and label that deviation honestly.
- Reserve disk for native model/optimizer/checkpoint writes above the 25 GiB
  floor before launch; current headroom is not permission to delete old work.
  Confirm runtime/storage budget for a full run separately.
- Require fresh reload agreement and reliable pickup on the original five
  training resets (with expert controls) before the fixed held-out pickup bank.
  Keep validation separate from recipe/prefix/checkpoint selection.

Only after this baseline is understood should we compare a small qualified
corrective-data mixture against matched nominal-only continuation. Insertion,
RL/RLT and sim-to-real remain downstream gates. The teacher's 4/4 recovery
result is retained evidence of feasible correction, not a substitute for VLA
pickup reliability or proof that recovery data is necessary.
