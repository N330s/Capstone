# SmolVLA next-step plan

Updated 2026-10-23. Owner focus: fine-tuning/evaluation first, then RL. Teammates
own collection and scene improvements; version their changes as new releases.
Current evidence: [training status](docs/TRAINING_STATUS.md).
Execution instructions: [lab handoff](docs/LAB_HANDOFF.md).

## 1. Qualify the lab PC — next action

- [ ] Transfer the frozen 81/20 export, raw release, exact recording-compatible
  collector/robot assets, tokenizer/base weights, original full81 final checkpoint,
  and optional single-episode positive-control checkpoint. Preserve checksums.
- [ ] Build the container; require all runtime/CUDA/video/EGL/storage checks to pass.
  Docker Linux GPU qualification is pending, not already established locally.
- [ ] Run export readback, native/reference two-update parity, then a 64-update
  native end-to-end smoke test with fresh checkpoint reload.
- [ ] Benchmark batches 8, 16, 32 sequentially, with real loader data. Try 2/4 loader
  workers after a stable baseline. Choose windows/sec with memory headroom, not
  the largest batch or highest momentary GPU utilization. Stop on OOM/nonfinite
  metrics; no automatic retries. Requalify after changing versions/precision.
- [ ] Reproduce expert replay and the trained single-episode live pickup on the lab
  platform. Repeat inference seeds before assuming cross-OS reliability.

Do not start another long run until these checks pass. Return the reports listed
in the handoff so the actual batch and runtime can be chosen from measurements.

## 2. Controlled full-dataset exposure experiment

- Start from **original full81 step 12,243**, not the single-episode-specialized
  actor. Keep 81/20 split, training-only normalization, RGB/state/action contract,
  scene, horizon 50/prefix 5, and evaluation bank fixed.
- The portable launcher does **weight warm-start with a new native optimizer and
  scheduler**, not exact optimizer-state resume. Record this confound explicitly.
  Exact native optimizer resume across platforms requires a separate qualification.
- Evaluate cumulative exposure near 1, 2, then 5 sample-equivalent passes. Count
  previous 97,944 windows plus new batch × updates; these are not unique examples.
  At batch 8, reaching one pass needs **6,195 additional updates**. Recalculate
  steps if the lab batch changes; do not keep both steps and exposure claims fixed.
- Keep one recent recovery checkpoint and milestone candidates as storage allows;
  never automatically delete historical/selected weights. The current launcher
  saves periodically/final but does not automatically implement retention.
- Use bounded sessions, initially at most four hours each. Longer runs require an
  explicit new runtime decision. No launch is authorized just by reading this plan.

At each milestone: finite losses/gradients, physical-unit generated-command errors
at approach/closure/lift, fixed-window validation flow loss, and **fresh live pickup
trials**. Report grasp, ≥20 mm lift, ≥1 continuous second carry, drops and safety
aborts. A validation curve alone cannot pass this gate.

## 3. Pickup generalization gate

- Reuse the five published training resets with fresh successful expert controls;
  repeat policy trials across prespecified inference seeds.
- Freeze checkpoint selection using training/validation only. Report per-reset
  success and failures; do not tune on held-out results.
- After a training-reset improvement, run the 20 held-out episodes (or a declared
  fixed subset for a quick screen, followed by the full bank). Separate whole-episode
  offline validation from live-reset evaluation. Historical held-out live trials
  were **not run**, not zero successes out of five.
- If greater exposure still cannot fit training resets: revisit generated-action
  precision, phase imbalance/multimodal demonstrations, control alignment and
  normalizer integrity. Then test one change at a time. If training works but
  held-out fails: inspect coverage and collect targeted varied/recovery episodes.

No arbitrary success percentage is claimed today; agree repeatability and safety
acceptance criteria before scoring the expanded bank.

## 4. Insertion → RL → sim-to-real (gated, not implemented)

Once pickup is repeatable, qualify transport/alignment/insertion and extend the
live task metric without weakening force/collision limits. Keep torque/depth for
separate future ablations; they are not current model inputs.

Only after a usable SFT actor exists: verify the intended Physical Intelligence
RLT method/reference implementation and license, define rewards/terminations,
action/log-probability interface and offline-to-online evaluation, then qualify
tiny RL updates before long runs. RLT integration is not available yet.

For eventual OpenArm v1 hardware: calibrate joint/gripper units, camera poses and
timing, establish a hardware safety envelope and supervised low-speed tests,
then assess randomized simulation and sim-to-real. Paused-time simulation videos
are not evidence that inference meets real robot control deadlines.

Historical decisions and old experiment commands are preserved in
[the archive](docs/archive/PLAN_HISTORY_20261023.md); they are not the active plan.
