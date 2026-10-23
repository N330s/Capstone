# SmolVLA: current evidence and unresolved issues

Updated 2026-10-23. The aim is reliable OpenArm v1 table pickup before full plug
insertion. Training is offline supervised fine-tuning; **RL/RLT is not implemented**.

## Results we can actually support

| Experiment | Observed result | Limit |
| --- | --- | --- |
| Native full dataset: 81 train / 20 validation episodes | 12,243 updates × batch 8 = 97,944 windows; finite training metrics and exact fresh 500-tensor reload | Only 0.664 sample-equivalent passes over 147,499 train frames |
| Fixed-window offline validation | Native flow loss 0.466116 → 0.004507 | 160 windows spanning 20 episodes, flow time 0.5; not all frames or generated-command accuracy |
| Source-matched live pickup, full81 model | Expert replay 5/5; autonomous model 0/5 training-reset pickups | Four no-grasp command limits; one collision safety stop; held-out live evaluation not run |
| Additional native one-episode training | 1,386 updates at batch 8 on all 1,673 rows of `train_20002`; live pickup 1/1, 68.794 mm lift and 1 s continuous carry, no drops/abort | One reset/seed; warm-start plus new optimizer; no generalization or insertion claim |

The successful trial was a **live closed-loop simulation**: newly rendered RGB,
measured robot state, model-generated actions and evolving physics. It was not
recorded-state playback. The expert control was replay, scored separately.
Both training experiments retained complete demonstrations, including later phases.

Reports, plots, machine-readable summaries and rollout recordings remain local,
not uploaded in this handoff. Their local locations are listed in
[results/README.md](../results/README.md). Detailed historical methodology:
[native training](NATIVE_SMOLVLA_TRAINING.md), [evaluation](NATIVE_SMOLVLA_EVALUATION.md),
[action diagnosis](NATIVE_ACTION_DIAGNOSIS.md), [one-episode test](NATIVE_SINGLE_EPISODE.md).

## What failed and what remains uncertain

The full81 actor has approach/closure errors even when given correct recorded
training observations: selected closure first-five finger-command RMSE was
1.43–4.94 mm. Four live failures closed an empty gripper to about 3 mm and then
entered states below the demonstrations' measured finger-travel minimum of
18.72 mm. This creates an unseen-state/recovery problem after initial motor errors.
The fifth case collided early. The trained actor is not merely failing on unseen
object positions: it also fails training resets.

The following checks reduced specific bug hypotheses, but **do not prove there
are no bugs**: raw/export state/action/timing/masks, training normalization and
physical-unit roundtrip, all 500 checkpoint tensors, exact repeated first inference,
compressed versus original RGB, and cached versus joint forward paths. Longer
denoising or changing action execution chunks did not establish a fix. The native
loss includes 32 dimensions (eight physical + 24 padded), so a small scalar loss
is especially insufficient evidence for millimeter-sensitive generated actions.

**Leading next hypothesis:** too little full-dataset optimization/exposure.
The one-episode positive control is consistent with this, but also changed data
diversity and reset optimizer/scheduler; it is not a controlled causal proof.
Demo variability, phase imbalance and recovery coverage can still matter.

## Next discriminating experiment

Qualify the 4090 lab host, then warm-start **the original full81 checkpoint** and
measure pickup near 1, 2 and 5 cumulative sample-equivalent passes while keeping
the dataset, normalization, environment and evaluation bank fixed. Record the
optimizer restart explicitly. More batch size is not a mathematical prerequisite:
batch 8 already passed the single-episode control. Pick a batch from real loader
throughput and memory tests; do not add unqualified gradient accumulation.

As a scale comparison, the [official SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla)
shows 20,000 updates at batch 64: 1.28 million windows, about 13.1 times our full81
run's exposure. This is a reference recipe, not a guaranteed requirement for this
task, and current upstream docs may differ from our pinned 0.4.4 implementation.

If training-reset pickup improves, evaluate held-out resets and repeat seeds. If
it does not, inspect phase-specific generated commands before collecting more
indiscriminate demonstrations or starting RL. See [plan.md](../plan.md).

## Binding and runtime facts

- Exact collector snapshot: `b7297ba82701fe7e7814c32ea617a85f832b74a4`;
  robot assets: `161039cd74ea8675fb8197836fe5674659825c75`.
  This is the recorded-compatible scene, not necessarily the latest GitHub scene.
- State: 16 measured joints/gripper values across both arms. Action: seven right
  joint absolute targets in radians + right finger travel in meters. RGB: scene
  and wrist, 320×240, 50 Hz; horizon 50, execute five then reobserve. Depth/torque
  are deliberately excluded from this baseline.
- Trainable expert/projection and Adam moments FP32; native Accelerator BF16
  compute; frozen vision backbone; no phase quotas or controller assistance.
- Measured local RTX 4060 Ti 8 GiB: about 1.114 s/update end to end at batch 8
  versus median ~0.309 s optimizer compute; loader/non-compute overhead matters.
  **No measured 4090 throughput yet.** Benchmark and estimate with
  `remaining_windows / measured_windows_per_second`, plus checkpoint/evaluation time.
- Prior speculative 4090 planning range was 20–40 windows/s, not a benchmark.
  From the original checkpoint, five cumulative passes would need ~4.5–9 hours
  at that rate, plus evaluation. Do not promise that runtime before lab measurement.
- Simulation runs pause during inference; 50 fps rollout videos omit those pauses.
  They do not establish real-time control, hardware safety or sim-to-real readiness.
