# OpenArm SmolVLA training

Simulation data, native LeRobot SmolVLA fine-tuning, and physical pickup evaluation
for OpenArm v1: two arms, right arm active, left arm parked. The final task is
**plug on table → grasp → insert**. Current work is reliable pickup before insertion,
RL or sim-to-real.

## Start here

- [Lab PC handoff: inputs, commands, checks and next experiment](docs/LAB_HANDOFF.md)
- [Where to obtain the dataset and how to use it](docs/DATASET_ACCESS.md)
- [Docker/host setup](docs/LAB_SETUP.md)
- [Current evidence and unresolved problems](docs/TRAINING_STATUS.md)
- [Next-step plan](plan.md)
- [Active code versus historical experiments](docs/CODE_LAYOUT.md)

## What works, and what does not

| Test | Result | Meaning |
| --- | --- | --- |
| Native training/save/fresh reload | Passed; 12,243 updates at batch 8 | Software and numerical positive checks, not proof of a bug-free system |
| Full 81-episode model, five training resets | 0/5 live pickups; expert controls 5/5 | Reliable learned pickup is **not solved** |
| Same native recipe, additional single-episode training | 1/1 live pickup; 68.8 mm lift, 1 s carry | Memorization positive control only; repeatability/generalization unknown |

The full run sampled only **0.664 equivalent dataset passes**. More optimization
is worth testing, but insufficient training is a hypothesis, not the proven sole
cause. Small validation flow loss did not predict task success.

## Active workflow

Use `scripts/lab_smolvla.py` with [configs/lab_smolvla.json](configs/lab_smolvla.json).
It checks the frozen release, delegates optimization to **LeRobot 0.4.4's native
trainer**, bounds wall time, refuses output reuse, and verifies a fresh model reload.
See the handoff before running: raw data, model weights and the matched collector
snapshot are transferred separately; cloning this repository is insufficient.

The Docker recipe has not yet passed a Linux GPU/EGL build-and-run qualification.
New portable launcher changes have local tests, not a completed 4090 training run.
No API token, Hub upload, W&B upload, checkpoint deletion or automatic retry is
needed. Existing source-matched safety checks remain enabled.

Historical baselines and experiments are retained for traceability, not as competing
default training recipes. See [the archived README](docs/archive/README_20261008.md)
and [experiment history](docs/archive/PLAN_HISTORY_20261023.md).
