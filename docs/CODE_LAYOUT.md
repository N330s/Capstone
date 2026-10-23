# Code layout and supported direction

The active model is **SmolVLA using pinned public LeRobot 0.4.4**, not a new
optimizer/loss/trainer maintained here. Repository cleanup keeps old paths intact:
saved experiment reports bind source filenames and hashes to results.
Validation of this cleanup is recorded [here](VALIDATION_20261023.md).

| Area | Current role |
| --- | --- |
| `scripts/lab_smolvla.py`, `configs/lab_smolvla.json` | Portable lab check / parity / smoke / bounded training front door |
| `scripts/train_native_smolvla.py` | Native trainer entry point, telemetry, budgets, validation and save checks |
| `learning/vla/native_smolvla.py` | Small embodiment/mask/local-tokenizer/FP32 compatibility adapters |
| `learning/vla/lab_training.py`, `runtime_resources.py`, `process_control.py` | Command planning and portable owned-job safeguards |
| `scripts/verify_smolvla_export.py`, `learning/vla/smolvla_data.py` | Raw/export alignment and fixed data contract |
| `scripts/rollout_smolvla.py`, `learning/vla/smolvla_policy.py` | Live controller and source-matched rollout; replay/assisted modes explicitly labeled |
| `learning/vla/physical_grasp.py`, `pickup_evaluation.py`, `inference_inputs.py` | Physical pickup gate and actor-input audits |
| `Dockerfile`, `compose.yaml`, `docker/`, `scripts/container_preflight.py` | Lab runtime and qualification |
| `tests/` | Boundary, adapter, numerical, evaluation and historical regressions |

## Historical/reproduction tools — not default training entry points

`finetune_smolvla.py`, `finetune_recovery_smolvla.py`, `overfit_smolvla_episode.py`,
`run_stable_smolvla_campaign.py`, `train_gripper_feedback.py` and recovery/onset/
teacher/cadence/five-episode scripts reproduce earlier diagnostic recipes. Do not
mix them into the current native baseline or reuse their sampling/controller
heuristics without a declared ablation. The pi0 adapter and earlier two-blade
scenes are preserved baselines, not the current SmolVLA scene.

`run_native_smolvla_campaign.py`, `run_native_single_episode.py`,
`evaluate_native_smolvla.py`, and dated reporter/probe scripts reproduce specific
local experiments. Their fixed paths/cases are historical; use the portable lab
launcher and explicit rollout arguments for new lab jobs. `qualify_native_smolvla.py`
now accepts explicit lab paths but intentionally validates this frozen release.

The native training adapter's model/loss/optimizer behavior was not changed by
this cleanup. Entry-point portability and supervision changed, so lab-native
parity plus train/save/reload must pass again. The single-episode monitor retains
its historical default paths but accepts explicit paths for portable use.

## Storage and publication

- `data/`, `outputs/`, virtual environments and third-party snapshots remain local,
  ignored inputs/artifacts. No training checkpoint or raw camera recording goes into Git.
- New `results/` reports and `docs/evidence/` are ignored. No new dataset-derived
  plots, videos, CSVs or raw evidence are published by this handoff. Previously
  tracked baseline reports remain tracked; remote history is not rewritten.
- [Local evidence index](../results/README.md) distinguishes live pickup from
  replay and offline probes. Historical docs may point to local-only reports.
- Earlier README/plan/setup text is archived in `docs/archive/`; dated diagnostic
  notes elsewhere in `docs/` describe those experiments, not current instructions.

Do not mass-delete experiments/checkpoints to tidy the tree. Future removal should
be a separate reviewed change with dependency checks and evidence backups.
