# pi0 adapter milestone

Implemented 2026-09-12. This is a tested local data/numeric integration, not a
fine-tuned policy, a completed LeRobot export, or a validated upstream runtime.

## Selected implementation target

- Official openpi source pinned at `215abfb217dbac7d5f1273282331b9b1866c0479`.
- Candidate checkpoint: `gs://openpi-assets/checkpoints/pi0_base`, JAX LoRA.
  No checkpoint downloaded or hashed yet. Source license is Apache-2.0; record
  checkpoint provenance and applicable weight terms when fetching the weights.
- Upstream pins LeRobot at `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`.
  Do not install an arbitrary latest LeRobot release for this integration.
- Local machine: RTX 4060 Ti, 8,188 MiB total VRAM, Windows. No openpi, JAX, or
  LeRobot installation in the current Python environment. WSL enumeration was
  denied, so Linux availability is unknown, not established as absent.

The pinned [openpi requirements](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md#requirements)
specify Ubuntu support and more than 22.5 GB for LoRA. A separate Linux GPU host
must be confirmed before training. The pinned PyTorch path does not support LoRA;
the presence of local PyTorch does not resolve that requirement.

## Contract

| Field | Adapter behavior |
| --- | --- |
| State | Left 7 joints + finger travel, then right 7 + finger travel; normalize 16 values, zero-pad to 32 |
| Action | Absolute right 7 joint targets + per-finger travel; normalize 8 values, zero-pad to 32 |
| Model output | Decode first 8 dimensions; remaining dimensions never command the parked arm |
| Scene image | `scene_rgb` -> `base_0_rgb` |
| Wrist image | `wrist_rgb` -> `right_wrist_0_rgb` |
| Missing left wrist | Black image with false image mask |
| Prompt | Instruction only; no privileged pose, force, phase, or timestamps |
| Horizon | 50 commands; a one-second prediction at 50 Hz, not a one-second execution commitment |

No ALOHA joint flips, gripper conversion, delta-action transform, or DROID mapping
is applied. `OpenArmCodec` owns normalization using the pilot training statistics
and its constant-dimension rule. Upstream `norm_stats` must be `{}` to avoid double
normalization. This explicit choice differs from upstream's usual normalization
ownership; include our normalization JSON and adapter hash in every checkpoint
and serving bundle. Never substitute base-checkpoint robot statistics.

The adapter accepts simulator HWC uint8 RGB and LeRobot-style CHW float RGB in
[0,1]. Upstream remains responsible for 224x224 resizing, tokenization, and model
image preprocessing; those operations have not been runtime-tested here.

## Terminal windows and data path

The audited [pi0 loss](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)
does not accept the pilot's action mask. `FullWindowPilotDataset` therefore uses
only complete same-episode windows. There are 1,968 training windows and 492
validation windows at horizon 50. The final 49 start positions per episode are
excluded, but their actions still appear as targets in earlier complete windows,
including the last window ending at seating/hold. This changes sampling weights;
it is a smoke-test strategy, not a substitute for later masked-loss support.

`openpi_integration.transformed_pilot` explicitly connects this raw-pilot dataset
to upstream transforms. It is NOT wired into upstream's stock LeRobot loader or
trainer. This is a deliberate intermediate step to test contracts without installing
a large unsupported training stack. A future launcher must consume the filtered
dataset or use a validated version-pinned LeRobot export with equivalent filtering.
Never pass unfiltered terminal windows into the stock trainer.

## Validation and known failed check

```powershell
python -m unittest discover -s tests -v
python scripts/validate_vla_adapter.py --replay --output results/vla_adapter_new_run
```

The output directory must not exist. The script checks pinned source identity,
episode checksums, train-only statistics, split separation, window boundaries,
camera/sample shapes, and action round-trip. It saves an index, normalization,
configuration and provenance report without changing original episodes.

- 23 unit tests pass, including all 16 prior physical/interface tests.
- All 20 episodes pass numeric conversion; maximum action error is 2.12e-9.
- Float32-transport replay of episode 0 succeeds in the original 172 commands.
- Maximum qpos error is 1.47e-6 (mixed joint units).
- The strict qvel tolerance of 1e-3 **fails**: peak difference is 2.16e-3 rad/s
  in the free plug's angular velocity at command 167, during seating.
  Right-arm joint velocity error stays below 9.90e-6 rad/s; plug linear velocity
  error below 3.95e-5 m/s. Final maximum qvel error is 3.33e-4.

See `results/vla_adapter_v1/report.json` and `float32_diagnostic.json`. Overall
`passed` remains false; neither the physics nor the tolerance was relaxed.
The earlier lossless replay remains valid. Float32 model transport is a distinct
test and should not be described as exact replay or actual model inference.

## RLT feature audit

The source provides `embed_prefix`, contextual prefix outputs inside the
transformer, and action sampling. `embed_prefix` alone returns pre-transformer
embeddings; it is not automatically the paper's desired contextual representation.
The existing policy endpoint returns actions, not learned RL tokens. Feature
extraction, selection of layer/tokens, frozen-gradient checks, reconstruction
training and reference-action equivalence remain unimplemented. Do not claim
RLT compatibility solely because internal model source is accessible.

## Next actions

1. Confirm a suitable Linux GPU host; pin its dependency environment without
   touching the Windows simulator environment.
2. Characterize float32 contact sensitivity over additional held-out seeds and
   half-timestep replay. Decide dimension-specific physical tolerances from those
   results and calibration needs; retain the original failed strict comparison.
3. Runtime-test `transformed_pilot` through actual upstream preprocessing and
   model observation construction, then wire the filtered dataset into a tiny
   training launcher. Alternatively complete and round-trip the pinned LeRobot
   export; keep splits separate and normalization train-only.
4. Overfit one or two episodes, save/reload the bundle, then test closed-loop
   serving with measured latency. Execution prefix 1 is only a diagnostic
   candidate; do not claim 50 Hz real-time VLA inference.
5. Expand demonstrations only after the contracts and rollout evaluation work.
   RLT and physical execution remain later gates in [plan.md](../plan.md).
