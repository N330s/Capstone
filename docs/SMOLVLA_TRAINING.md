# SmolVLA implementation: first integration slice

## Closed-loop integration executed (2026-10-04)

The saved update-24 checkpoint now runs through the source-matched teammate v3
environment. It executed 12 commands before the existing collision guard stopped
a finger/socket-frame contact before grasp. A 100-command recorded expert control
from the same reset reproduced qpos/qvel exactly without robot collision. The
inference-to-simulation interface works, but autonomous task success and real-time
serving are unqualified: warm prediction averages 0.304 s for a 0.10 s prefix.

See [SIM_POLICY_ROLLOUT.md](SIM_POLICY_ROLLOUT.md) for commands, provenance,
reports, video, contact diagnosis and limitations. Next: full-window supervised
training, process-level resume and repeatable policy evaluation, before RLT.
Older sections below retain the status of their respective experiments.

## Multiple samples, validation and resume passed (2026-10-04)

`scripts/qualify_smolvla_training.py` completed 24 stochastic flow-matching updates
using 12 spaced windows across all three training episodes. Four windows from
`train_00003` form the held-out loss check. All preprocessing statistics come
from the training dataset. Evaluation uses fixed noise/time so before/after
comparisons are repeatable; training samples fresh noise/time each update.

| Selected-window mean flow loss | Before | After |
| --- | ---: | ---: |
| Training, 12 windows | 0.323525 | 0.245566 |
| Validation, 4 windows | 0.381867 | 0.292170 |

Evidence: [qualification report](../results/smolvla_training/qualification_v2.json).
This small diagnostic sample spans one held-out episode; the loss improvement
does not establish autonomous task success or broad generalization.

At update 12, saved the model, processors, AdamW moments, StepLR state, Python/
NumPy/CPU/CUDA random states, sample order and next step. Then ran updates 13-15,
reloaded the saved state, and repeated those updates. Losses, gradient norms,
learning rates, samples, trainable weights, optimizer state, scheduler state and
random states all match **exactly**. Training continued to update 24 from the
restored branch. This validates reconstruction from saved state on this runtime;
it does not yet test a separate-process resume CLI or a changed hardware/runtime.

Checkpoints and event log:

- `outputs/smolvla_qualification_v2/resume_checkpoint` (update 12).
- `outputs/smolvla_qualification_v2/final_checkpoint` (update 24).
- `outputs/smolvla_qualification_v2/events.jsonl` (includes duplicated updates
  13-15 for the resume comparison; report counts 24 unique updates + 3 comparisons).

Peak PyTorch-allocated GPU memory was 2.14 GB on the RTX 4060 Ti; the measured
interval was 80.4 seconds, excluding process imports/downloads. The loop uses
batch one, BF16 autocast, AdamW at 1e-5, and a short diagnostic StepLR schedule.
It keeps deterministic CUDA algorithms enabled for the resume comparison.

The first qualification attempt exposed an action batching bug: a raw sample has
an Hx8 action chunk, while stochastic model training expects BxHx8. Fixed-noise
broadcasting in the earlier smoke test had hidden it. Both scripts now collate
samples before the upstream processors. The failed v1 directory is retained.

Repeat with a fresh output root:

```powershell
$env:HF_HOME = 'C:/Users/Admin/plug_env/data/hf_cache'
.venv-smolvla/Scripts/python.exe -u scripts/qualify_smolvla_training.py --export data/collection_smolvla_smoke_v2 --checkpoint data/models/smolvla_base --backbone data/models/smolvlm_backbone --output outputs/smolvla_qualification_repeat --steps 24
```

Checkpoint state loads are for trusted, locally generated files, including Python
and NumPy random-state objects. `learning/vla/training_state.py` centralizes RNG
capture/restoration and exact nested-state comparison. Source dataset and sample
plans are checked against the saved export receipt when restoring.

Next milestone is policy inference through the teammate's matching v3 environment:
load the final model/processors, convert observations using the shared adapter,
execute short action prefixes through the existing bounds/rate-limited controller,
and record failure phase, latency, contacts and completion. Preserve the small
training dataset split; more demonstrations and actual rollout results determine
when to scale training and start RLT.

## GPU training smoke test passed (2026-10-04)

The real pretrained model completed three updates on the midpoint sample of
`train_00000` from the verified export. The isolated environment now has
PyTorch 2.10.0+cu128 / torchvision 0.25.0+cu128; the original simulator environment
is unchanged. Official model revisions are recorded in
`configs/smolvla_model_sources.json`; no architecture reduction was made for the
local RTX 4060 Ti. Used batch one, BF16 autocast, AdamW at 1e-5 and fixed
flow-matching noise/time for the fit check. This is a diagnostic loop using the
upstream model/processors, not qualification of the full upstream training CLI.

Evidence: [training report](../results/smolvla_training/smoke_v3.json).

- Loss before successive updates: 0.231196, 0.189250, 0.155089; after: 0.125197.
- 99,880,992 trainable parameters; checked parameter updates and absence of
  gradients on frozen parameters. Gradients and losses were finite.
- Terminal loss consumed the padding mask. LeRobot 0.4.4's SmolVLA forward reads
  `actions_id_pad`, but the dataset supplies `action_is_pad`. The explicit shim
  in `learning/vla/smolvla_compat.py` bridges that spelling for the smoke loop.
  The regular launcher now uses `scripts/lerobot_smolvla_train.py`, applying the
  same mapping in-process without editing the installed package. Re-audit on
  version upgrades; the regular trainer itself has not yet been exercised.
- Saved and reloaded all model weights exactly. Reloaded numeric preprocessing
  agrees exactly. Reloaded model plus postprocessor emits finite `(1, 50, 8)`
  physical-unit action chunks. Commands were not sent to the simulator/robot.
- Peak PyTorch-allocated GPU memory: 2.09 GB; measured test interval: 64.2 seconds
  (excludes dependency/model downloads and process import overhead). This is not
  total GPU usage or a full training-throughput estimate.

Checkpoint/processors: `outputs/smolvla_training_smoke_v3/pretrained_model`.
Optimizer state is saved separately, but exact interrupted-job resume has not
been tested. Earlier v1/v2 outputs are retained; v2 had a checkpoint but no final
verification report, so it is not recorded as a completed test.

Repeat with a fresh output directory:

```powershell
$env:HF_HOME = 'C:/Users/Admin/plug_env/data/hf_cache'
.venv-smolvla/Scripts/python.exe -u scripts/smoke_smolvla_training.py --export data/collection_smolvla_smoke_v2 --checkpoint data/models/smolvla_base --backbone data/models/smolvlm_backbone --output outputs/smolvla_training_smoke_repeat --steps 3
```

This verifies data -> preprocessing -> gradient updates -> save/reload -> decoded
actions. It does **not** establish validation/generalization, task success or a
working closed-loop controller. Next: a multi-sample tiny-fit run with held-out
loss and exact resume checks, then inference through the matching v3 scene and
controller. Portable university bundles also need rebasing the saved absolute
backbone/tokenizer paths to that machine. RLT remains after the supervised gates.

The older entries below are chronological records, superseded where noted above.

## Real export/readback completed (2026-10-03)

Installed LeRobot 0.4.4 in the isolated `.venv-smolvla` environment. `pip check`
passes; the simulator's original Python environment is unchanged. This installation
resolved to CPU-only PyTorch 2.10.0, so it qualifies data processing only, not CUDA
training. The launcher now explicitly rejects execution without CUDA.

The first export found a real codec API mismatch: LeRobot accepts `h264`, not
`libx264`. Fixed the exporter and added a regression assertion. The incomplete
`data/collection_smolvla_smoke` directory was retained; the successful export is
**`data/collection_smolvla_smoke_v2`**, approximately 38 MB for 7,251 rows and both
camera streams. Original NPZs and selection are unchanged.

Real readback passes for the first, midpoint and last command in each of all four
episodes: physical-unit states/actions agree to 1e-6 absolute tolerance, task
strings agree, camera tensors decode with the expected shape/range, and action
padding masks agree (one valid action at every terminal row). This checks 12
samples, not every frame. Per-camera RGB mean absolute compression error ranges
from approximately 1.6 to 11.5 on a 0-255 scale. This is not a visual quality gate;
inspect small connector details before scaling lossy exports.

Evidence: [readback report](../results/smolvla_export/readback.json), including
resolved package versions and CPU/CUDA status. The SmolVLA configuration imports,
but model weights, forward/backward, normalization through the actual processors,
save/reload and closed-loop evaluation are **not yet tested**. Ten adapter unit
tests pass. No model download or training was performed.

Repeat readback (choose a new report filename to preserve existing evidence):

```powershell
$env:HF_HOME = 'C:/Users/Admin/plug_env/data/hf_cache'
$env:HF_HUB_OFFLINE = '1'
.venv-smolvla/Scripts/python.exe scripts/verify_smolvla_export.py --dataset test_collection_new_env --selection configs/smolvla_collection_smoke.json --export data/collection_smolvla_smoke_v2 --report results/smolvla_export/readback_repeat.json
```

Next milestone: a CUDA-capable university environment, pinned local checkpoint
and backbone/tokenizer artifacts, then a one-batch forward/backward and short
tiny-fit/save/reload test. Use `data/collection_smolvla_smoke_v2` as `--export`.
Do not treat the installed CPU dependency set as a university CUDA lockfile.
The older status entries below describe earlier implementation boundaries.

## automateData dataset integration (2026-10-03)

Inspected `automateData` at `b7297ba82701fe7e7814c32ea617a85f832b74a4`
without switching branches or merging the teammate's work. The writer stores
lossless training RGB in NPZ; numbered JPEG files are viewing copies, not the
training source. Source metadata uses `openarm_v1_joint_targets_v3`, with the same
16-state/eight-action ordering, explicitly checked by the new reader.

`test_collection_new_env` contains four complete episodes. Its manifest also
lists `train_00005`, which is missing, and assigns every episode to training.
The explicit `configs/smolvla_collection_smoke.json` selection uses episodes
00000-00002 for training (5,501 commands) and 00003 for validation (1,750 commands).
The original manifest is untouched, the missing episode is recorded as excluded,
and selection provenance travels in the export receipt. Without a selection,
missing episodes are errors, not silently skipped. This tiny split is for pipeline
verification only, not generalization estimates or an untouched final benchmark.

Validated all four selected episodes, their checksums and 50 Hz timing (including
partial terminal intervals). Recorded phase labels span pickup and insertion;
labels alone do not prove physical replay or visual quality. Ten adapter tests
pass. Actual LeRobot export, model forward/backward, checkpoint reload and
closed-loop policy execution are still pending, so the complete pipeline is
**not yet verified**.

```powershell
python scripts/prepare_smolvla.py --dataset test_collection_new_env --selection configs/smolvla_collection_smoke.json
```

On the qualified LeRobot environment, export without altering the source:

```powershell
python scripts/prepare_smolvla.py --dataset test_collection_new_env --selection configs/smolvla_collection_smoke.json --export data/collection_smolvla_smoke
python scripts/train_smolvla.py --export data/collection_smolvla_smoke --checkpoint C:/models/smolvla_base --output outputs/collection_smoke --steps 100
```

Replace the checkpoint path with a pinned local model bundle. Add `--run` only
after reviewing the dry run and qualifying the runtime. Four episodes are enough
to exercise plumbing, not a reliable task policy. Acceptance remains: real dataset
readback with mask checks -> finite forward/backward and improving tiny-fit loss
-> checkpoint save/reload -> finite Hx8 physical-unit inference -> a short rollout
using the **matching v3 scene/controller**, not the older local insertion scene.
Retain exact raw terminal durations for eventual RL; the BC export alone is not
an RL replay archive. The small smoke launch does not yet evaluate validation loss.

The current local machine has CUDA-enabled PyTorch but no LeRobot installation.
The university 4090 remains the planned training target. No package installation,
model download or actual training is implied by source validation.

Implemented 2026-09-27. This is a data boundary and upstream smoke-run launcher,
not yet a trained policy, closed-loop evaluator, resume-tested trainer or RLT loop.
The scene and automated collector are intentionally unchanged.

## What works locally

- `learning/vla/smolvla_data.py`: checks successful scripted-expert episodes,
  checksums, episode/split identity, scene/controller consistency, RGB/state/action
  shapes and timing. Rejects duplicate contents and unsafe paths. Current input is
  `openarm-pilot-v1`; agree an additional reader if the teammate delivers another
  schema. Demonstration phase coverage is not inferred from successful insertion.
- Raw SmolVLA samples: two camera views, 16 state values, eight absolute right-arm
  commands and a task string. No pi0 preprocessing, numeric normalization, model
  padding, privileged poses, reward or phase labels enter the model boundary.
  Upstream processors own normalization and model-specific padding. A terminal
  `action_is_pad=True` means exclude that action, opposite to the old valid mask.
- Source commands remain at 50 Hz. A positive partial final interval is accepted
  because success can stop physics early; intermediate variable intervals fail.
  Exact durations and T+1 terminal observations remain in the raw source. The BC
  export contains T real command rows, never a fabricated terminal action.
- Optional local LeRobot exporter writes train/validation as separate datasets,
  preventing validation data from contributing to training normalization. A
  provenance receipt appears only after both finalize. Existing output is refused.
- Smoke launcher validates local export/checkpoint metadata, prints an argument
  list by default and executes only with `--run`. No automatic model downloads,
  Hub uploads or W&B uploads. The shared simulator still owns action limits.

## Commands

Lightweight validation (no torch/LeRobot/model required):

```powershell
python scripts/prepare_smolvla.py --dataset data/openarm_v1_varied10
python -m unittest discover -s tests -p test_smolvla.py -v
```

The existing varied pilot passes with 8 training episodes / 1,401 commands and
2 validation episodes / 345 commands. It is **held-plug insertion only**, useful
for integration but insufficient to teach table pickup.

On the university machine, create a separate virtual environment and install
`requirements-smolvla.txt` after confirming the compatible CUDA/PyTorch setup.
LeRobot **0.4.4** is the release-pinned integration target. A fully resolved host
dependency lock, checkpoint revision/license and backbone/tokenizer cache hashes
remain to be captured during runtime qualification. Do not reuse openpi's pins.

```bash
python -m pip install -r requirements-smolvla.txt
python scripts/prepare_smolvla.py --dataset /project/raw/pilot --export /project/datasets/pilot_smolvla
python scripts/train_smolvla.py --export /project/datasets/pilot_smolvla --checkpoint /project/models/smolvla_base --output /project/runs/smoke_001
```

The last command only prints the planned invocation. After reviewing it, append
`--run` to execute a 100-step, batch-one CUDA smoke run. The local checkpoint must
contain `config.json` and `model.safetensors`; pre-stage its tokenizer/backbone
dependencies too. Execution sets offline flags, so missing artifacts fail instead
of downloading unexpectedly. File presence is not a weight-integrity check: pin
and hash the model bundle before claiming reproducible training.

The initial run uses float32, zero loader workers, frozen vision encoder,
action-expert/state-projection updates and a 50-command prediction chunk. Ten
execution steps is a placeholder, **not** a contact-safe deployment recommendation.
Profile memory before enabling mixed precision or increasing batches. Only the
final smoke checkpoint is scheduled; there is no automatic pruning. The launcher
does not yet implement resume or gradient accumulation. Offline validation loss
and closed-loop evaluation are separate pending work; `eval_freq=0` avoids
pretending the custom environment is registered upstream.

## Required next acceptance tests on the university host

1. Run a real upstream export/readback; inspect encoded slot/blade visibility,
   camera order, state/action values and terminal action masks. Export integration
   is currently tested with a fake writer, not real video codecs or LeRobot.
2. Verify model input features are inferred from the OpenArm dataset, processors
   use training-only statistics, loss masks exclude terminal padding, and forward,
   backward, save/reload and physical-unit inference commands work. Merely building
   the command locally does not prove compatibility of the pretrained bundle.
3. Record peak GPU/RAM, decode speed, training throughput and checkpoint size.
   Add interrupted-job resume and test optimizer/scheduler/RNG/data restoration.
4. Obtain 5-10 synchronized **full-task** episodes from the collection teammate.
   Overfit 1-2 training episodes, then build the phase-aware rollout evaluator with
   the scene teammate's reset/step interface. Preserve an untouched benchmark.
5. Expand to supervised training only after these checks, then implement the RLT
   feature/token interface. No online RL or physical robot commands are provided
   by this first slice.

API references inspected for this integration target:
[dataset writer](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/lerobot_dataset.py),
[SmolVLA configuration](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/configuration_smolvla.py),
[training configuration](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/configs/train.py).
