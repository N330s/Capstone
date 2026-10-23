# Lab handoff: native SmolVLA on RTX 4090

Updated 2026-10-23. For the teammate setting up the run. Start with infrastructure
and short checks; **do not launch a long job until those checks pass**.
Current limitation: full81 model = 0/5 live training-reset pickups; a separately
overfit one-episode model = 1/1. See [status and diagnosis](TRAINING_STATUS.md).

## 1. Checkout and stage inputs

```bash
git clone --branch chore/native-smolvla-lab-handoff https://github.com/N330s/Capstone.git
cd Capstone
```

Use the branch's final commit ID when recording the run. This repository does
**not** include raw datasets, weights or third-party source snapshots. Arrange
their transfer from the development PC separately; GitHub clone alone cannot run
the experiment. No lab passwords, API tokens or personal cache credentials belong
in Git or the image.

For the original Drive source folders, safe download instructions and the
recommended verified-release transfer, see [dataset access](DATASET_ACCESS.md).
Raw datasets, exports, model weights and new dataset-derived evidence are not
published on this branch.

Under an approved persistent storage root, stage these directories. The left-hand
paths below are relative to `C:/Users/Admin/plug_env` on the development PC;
the right-hand paths are relative to `OPENARM_STORAGE` on the lab host.

| Development source | Lab destination |
| --- | --- |
| `data/drive_v2c30_20261012/merged_all_v1/` | `raw/merged_all_v1/` |
| `data/drive_v2c30_20261012/export_all_v1/` | `datasets/export_all_v1/` |
| `data/models/smolvla_base/` | `models/smolvla_base/` |
| `data/models/smolvlm_backbone/` (config/tokenizer files only; omit `.cache`) | `models/smolvlm_backbone/` |
| `outputs/native_smolvla_20261021_v2/training/checkpoints/012243/pretrained_model/` | `models/full81_step012243/` |
| `outputs/native_single_episode_20261022_v2/training/checkpoints/001386/pretrained_model/` | `models/single_episode_step001386/` (positive control) |
| `third_party/capstone_automateData/` including its required robot assets | `assets/capstone_automateData/` |
| `third_party/openarm_mujoco/` | `assets/openarm_mujoco/` |

Also create writable `runs/`, `cache/` directories and existing `raw/`, `models/`,
`assets/`, `datasets/` parents. Copy **all** model processor JSON/safetensors with
each pretrained directory, not just weights. The optional full original native
checkpoint (including optimizer/RNG/training state) can be backed up separately;
the active launcher does not resume its optimizer. Preserve original inputs.

Measured current inputs: raw release ~8.67 GiB, export ~0.57 GiB, original base
~0.85 GiB. A full new native checkpoint including optimizer is ~2.04 GiB on this
recipe. Models, two robot-asset locations, image/build cache, repeated benchmarks,
videos and reserve add up; allocate **150–200 GB free** initially if available.
This is headroom for experiments, not a claim that the dataset itself needs that
much. Do not copy all historical runs or compressed archive duplicates.

Source/asset versions must match recordings, not the latest scene branch:

- Collector `b7297ba82701fe7e7814c32ea617a85f832b74a4`.
- Robot `161039cd74ea8675fb8197836fe5674659825c75`.
- Export receipt SHA-256 `cd3f7b7b02947f85dfbef4b46d218a932d753aa65bc1aa28412d3d5b819db63f`.
- Original full81 model SHA-256 `0cefdc6ffc0c634f17fe35a94d3940b7f15de2833c8e3142837b5a1a5e7df1de`.
- Positive-control model SHA-256 `6873ab243639b6b8df563dcb92fc8e29071d4d4abe4050627bb51341c9ef927f`.

Keep byte contents unchanged, including collector source line endings. Verify
transferred files with SHA-256 on both hosts; episode metadata and the readback
audit also bind raw arrays. If materializing a Git snapshot instead of copying,
disable automatic CRLF conversion **for that checkout**. Don't alter hashes to
make mismatched files pass. Model source revisions: `configs/smolvla_model_sources.json`.

## 2. Build and qualify the runtime

Follow [LAB_SETUP.md](LAB_SETUP.md): approved driver/Docker setup, `.env`, storage
permissions, `docker compose build smolvla`, then full container preflight.
Require `/runs/lab_preflight_001.json` → `passed=true`, with no skipped checks.
The development PC cannot currently build/run the Linux Docker image, so this
lab qualification is an actual pending step, not a formality.

The configuration `configs/lab_smolvla.json` uses **container paths**, not host
paths. Adjust the file before building if your staged names differ. Code/config
are baked into the image; rebuild after edits. Existing output names are refused.

```bash
docker compose run --rm smolvla python scripts/lab_smolvla.py check
docker compose run --rm smolvla python scripts/verify_smolvla_export.py --dataset /raw/merged_all_v1 --export /datasets/export_all_v1 --report /runs/export_readback_001.json
docker compose run --rm smolvla python scripts/lab_smolvla.py qualify --name native_parity_001
docker compose run --rm smolvla python scripts/lab_smolvla.py smoke --name smoke_b8_w0_001 --batch-size 8 --workers 0 --steps 64
```

Expected: exact frozen release hashes; readback `passed=true`; native/reference
two-update parity `native_reference_exact=true`; smoke `completion.json` with
finite updates, `reload_exact=true` and `throughput.windows_per_second`.
Smoke performs **real native data loading/training/save/fresh reload**, not only
a fixed synthetic batch. It creates a disposable diagnostic actor; the later
training run independently starts from the original full81 model.

`check` verifies release/model/source bindings but does not decode all videos;
the separate readback checks first/middle/final anchors of every episode, reports
RGB compression errors, and is not a frame-by-frame task-quality certification.

## 3. Measure usable throughput

Run sequentially, with no other GPU jobs. Do not blindly fill VRAM.

```bash
docker compose run --rm smolvla python scripts/lab_smolvla.py smoke --name smoke_b16_w0_001 --batch-size 16 --workers 0 --steps 64
docker compose run --rm smolvla python scripts/lab_smolvla.py smoke --name smoke_b32_w0_001 --batch-size 32 --workers 0 --steps 64
```

Stop increasing after OOM/nonfinite/resource failures; preserve failure reports.
Then test the best safe batch with `--workers 2` (and optionally 4), fresh names.
Compare end-to-end windows/s, `native_status.json` peak reserved VRAM, and resource
events. Keep at least ~1 GiB practical GPU headroom for the selected configuration,
not merely the 512 MiB emergency guard. The launcher does not auto-select a batch.
64 updates is a short screening estimate; confirm sustained throughput later.

Safeguards: GPU temperature <85°C, available RAM ≥1 GiB, GPU free ≥512 MiB,
**25 GiB disk reserve on the output filesystem**, plus estimated checkpoint-write
space. Exactly one visible GPU is required for unambiguous monitoring. Configure
Compose's selected GPU; host `CUDA_VISIBLE_DEVICES` alone does not filter every
`nvidia-smi` installation. No background deletion, automatic retry or multi-GPU
training is implemented. Finite metrics and guards do not guarantee system safety.

## 4. Reproduce physical controls before trusting a new training run

First replay the expert in fresh physics, then run the single-episode positive
control. This qualifies the new platform, not generalization. Separate output
names keep control and policy evidence distinct.

```bash
docker compose run --rm smolvla python scripts/rollout_smolvla.py --snapshot /assets/capstone_automateData --episode /raw/merged_all_v1/train_20002 --checkpoint /models/full81_step012243 --backbone /models/smolvlm_backbone --output /runs/expert_20002_001 --mode replay --max-steps 1200 --prefix 25 --physical-grasp-metrics --pickup-only
docker compose run --rm smolvla python scripts/rollout_smolvla.py --snapshot /assets/capstone_automateData --episode /raw/merged_all_v1/train_20002 --checkpoint /models/single_episode_step001386 --backbone /models/smolvlm_backbone --output /runs/positive_20002_seed1000_001 --mode policy --max-steps 1200 --prefix 5 --seed 1000 --noise-schedule step-keyed --numerics controlled --verify-first-prediction --physical-grasp-metrics --pickup-only
```

Expert report: `pickup_evaluation.passed=true`. Policy report:
`autonomous_pickup_success=true`, no safety abort/drop, repeated first prediction
equal. View actual `rollout.mp4` plus report/trajectory, not only terminal text.
Repeat policy seeds 1001/1002 with fresh names; a failed repeat is evidence to
inspect, not a reason to select only successful videos. Do not assume Linux/4090
physics or GPU numerics are bitwise identical to Windows/4060 Ti.

These rollout commands retain simulator safety limits but do not have the lab
training supervisor's independent wall timer. Run them attended initially;
the 1,200-command ceiling is simulated command count, not a wall-time limit.
Do not use `recorded-state`, `recorded-observations` or a teacher/gripper override
when reporting autonomous policy success.

## 5. Proposed first useful training run (after the above gates)

Use the original full81 checkpoint, **not** the single-episode actor. This is a
weight warm-start with a **new native optimizer/scheduler**, not an exact resume.
The normalization and split stay unchanged. Do not use an unqualified gradient-
accumulation flag: the pinned update loop steps optimizer/scheduler per batch.

There are 147,499 train frames and 97,944 windows already sampled by the original
actor. Additional updates to reach a cumulative sample-equivalent pass:

| Batch | About 1 pass | About 2 passes | About 5 passes |
| ---: | ---: | ---: | ---: |
| 8 | 6,195 | 24,632 | 79,944 |
| 16 | 3,098 | 12,316 | 39,972 |
| 32 | 1,549 | 6,158 | 19,986 |

Values are measured **from the original checkpoint**, not added successively to
each other. Sampling is shuffled windows with repeats, not unique-data coverage.
Choose a scheduler horizon deliberately; restarting new short schedules at each
milestone is a different experiment from one uninterrupted five-pass schedule.
The table gives exposure accounting, not a scheduler prescription. Inspect
intermediate saved checkpoints without restarting optimization where possible.

Example if batch 8 remains best: inspect the command first, then explicitly start
a four-hour-bounded run toward one cumulative pass:

```bash
docker compose run --rm smolvla python scripts/lab_smolvla.py train --name full81_to1pass_b8_001 --batch-size 8 --steps 6195 --dry-run
docker compose run -d --name openarm-full81-to1pass-001 smolvla python scripts/lab_smolvla.py train --name full81_to1pass_b8_001 --batch-size 8 --steps 6195
docker logs -f openarm-full81-to1pass-001
docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' openarm-full81-to1pass-001
```

Select measured worker count explicitly (`--workers 2` only after it passes).
For batch 32, use 1,549 updates, not 6,195, for the same first-pass exposure.
If time expires early, use actual completed updates in all exposure claims.
One detached container survives SSH disconnect; closing `docker logs` does not
stop it. Do not start a second job while the first runs. `docker stop` is an
emergency termination and does not guarantee a final checkpoint.

The supervisor's four-hour wall cap includes setup, save and fresh reload, with
120 seconds reserved from the training soft budget. An unusually slow save/reload
can fail the hard cap; preserve the report instead of assuming the checkpoint is
complete. Periodic/final native checkpoints are kept; no automatic retention or
exact cross-platform optimizer resume is implemented.

Observe `/runs/<name>/training/native_events.jsonl` for loss, gradient norm,
learning rate, update timings, resource samples and fixed-window validation.
`native_status.json` is only the latest event, not the full history. Native flow
validation is 160 fixed windows across all 20 validation episodes, not full-frame
validation. Long-run `completion.json` requires a fresh exact tensor reload;
failure is recorded separately. **No autonomous rollout is automatically scored
by the training launcher.**

## 6. Evaluate and send back evidence

Use the new checkpoint path printed in `<name>/completion.json`, replacing
`--checkpoint` in the live policy command above. Reuse these five training resets:
`train_20002`, `train_20003`, `train_20042`, `train_22001`, `train_22045`.
Run fresh expert controls on each, same prefix/numerical/noise protocol, then
live policies. Report failures and safety stops as well as successes. Only after
training-reset improvement expand to the frozen 20 held-out resets and repeat
seeds; do not label unrun cases failures or claim generalization from training resets.

Send back (no raw dataset or credentials needed):

- Git commit; lab OS/GPU/driver; image ID/digest and resolved requirements.
- Infrastructure preflight and export readback reports; parity report.
- Each benchmark's `completion.json` / `failure.json`, batch/workers, peak memory.
- Training plan, complete `native_events.jsonl`, completion/failure, checkpoint
  `openarm_native_complete.json` and fresh reload report.
- Each trial's `report.json`, `rollout.mp4`, and optionally trajectory/chunks for
  diagnosis. Include per-reset/seed pickup results, drops and safety aborts.

No learned insertion, RLT or sim-to-real success is established by these steps.
The next decisions are explicitly gated in [plan.md](../plan.md).
