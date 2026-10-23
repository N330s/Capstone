# Bounded local SmolVLA training

## New FP32 multi-parent campaign (2026-10-20)

See [FP32_MULTIEPISODE_TRAINING.md](FP32_MULTIEPISODE_TRAINING.md).
The new offline supervisor at `outputs/fp32_multiepisode_20261020` compared
12 uninterrupted GPU updates with an exact 8+4 resume at batch eight: **passed**,
including FP32 moments/RNG and complete update history, with 2,636 MiB of
conservative device headroom. It has started the corrected full-data baseline. Qualification
and SFT share four hours; the declared final checkpoint is automatically tested
on the unchanged pickup validation bank. This is not continuation of the legacy
BF16 optimizer or a reuse of its batch-16 qualification. Read the new campaign's
`health.json`, `qualification/exact_resume.json` and terminal `completion.json`
for current state; the older campaign sections below are historical.

## Precision correction (2026-10-19; supersedes historical batch qualification)

See [TRAINING_PRECISION_FIX.md](TRAINING_PRECISION_FIX.md). Trainable action-expert
and projection weights now remain FP32, with FP32 AdamW moments and BF16 forward
autocast. The old mixed-storage runs below demonstrate historical runtime/I/O
behavior, not an effective-learning or current memory qualification.

The corrected one-parent batch-eight diagnostic has reached 4.14 GB peak PyTorch
allocated memory on this 8 GB GPU; this is not total device usage or a completed
long-run qualification. Do not reuse the old batch-16 recommendation without a
new benchmark and device-headroom check. The FP32 batch-eight GPU optimizer/RNG
exact-resume qualification passed on 2026-10-20 (12 vs 8+4 updates). Legacy weights
can warm-start a new optimizer; legacy BF16 optimizer continuation is rejected.
An expert-only FP32 inference overlay is about 399.5 MB of tensor data; a full
checkpoint additionally needs the frozen backbone and optimizer/RNG state.

The Windows RTX 4060 Ti (8 GB) can train the existing SmolVLA action expert and
state projection with a frozen VLM. Qualify each new dataset release and batch
size first; the earlier batch-one peak of about 2.14 GB is PyTorch allocation,
not total GPU usage. No policy success is inferred from loss alone.

## Current campaign (2026-10-12)

All three public Drive archives were downloaded with gdown, about 14.63 GB total.
Their completed manifests contain **101 episodes / 184,603 commands**, not the
99 episodes mistakenly reported during initial intake. The explicit merged view
passed strict validation: 81 training episodes / 147,499 commands and 20 held-out
episodes / 37,104 commands. Source and original split manifests are preserved.

The first archive's 34-episode export passed 102 real video/action/terminal-mask
boundary and midpoint probes. A 100-update batch-four BF16 qualification on this
subset completed with finite losses/gradients and 2.53 GB peak allocated VRAM.
Selected held-out fixed-noise flow loss improved 0.488800 -> 0.058982; selected
decoded gripper MAE improved 10.81 -> 3.66 mm. These offline probes do not establish
autonomous grasp/insertion or generalization to the reserved evaluation bank.

The full 101-episode export completed at 17:51 Bangkok time and passed actual
video/action readback. v2's batch-16 preflight completed 199 updates within its
600-second cap (including final evaluation/save), with selected held-out loss
0.453493 -> 0.020747 and 4.91 GB peak allocated VRAM. The v2 campaign then stopped
at the base-model inference gate; it did not start the main run. All artifacts
are preserved, including its failed logs.
The initial worker's provisional local repo-id prefix contains `_99`; it is an
identifier only, not the count. Dataset receipts/metadata record the actual 101
episodes; future launches use the neutral `_release` prefix.

The coordinator was handed over from v1 to v2 without stopping or repeating the
original exporter. v2 reuses the passed merge/audit gates and waits for its completed
export receipt. Before the full-release training preflight, it benchmarks batches
4, 8 and 16 on the completed first-archive export in separate fresh processes:
40 real optimizer updates each, with the first ten excluded from throughput timing.
The fastest safe candidate is used for both the 400-update full-release preflight
and main training, with the learning rate unchanged. Safety requires finite losses
and gradients, completed updates, and at least 1 GiB of conservative GPU headroom.
The estimate uses device-memory snapshots and peak PyTorch reserved memory plus
pre-launch desktop/device usage, not PyTorch allocation alone. This is a bounded
candidate comparison, not proof of the maximum possible batch or long-run stability.
Selection and measurements: `outputs/v2c30_campaign_v2/batch_benchmark/report.json`.
All benchmark process time counts against the four-hour qualification/main budget;
failed candidates retain their logs and do not silently restart.

Completed local comparison selected **batch 16**: batch 4 = 8.19 samples/s,
batch 8 = 10.76 samples/s, batch 16 = 11.18 samples/s. All three completed
40 finite-gradient updates. Batch 16 peaked at 4.91 GB allocated / 4.98 GB
reserved; device peak was 6,593 MiB, leaving 1,595 MiB conservative headroom.
It was about 36% faster in samples/s than batch 4, but only 4% faster than batch 8
in this short comparison while CPU export was running. Batch 16 is the fastest
tested safe choice, not a proven absolute maximum. Benchmark time was 214.67 s
and is deducted from the training allowance. The full-release preflight must
still pass before the longer batch-16 run starts.

The combined qualification/main-run GPU training budget is at most four hours;
the main supervisor is capped at 3 h 50 min or less after qualification overhead.
Download, CPU dataset preparation and post-training simulation inference are
outside this optimizer-training budget. Preparation may take tens of minutes.
Hour checks are local reports, not scheduled AI/chat replies.

Current progress is in `outputs/v2c30_campaign_v4/status.json`; a failure or final
completion creates `completion.json`. The main job's hourly reports appear under
`outputs/v2c30_sft_overnight_v1_monitor` after its stability gates pass. Previous
v1/v2/v3 outputs remain unchanged for diagnosis and provenance.

### Windows I/O stability and authorized overnight run

The first long run stopped at 18:45:36 Bangkok after 75 finite-gradient updates.
Windows refused replacement of `status.json` while a reader held it open;
the failure was in telemetry, not a reported OOM/nonfinite model error. The old
main run had no checkpoint yet; the earlier 199-update qualification remains.

`learning/vla/runtime_io.py` now uses native Windows delete-sharing readers,
unique temporary files and bounded atomic-replace retries. Status snapshots are
best-effort on a sharing/permission failure, retaining the full event in JSONL
and stdout; critical checkpoint markers still fail if their durable write cannot
complete. Actual Windows tests cover a held non-delete-sharing reader and 200
concurrent atomic updates without partial JSON. The supervisor uses both event
and status timestamps for heartbeat health, stops its own process tree after a
15-minute stalled heartbeat, and records terminal `running=false`. Unexpected
monitor exceptions also terminate the owned trainer rather than leaving an orphan.

The user explicitly extended the cumulative allowance to **eight hours total**.
This is opt-in (`--authorized-total-seconds 28800`), not a new default: ordinary
trainer/supervisor calls still enforce four hours. v4 debits 1,400.40 seconds of
earlier qualification, inference checks and failed training, plus all new GPU
qualification time, before assigning the main supervisor allowance. Preparation
and post-training simulation remain outside optimizer-training time.

`scripts/run_stable_smolvla_campaign.py` reuses the completed release and unchanged
model input hashes, then qualifies the new trainer in fresh processes. Exact
12-update baseline vs 8+4 resumed updates passed bit-for-bit model comparison and
exact optimizer/scheduler/RNG/sampler/config comparison. Next, live supervised
batch-16 training runs longer than the previous failure with frequent shared
status reads and a deliberately held legacy reader. A failed test prevents the
unattended phase from starting.

Live qualification passed with **131 updates**, **4,721 shared-reader checks**,
zero read errors, and one expected nonfatal warning from the deliberately held
reader. Selected held-out probe loss improved **0.458730 -> 0.081399**. The eight-
minute training qualification also exercised early saves and safe checkpoint
retention; two superseded same-run recovery checkpoints were pruned, while the
initial, best and recent/final checkpoints remained available.

The main session **resumed step 131 at 20:13:17 Bangkok on 2026-10-12**, with batch
16, BF16, target **12,048 cumulative updates**, and a **26,602-second** remaining
supervisor allowance (7 h 23 min 22 s). Its hard deadline is approximately
**03:36:39 Bangkok on 2026-10-13**; final evaluation/save are reserved before that
deadline. Prior time was 1,400.40 s, new qualification time 736.75 s, and about
60 s remains as an outer margin within the 28,800-second authorized total.
The trainer restores its own 520.469-second cumulative checkpoint clock; the
supervisor covers just this new process session, avoiding a double time debit.
At the handoff check, optimizer updates continued beyond step 190 with finite
loss/gradient norms and about 4.91 GB peak allocated VRAM; the monitor was healthy.

New training output is `outputs/v2c30_sft_overnight_v1`. Save at update 25, then
approximately every 15 minutes in the main session, with hourly health checks and
1,000-update evaluation. The main session resumes its own qualified checkpoint
with matching configuration/optimizer/scheduler/RNG, so qualification updates are
not discarded. Earlier short-run checkpoints with a different trainer hash are
never silently resumed. On resume, the previous completed session report is
archived under the new monitor directory, and the current report is marked
incomplete until training finishes; an old qualification report cannot masquerade
as a completed main run.

Checkpoint retention is explicitly enabled for this new output only: preserve
initial, best, final and the two most recent completed recovery checkpoints.
Superseded same-binding recovery directories are deleted to bound disk use;
their removal is logged in `events.jsonl` and is not recoverable. Other runs,
incomplete directories, mismatched bindings and symlinks/junctions are never
cleanup targets. Low-disk, finite-gradient and independent deadline guards remain.
The temporary process-scoped sleep guard covers this eight-hour authorized run;
global power settings are unchanged. Manual sleep, reboot or power loss still
interrupts training and requires explicit restart from a complete checkpoint.

Evidence: `outputs/v2c30_campaign_v4/exact_resume.json`,
`outputs/v2c30_campaign_v4/io_stress.json`,
`outputs/v2c30_campaign_v4/preflight_gate.json`, and the new monitor's `launch.json`.
Several-minute qualification is evidence of the tested fixes, not a bug-free or
eight-hour stability guarantee. Loss improvement alone is not grasp/insertion success.

### Explicit inference-gate recovery and main run

The original base checkpoint describes a different robot: state/action dimensions
6/6 and three cameras, whereas this OpenArm task is 16/8 with two cameras. SFT
correctly adapts these features and processors; the original baseline rollout
incorrectly loaded the untouched 6-action configuration and failed the strict
physical-action boundary. The action validator was not relaxed or bypassed.

An unmodified base comparison now requires explicit `--base-export` in
`scripts/rollout_smolvla.py`. It uses the same upstream feature adapter and
train-only statistics as SFT, without changing weights. Adapted checkpoints
continue to load their saved processors and reject this override. Both paths
validate the OpenArm feature contract and require a finite decoded 1x50x8 chunk.
Normalization provenance is included in the rollout report.

With the user's explicit restart authorization, `scripts/resume_smolvla_campaign.py`
created a new v3 control directory, verified original qualification hashes and
the complete checkpoint marker, reused data/readback/benchmark/training/exact replay,
and reran base/preflight-policy inference in fresh processes. Both executed 100
commands without robot safety abort, but neither established task success in that
two-second smoke budget. No scene/controller/collision guard was changed.

The first main batch-16 BF16 attempt launched at 18:41:36 Bangkok time. Supervisor allowance
was **13,179 seconds (3 h 39 min 39 s)**; its planned hard deadline was 22:21:15.
That attempt stopped at 18:45:36 as described above; this old deadline is inactive.
The trainer's lower soft allowance reserves final evaluation/checkpoint time.
The ledger debits earlier subset training, benchmarks, the full original short-run
cap, failed/retried inference checks and a conservative diagnostic allowance.
The restart is not a fresh four-hour budget. Model loading, evaluations and saves
are inside the main supervisor limit; post-training simulation is outside training.

The cosine horizon is now **5,709 updates**, sized from the full-release preflight's
2.203-second median update and remaining allowance, instead of the previous
unreachable 60,000-update target. LR remains 1e-4 with warmup/cosine decay; the main
run starts from the same base weights, not a mismatched short-run optimizer state.
Hourly health/checkpoints and 1,000-update loss probes remain enabled. The watchdog
kills its own process descendants too, including Windows venv launcher children.
Gate/budget evidence: `outputs/v2c30_campaign_v3/preflight_gate.json`;
launch evidence: `outputs/v2c30_sft_4h_v1_monitor/launch.json`.

## Dataset intake

Install `requirements-download.txt` in a separate `.venv-download` environment.
Use public Drive access with `--no-cookies`, preserving archives in ignored
`data/` storage. [gdown](https://github.com/wkentaro/gdown) supports folder listing
and partial-download resume. Do not execute files from an archive.

`scripts/extract_collection_rar.py` rejects unsafe paths and links, checks expanded
size against a 25 GiB reserve and extracts only manifests, lossless NPZ episodes
and diagnostics. Redundant JPEG previews and failed preflight recordings remain
in the original archives. A completed extraction gets a receipt.

`scripts/merge_collection_release.py` makes an immutable hard-linked view, checks
the existing strict data contract and rejects evaluation-seed episodes. Original
manifests stay untouched. When sources have no validation episodes, a deterministic
hash-ranked 20% episode holdout is explicit in the new manifest. This is a validation
set used for model selection, not the untouched evaluation bank. Do not modify a
hard-linked NPZ/metadata file; doing so also changes its original view.

Export with `scripts/prepare_smolvla.py`, then run
`scripts/verify_smolvla_export.py`. Inspect camera/stage examples with
`scripts/inspect_collection_training.py`. Reject mismatched scene/controller
releases, incomplete exports, cross-split duplicate contents, timing errors and
unsafe action mappings before a long run.

## Full-window trainer

`scripts/finetune_smolvla.py` uses LeRobot 0.4.4's actual policy, training-only
processors, optimizer preset and cosine scheduler. It applies our audited terminal
mask spelling shim. Actor inputs remain two cameras, bimanual state and language;
phase/contact/pose diagnostics are never actor inputs.

Every recorded command is eligible, sampled uniformly with replacement. Terminal
chunks are padded and masked, never borrowed from another episode. This replaces
the earlier four-window-per-episode diagnostic sampler. Batch size is qualified
with actual VRAM/throughput measurements, not inferred from GPU capacity alone.

The bounded run records stochastic loss, finite gradient norms, samples, learning
rate, step latency and peak allocation. Fixed-noise timeline probes measure train
and held-out flow loss. Eight probes per split also report decoded seven-joint MAE
in radians and gripper MAE in millimetres with terminal padding excluded. Probe
errors are diagnostic samples, not full validation or closed-loop task performance.

Checkpoints contain the complete model and processors, optimizer/scheduler, RNG,
dedicated sampler state, input/config hashes and update cursor. `complete.json`
marks a finished checkpoint. Resume is restricted to trusted checkpoints from the
same output root; pickle-backed training state must never come from downloaded
data. Model/dataset/config binding is checked before restoration. An independent
process resume comparison is required before an unattended campaign.

## Four-hour supervisor and local monitoring

Example after dataset and batch qualification (replace `RELEASE`):

```powershell
$job = Start-Process -FilePath .venv-smolvla/Scripts/python.exe -WindowStyle Hidden -PassThru -ArgumentList @(
  '-u', 'scripts/run_smolvla_job.py', '--output', 'outputs/RELEASE_sft',
  '--max-seconds', '14400', '--', '--export', 'data/RELEASE/export',
  '--checkpoint', 'data/models/smolvla_base', '--backbone', 'data/models/smolvlm_backbone',
  '--steps', '60000', '--batch-size', '1', '--eval-every', '1000',
  '--save-every', '60000', '--probes', '64'
)
$job.Id
```

The supervisor's deadline includes imports, evaluation and checkpointing. The
trainer reserves time for final evaluation/save; the supervisor terminates its
own child if the outer deadline is reached. It does not change global power settings.
`scripts/keep_training_awake.py` can request temporary Windows sleep prevention
for a specific campaign process handle, automatically released when it ends or
after at most six hours (preparation plus the bounded training). The current
campaign uses this guard; it does not keep the display on. Manual sleep/reboot
still interrupts work.

Monitoring runs as local Python processes and needs no AI/API tokens. Files:

- `outputs/RELEASE_sft/events.jsonl`: per-update samples/loss/gradient/throughput.
- `outputs/RELEASE_sft/status.json`: latest training heartbeat.
- `outputs/RELEASE_sft_monitor/train.log`: trainer output and errors.
- `outputs/RELEASE_sft_monitor/health.json`: current process/heartbeat health.
- `outputs/RELEASE_sft_monitor/hourly_checks.jsonl`: launch and hourly GPU/status snapshots.
- `outputs/RELEASE_sft_monitor/completion.json`: exit status and deadline outcome.

The trainer also evaluates and checkpoints at each elapsed hour. With
`--save-every=60000`, hourly and final saves bound storage without deleting old
artifacts. Keep at least 8 GiB free while training; model checkpoints require more
space than loss logs. A `STOP` file in the run directory requests a clean stop.
Chat itself does not automatically reopen every hour: hourly checks are performed
by the local supervisor/trainer and can be inspected whenever requested.

After training, compare base and selected checkpoint in the source-matched
environment with identical resets, collision limits and action prefixes. Record
grasp, carry and insertion outcomes, latency and clipping. Only these closed-loop
results establish progress toward the actual task. RLT remains a later stage.
