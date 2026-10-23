# Corrected multi-episode pickup training

## Current bounded campaign (2026-10-20)

The user authorized training toward pickup generalization. The offline supervisor
was launched at **00:34 Bangkok time** under
`outputs/fp32_multiepisode_20261020`. Exact GPU checkpoint resume **passed**
after 183 seconds; 12 uninterrupted updates and 8+4 resumed updates match
exactly. Batch-eight peak reservation is 4.186 GB, with 2,636 MiB conservative
device headroom. The main multi-parent SFT stage has started. All 64 selected
regression tests pass; full-checkpoint evaluator preparation and plot generation
also passed smoke checks. These checks are not task-success results. The four-hour
wall-clock budget includes qualification, imports, checkpoint I/O and SFT.
Automatic paired pickup evaluation has another 45 minutes, followed by at most
five minutes for audits/plots. Expected latest completion is about 05:24, not an
instruction to keep training until then. Early safety stops remain enabled.

This run is not a promise of generalization. The previous frozen one-parent
diagnostic passed only 1/5 validation resets. A new result must be measured.

### Training contract

- Existing nominal export: 81 training parents / 147,499 commands; 20 validation
  parents / 37,104 commands. No new split and no validation samples in optimizer
  updates. Uniform command sampling with replacement; masked terminal chunks.
- Warm-start the original adapted best weights at
  `outputs/v2c30_sft_overnight_v1/checkpoints/step_0009426_best/pretrained_model`.
  Use a **new optimizer**, not the legacy BF16 optimizer or one-parent overlay.
- Frozen VLM; trainable action expert/state projections and Adam moments FP32;
  BF16 forward autocast. Batch 8, peak learning rate 1e-5, fixed seed 1000,
  upstream warm-up/decay with 20,000-update ceiling and the wall-clock limit.
- Inputs remain two RGB views + 16 joint/finger positions; output is a 50-step
  chunk of eight absolute right-arm/finger targets. Torque/depth remain unused.
- Fixed timeline flow-loss probes and eight decoded-action anchors per split
  run initially, every 500 updates, on timed recovery/hourly boundaries, and at
  the end. These are **sampled probes, not full validation**. Live simulation
  is the task-success measurement.

### Qualification and safety

Independent GPU processes compare 12 uninterrupted updates against 8+4 resumed
updates of the actual main trainer. Model tensors, FP32 Adam moments, scheduler,
Python/NumPy/Torch/CUDA RNG, sampler, loss, gradient norm, LR and sampled commands
must match exactly. Conservative desktop-plus-reserved GPU headroom must be at
least 1 GiB. This qualifies batch 8, not an absolute maximum batch size.

Local monitoring records health roughly every 15 seconds and a durable hourly
snapshot: finite update telemetry, GPU use, temperature, RAM, disk, heartbeat
and child PID. The supervisor stops on thermal/RAM/disk/heartbeat failure and
does not silently restart. A scoped Windows keep-awake request lasts only while
this campaign is alive; global power settings and display settings are unchanged.
There are no AI/API calls, uploads or hardware robot commands.

The trainer estimates full model + two FP32 Adam moments plus serialization
slack. Writes must retain a **25 GiB disk reserve**; updates stop early with room
for a final checkpoint. Keep this new run's best, latest/recent and final
checkpoints; its initial recovery remains until a complete same-run successor
at 500+ updates exists. Once exact resume passes, discard the newly created
disposable qualification checkpoint bundles, preserving hashes, small metadata
and all logs. Previous experiments are outside this automatic retention scope.

The initial launch plan preserved the full reference and early recovery bundles.
Observed free space subsequently fell about 3.5 GiB beyond checkpoint sizes
(cause not established). To avoid prematurely shortening training, the explicit
`retention_amendment.json` changes **only this new run's bootstrap retention**.
The new qualification reference (2.04 GiB) was permanently retired after the
25-update full training checkpoint completed; the latter retires only after a
new complete checkpoint at 500+ updates, never while current best/latest.
`scripts/retire_fp32_bootstrap.py` implements these exact two targets, bounded by
the existing campaign PID/deadline. Its hashes/metadata/receipts are retained in
`bootstrap_retention`; failures are logged and cannot authorize other cleanup.
Free disk recovered to about 33.27 GiB with the first training checkpoint present.

With explicit user permission, nine old short-run benchmark/preflight bundles
were permanently removed to free **13.4997 GiB**, taking free space to about
38.83 GiB before launch. Exact targets and preserved metadata are recorded in
`outputs/approved_preflight_cleanup_20261020.json`. All old logs, datasets,
original best weights, repaired FP32 overlay and pickup evaluation artifacts
remain. No additional old-output deletion is authorized.

### Frozen task evaluation

Use the **final full checkpoint**, selected before training, not a sweep over
checkpoints or execution prefixes. Evaluate the same five validation resets
(`train_20004`, `train_21009`, `train_21022`, `train_22000`, `train_22042`) and
the separate training-reset control `train_20002`, with exact expert replay
controls. Prefix 25, policy seed 1000, at most 1,200 commands per condition.

Keep the source-matched immutable collector scene, explicit physical-contact
settle bookkeeping adapter, safety guards and existing pickup metric: grasp +
both finger contacts + actual 20 mm plug lift + one continuous second of carry,
with no prior drop or safety abort. No recorded-input assistance, forced states
during policy execution, command correction or privileged actor features.

All full-model conditions are attempted even if the training-reset policy
control fails; report that control failure explicitly. Never remove a failed
validation reset from the denominator. Expert failures invalidate the screen.
An artifact timeout preserves partial evidence and marks evaluation incomplete;
it does not replace or silently rerun the physical outcome.

Passing 5/5 is only a small same-scene **validation** screen. This split
influenced the earlier checkpoint selection; a full validation bank and fresh,
disjoint tabletop test bank are required before broad generalization claims.
Insertion, RL and sim-to-real deployment are not automatically authorized by a
passing pickup screen.

### Where to inspect

- `outputs/fp32_multiepisode_20261020/health.json`: current stage/health.
- `.../qualification/exact_resume.json`: GPU precision/resume gate result.
- `.../training/events.jsonl`: every sampled command, loss, gradient norm, LR,
  update duration and GPU allocation; `training/report.json` is terminal summary.
- `.../training/checkpoints`: resumable bundles, `complete.json` written last.
- `.../pickup_evaluation`: policy/expert native videos, traces, manifests,
  diagnostics and physical-outcome reports.
- `results/fp32_multiepisode_20261020`: after successful execution/audit,
  pickup plot/CSV/final frames, training/validation plot and training summary.
- `.../completion.json`: terminal success or failure. A completed campaign can
  still have a failed generalization screen; read the physical success count.

For a graceful early stop, create `STOP` inside the new campaign's `training`
directory. Do not kill unrelated Python processes. Disconnecting the chat does
not stop the detached local supervisor; shutdown/reboot will stop it.

Entry point: `scripts/run_fp32_multiepisode.py`. Never launch a second instance
against the same output root, and do not edit training/evaluation implementations
while the run is active. Sources and input hashes are archived at launch.
