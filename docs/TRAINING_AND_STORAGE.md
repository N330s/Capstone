# University training: interfaces, compute and storage

Planning update: 2026-09-27. These are proposed budgets and acceptance criteria,
not completed training or measured university-machine performance.

## Scope and integration contract

Train SmolVLA for table -> grasp -> transport -> insert on OpenArm v1, right arm
active and left parked. Then adapt PI's RL Token method to improve insertion.
Collection and scene development belong to teammates; the training workstream owns
validation/export, training configuration, deployment/evaluation and later RLT.

Before bulk collection, agree and version the following:

- Two RGB streams (fixed and wrist), camera names, resolution, calibration,
  timestamps and encoding. Current simulation uses 320x240; the larger 640x480
  storage example below is a budget scenario, not a requested scene change.
- Sixteen state values: left seven joints and mean finger travel, then right
  seven joints and mean finger travel. Eight actions: right seven absolute joint
  targets in radians and symmetric per-finger travel in metres, not jaw aperture.
- Preserve the 50 Hz command timeline. Capture observations before the associated
  command, retain the terminal observation, and distinguish requested commands
  from clipped/applied commands. Record actual execution durations and overrides.
  Do not silently lower control frequency to save disk. Any lower camera rate
  requires explicit timestamp alignment and validation in training and deployment.
- Episode/task/phase/source IDs, outcome, termination versus truncation, reset seed,
  scene/controller versions and hashes. Store privileged poses/contacts separately;
  they may support simulator reward and diagnostics but are not actor inputs.
- Deterministic reset/step and phase-aware full-task execution without resetting
  between pickup and insertion. Scene owner supplies success/safety semantics;
  training owner tests inference commands through the same controller.
- Versioned LeRobot export; split by complete episode and scene configuration.
  Fit statistics on training only. Keep an untouched 100-reset final benchmark
  separate from the validation resets used for checkpoint/hyperparameter selection.

First delivery: 5-10 complete synchronized episodes plus a manifest and replay
report. Validate decoding, alignment, units, limits, chunk end masks and image
visibility before requesting roughly 50 demonstrations for a first baseline.
Scale toward 100-300 based on observed failures, not an assumed sufficient count.
Historical held-plug pilots can exercise adapters but do not cover table pickup.

## Compute and training sequence

The RTX 4090 has 24 GB VRAM. It is a reasonable starting target for a small-model
fine-tuning pipeline, but this is not a measured memory-fit or throughput promise.
Confirm actual available VRAM, university OS/driver, job limits, install rights,
RAM and persistent quota. Prefer Linux and plan for 32 GB RAM initially, ideally
64 GB for decoding and simulation workers; tune worker counts to measured usage.
Source: [NVIDIA specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/).

1. Pin LeRobot and the SmolVLA checkpoint in a separate environment from the old
   pi0 integration. Reuse the upstream trainer and model-specific processors.
2. Record which parameters are trainable. Start with the frozen-backbone,
   action-expert recipe after checking the pinned implementation. Test supported
   mixed precision, microbatch 1-4, then accumulation and larger batches as memory
   allows. Do not copy the documentation's A100 throughput estimate to the 4090.
3. Profile one batch, then overfit 1-2 training episodes; verify save/reload and
   interrupted-job resume. Log peak VRAM/RAM, steps/sec, checkpoint bytes and decode
   throughput. No long job until decoded commands pass the controller contract.
4. Evaluate scripted expert, pretrained policy and fine-tuned policy on matched
   development resets. Measure full-task and per-phase success, contacts and
   latency. Choose execution prefix from latency/contact tests, not chunk length
   alone. Run training and evaluation sequentially at first.
5. Freeze a successful supervised bundle. Audit SmolVLA embedding access against
   the pinned RLT paper, train the token bottleneck on training episodes, validate
   separately, then freeze both models. This is a SmolVLA adaptation, not an exact
   reproduction of the paper's model.
6. Implement and test insertion-only RLT: chunk actor/critic, reference anchoring,
   dropout, sparse success, duration-aware replay/discounts and safety aborts.
   Include supervised-policy handoff states, not only ideal expert starts. Start
   with one environment; GPU speed alone does not determine simulation throughput.
7. Compare against frozen SFT on held-out resets, then test complete table-to-insert
   trajectories. Use multiple RL seeds when feasible and disclose interaction and
   wall-clock budgets. Real transfer remains a later calibration/safety milestone.

SmolVLA is a 450M-parameter model with an upstream LeRobot fine-tuning workflow.
The routine part is running that trainer; trustworthy data mapping and deployment
are the main integration work. RLT introduces a learned representation and online
actor/critic adaptation, so it is a separate implementation/validation project.
Sources: [SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla),
[RLT paper](https://arxiv.org/abs/2604.23073).

## Recommended disk allocation

Request **500 GB of free, persistent SSD working space**, excluding OS and other
users' files. **1 TB free** is preferable for multiple dataset/scene versions and
longer experiments. **150-200 GB** is a lean initial fine-tuning allocation only.
All sizes below use decimal GB/TB and assume compressed video, bounded caches and
retention. They exclude a separate durable backup and indefinite raw archives.

| Category | Working budget |
| --- | ---: |
| Environments, packages, source/assets and installation caches | 40 GB |
| Pretrained weights and model/backbone download caches | 20 GB |
| Compressed demonstrations, metadata and staged conversion | 120 GB |
| Retained supervised checkpoints including resumable state | 60 GB |
| RLT feature cache, bounded replay and RL checkpoints | 100 GB |
| Selected evaluation videos, logs and reports | 20 GB |
| Temporary files, conversion overlap and free headroom | 140 GB |
| Total allocation | **500 GB** |

These are allocation caps, not predictions that every category uses that amount.
Rebudget after a representative ten-episode export and the first saved checkpoint.
If multiple full archives or high-bitrate camera sets exceed this budget, request
more storage; do not silently discard source data or lower image quality.

### Why camera encoding dominates

Assume 1,000 episodes, 40 seconds each, two cameras and 50 frames/second:

- Raw uint8 RGB at 640x480: `1000 * 40 * 50 * 2 * 640 * 480 * 3`
  = **3.6864 TB**, before any duplicate export or metadata.
- At the current 320x240 resolution the same raw workload is **921.6 GB**.
- Encoded video averaging an illustrative **4-8 Mbit/s per camera**:
  `episodes * seconds * cameras * bitrate / 8` = **40-80 GB**.
  These bitrates are assumptions, not measurements or required codec settings.
  Two complete encoded copies require 80-160 GB; stage conversion in batches if
  the dataset allocation cannot hold both. Three cameras multiply sizes by 1.5.

At 100 episodes, divide these numbers by ten; at 300, multiply by 0.3. Frame rate
affects raw storage directly; encoded size depends on the measured bitrate.
Compressed NPZ RGB arrays are not equivalent to inter-frame video compression.
Inspect decoded plug edges, socket slots and contact motion before selecting an
encoding setting; precision visibility matters more than a nominal small file.
Retain a small lossless comparison sample and numerical replay traces.

### Checkpoints and RL storage

At 450M parameters, weights alone are approximately 0.9 GB in 16-bit or 1.8 GB in
32-bit format. These are arithmetic estimates, not measured download sizes.
Resumable checkpoints also contain optimizer/scheduler/RNG state and possibly
master weights; budget roughly **2-6 GB each**, measuring the actual configuration.
Retain best, latest and one milestone per run initially, and limit active runs.
Never remove the only resumable checkpoint or an unbacked-up selected result.

Do not put duplicated raw images into every replay entry. One million stored
transitions with two 640x480 RGB images at both boundaries would take **3.6864 TB**
before overhead. This counts stored transitions; chunk transitions and 50 Hz
primitive control steps are different quantities and must be reported separately.

After the VLA/token bundle is frozen, use bounded replay with tokens, states,
reference/executed actions, durations, rewards, terminal masks and frame IDs.
For illustration, 256-float tokens at both boundaries, two 16-float states, two
50x8 reference chunks and one executed 50x8 chunk consume about **7 GB per million
entries** in float32 before masks/indexing/other metadata. Token dimension and
chunk size here are budgeting examples, not selected RLT hyperparameters.
Keep selected compressed rollout videos and explicit provenance. Changing the
frozen checkpoint, token, processors or scene contract invalidates incompatible
features/replay; rebuild from retained inputs or recollect rather than mixing them.

Token training can need much larger VLA embeddings: even 100,000 samples with
256 tokens x 1,024 features x 2 bytes use **52.4 GB**. Prefer on-demand extraction
or a bounded feature cache; do not materialize every layer at every video frame.

### University-machine retention and portability

- Put model caches, datasets, run outputs and temporary conversion files under
  explicit project storage roots, not an unknown small home-directory quota.
- Use one shared immutable pretrained cache. Avoid hidden dataset copies per run.
  Keep heavy artifacts out of Git; version manifests, hashes, configs and code.
- Preserve exact dataset splits, processors, normalization, scene/controller
  hashes, seeds and environment lock with every selected checkpoint.
- Save resumable state at a frequency compatible with university job interruptions.
  Copy selected checkpoints/manifests/reports to approved durable storage; a local
  SSD quota is not a backup. Confirm access and data retention after the semester.
- Alert at 80% of allocation and stop new collection before exhausting space.
  Retention must be explicit and scoped; no automatic deletion of unique data.
- For sim-to-real, keep hardware units, camera calibration, action limits and
  timing in the same versioned interface. Improved visuals alone do not validate
  contact forces, grasp friction or real controller behavior.
