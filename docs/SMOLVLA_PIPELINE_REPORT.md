# OpenArm / SmolVLA: pipeline status and evidence

Updated 6 October 2026. Measured training and rollout results were generated on
4 October; no new training is implied.

## Claim supported by the evidence

We have demonstrated the engineering loop:

**Automated simulation demonstrations -> validated LeRobot dataset -> real
SmolVLA fine-tuning -> checkpoint/processor restoration -> policy-driven simulation.**

Short-run training is numerically healthy and reproducible on the tested runtime.
We have not demonstrated autonomous table-to-insertion success, guaranteed
bug-free code, long-run stability, online RL or sim-to-real transfer.

## Dataset and training visualizations

- [Six-page PDF](../results/dataset_visualization_v3/dataset_training_report.pdf)
- [Dataset contents and split](../results/dataset_visualization_v3/01_dataset_overview.png)
- [Actual two-camera storyboard](../results/dataset_visualization_v3/02_camera_storyboard.png)
- [Full episode phases vs. sampled windows](../results/dataset_visualization_v3/03_phase_coverage.png)
- [One real state/action training sample](../results/dataset_visualization_v3/04_training_sample.png)
- [Implemented training workflow](../results/dataset_visualization_v3/05_training_pipeline.png)
- [Measured training evidence](../results/dataset_visualization_v3/06_training_evidence.png)
- [Visualization guide](../results/dataset_visualization_v3/README.md)
- [Visualization provenance](../results/dataset_visualization_v3/evidence.json)

The slides use real recorded expert pixels and saved metrics. Expert images are
not learned-policy successes. PNGs render directly on GitHub; the HTML gallery
requires downloading the folder and opening `index.html` locally.

## Environment and collection

The matching collector environment comes from the teammate's `automateData`
branch at [`b7297ba82701fe7e7814c32ea617a85f832b74a4`](https://github.com/N330s/Capstone/tree/b7297ba82701fe7e7814c32ea617a85f832b74a4).
All ten recorded environment source hashes, workspace and robot XML hashes match;
the runtime scene/control contract and held-out initial state also match. This
is the Type O / universal_th socket scene with a flexible cable, not the older
two-blade baseline. The active arm is right; the left OpenArm v1 arm stays parked.
The matching source is not necessarily the branch's latest revision.

The automated scripted expert records table start, descent, close, lift,
transport, alignment and insertion at 50 Hz. Each recording includes two
320x240 RGB streams, 16 joint/gripper state values, eight right-arm joint/gripper
targets, requested/applied commands, timestamps, outcomes and provenance. Exact
simulator poses/contact diagnostics are not actor inputs.

Four available successful expert episodes contain 7,251 transitions:
5,501 training transitions in three episodes and 1,750 validation transitions in
one complete held-out episode. Normalization uses training data only. The local
manifest's missing `train_00005` is explicitly excluded by the smoke selection.

## Measured checks

| Stage | Evidence | Result and boundary |
| --- | --- | --- |
| Real LeRobot export/readback | [Readback JSON](../results/smolvla_export/readback.json) | Twelve boundary/midpoint checks across four episodes pass: numeric labels, image decoding and terminal padding. Not an exhaustive image-quality audit. |
| Real GPU training | [Qualification JSON](../results/smolvla_training/qualification_v2.json) | 24 stochastic updates, 12 training windows, four held-out evaluation windows; all reported losses and gradient norms finite. |
| Fixed-noise training loss | Same qualification report | Mean 0.323525 -> 0.245566 (24.1% reduction). |
| Fixed-noise validation loss | Same qualification report | Mean 0.381867 -> 0.292170 (23.5% reduction), on one held-out episode. |
| Model/processors save/reload | [Smoke JSON](../results/smolvla_training/smoke_v3.json) | Exact weight/processor reload, finite 1x50x8 physical-unit inference; trainable weights updated and frozen gradients absent. |
| Saved-state resume | Qualification JSON | Three updates repeated exactly, including sample position, optimizer, scheduler, weights and RNG. Same process/runtime, not a separate-process university resume test. |
| Policy -> simulator | [Rollout JSON](../results/smolvla_rollout/heldout_v2/report.json) | 12 commands before finger/socket-frame collision; guard stops before grasp. Interface execution works, task success does not. |
| Expert control replay | [Replay JSON](../results/smolvla_rollout/expert_replay_v1/report.json) | 100 commands reproduce qpos/qvel exactly with zero unwanted robot contact. Two-second check, not new full-episode success. |

The actual setup is LeRobot 0.4.4, CUDA PyTorch 2.10.0+cu128 and MuJoCo 3.11.0.
Model revisions are pinned in [smolvla_model_sources.json](../configs/smolvla_model_sources.json).
Earlier integration bugs in batching and padding-mask naming were corrected.
The tested repository suite has 45 passing tests; this is not a proof of no bugs.

Warm model-plus-postprocessing inference was approximately 0.304 s, slower than
the 0.10 s execution prefix. Simulation pauses during inference; real-time serving
and hardware control are not qualified. See [rollout details](SIM_POLICY_ROLLOUT.md).

## Budget: 100 episodes on the university RTX 4090

Assumptions: approximately 36 s/episode, two 320x240 camera streams, 50 Hz,
the current compression recipe, frozen-backbone SmolVLA tuning and bounded
checkpoint retention. Larger resolution, longer episodes or more retained runs
change the estimate.

| Storage category | Planning allocation |
| --- | ---: |
| Source recordings, including previews | 20-25 GB |
| LeRobot compressed export | 1-3 GB |
| Dependencies, pretrained models and caches | 10-20 GB |
| Several resumable checkpoints | 6-12 GB |
| Conversion headroom, logs and evaluation videos | 20-30 GB |

Expected working footprint: roughly 60-90 GB. Reserve **150-200 GB free persistent
SSD space** for comfortable initial SFT experiments. The earlier 500 GB budget
is for a larger SFT + RL campaign, not a requirement for this one 100-episode run.
Backup is separate. Existing measurements: four-episode source bundle 0.87 GB;
export 38 MB; final weights plus resumable state 1.61 GB.

For 20,000 optimizer updates, provisionally budget **5-14 GPU hours** on one 4090.
This assumes 0.7-2 s/update at batch approximately 4-8/BF16 and 25% overhead for
validation/checkpointing; it is a planning assumption, **not measured 4090
throughput**. Collection, conversion and closed-loop evaluation are additional.
The [official SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla) gives a
different A100 reference estimate; it cannot be directly substituted for our host.

Our measured local diagnostic on an RTX 4060 Ti used batch one, peak
PyTorch-allocated memory 2.14 GB and an 80.4 s qualification interval. The interval
includes evaluation/reload checks and is not a long-run steps-per-second benchmark.
It should not be scaled directly into a 4090 speed prediction.

Before a long university job, warm up for 200 updates and time 1,000 additional
updates. Then estimate `hours = remaining_updates * seconds_per_update / 3600`,
adding measured validation/checkpoint overhead. A 100-episode dataset does not
determine time alone: updates and effective batch size determine sample exposure.

## Next steps

1. Obtain approximately 100 varied full-task demonstrations; split complete
   episodes/configurations and reserve untouched evaluation resets.
2. Qualify full-window training, longer-run numerical behavior and separate-process
   resume on the university host; keep checkpoint retention bounded.
3. Improve and evaluate supervised full-task success, contacts and inference timing.
4. Only then audit the SmolVLA feature/token interface and implement insertion-only
   PI RLT adaptation. Calibrated sim-to-real transfer remains a later milestone.

## What is intentionally not in Git

Raw demonstration archives, downloaded models, trained checkpoints, virtual
environments and separate third-party checkouts stay local. Reports, code,
configuration, hashes, sample visualizations and short diagnostic trajectories
are included. Re-running training requires staging the data/model/asset bundles;
the visualization report itself can be opened without them.

For visualization regeneration, install `requirements-visualization.txt`, stage
the source recordings and run `scripts/visualize_smolvla_dataset.py` with a fresh
output directory. For training setup and boundaries, see [SMOLVLA_TRAINING.md](SMOLVLA_TRAINING.md).
