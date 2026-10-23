# Five-episode scaling of the successful open-hold recipe

## Completed outcome

Training and evaluation completed normally: 4,000 finite updates, exactly 800
per parent, approximately 85 minutes training and 114 minutes total. All five
expert replays and the saved single-parent control passed; fresh reload predicts
identically with unchanged normalization. The new live policy picks up on
`train_20002`, `train_20042` and `train_22001`; `train_20003` / `train_22045`
abort on finger self-contact. Thus **3/5 training-reset pickups**, and held-out
validation **not run** under the predeclared 5/5 progression gate. Perturbed
training-probe finger MAE improves 1.855 -> 0.675 mm; these are not validation
or autonomous generalization metrics. No training/evaluation job remains active.
See [the collision diagnosis](FIVE_EPISODE_COLLISION_DIAGNOSIS.md) for the measured
failure chain and next proposed takeover test. Launch-status notes below are
historical; no new correction training has started.

2026-10-20. This implements the user's agreement to test the actual successful
single-parent recipe on a small training cohort before another large uniform
run or new recovery collection. No physical robot execution, scene edits,
collision-guard changes, dataset trimming, old-output deletion or Hub/API use.

## Why this comparison

The successful 800-update single-parent model was **not** trained like the
11,300-update, 81-parent nominal model. Both started from the same original
`step_0009426_best`, but the successful run guaranteed four open-hold examples
per batch of eight and perturbed their measured finger state by at most 2 mm.
The 81-parent model sampled uniformly without that augmentation and was not
initialized from the successful model. It sampled `train_20002` only 1,047
times, compared with 6,400 total anchors from that parent in the successful run.
Thus that earlier comparison did not isolate the effect of episode count.

The separate 1,000-versus-1,000-update initial-wait ablation has completed:
the expert picked up, both policies failed their training-reset pickup gate,
and validation was not run. Excluding waits improved a fixed validation loss
probe but did not establish pickup. This run does not continue either candidate.

## Frozen protocol

Five original **training** parents are selected before new policy outcomes:

| Parent | Commands | Eligible stable open-hold anchors |
| --- | ---: | ---: |
| `train_20002` | 1,673 | 382 |
| `train_20003` | 1,805 | 391 |
| `train_20042` | 1,799 | 373 |
| `train_22001` | 1,791 | 386 |
| `train_22045` | 1,917 | 450 |

Selection is the known training control plus four evenly spaced remaining
lexicographically sorted training parents. All source NPZ hashes, metadata,
seeds, successful outcomes and scene/robot/instruction identities are checked.
Validation parents remain separate. Original global export indices, future
50-command action chunks and terminal masks remain unchanged.

- Initialize from the same original base as the successful single-parent run,
  not its updated weights and not the 81-parent candidate. The saved successful
  FP32 overlay remains an independent live positive control.
- **4,000 updates**, batch eight, round-robin one parent per batch: **800 updates
  and 6,400 sampled anchors per parent**, preserving the original per-parent
  exposure. Across the cohort, 16,000 anchors have the guaranteed open-hold quota
  and up to 16,000 have nonzero bounded sensor perturbations.
- Per batch: four stable open-hold anchors plus four uniform anchors from that
  same parent, with replacement. The uniform half may also contain open-hold
  examples; 50% is a guaranteed quota, not the exact total open-hold fraction.
- Apply independent uniform +/-2 mm noise only to measured right-finger state
  index 15 on the first four training examples, before normalization. Preserve
  the clean RGB, action targets, other state values and original recordings.
  This is sensor-noise augmentation, not physically consistent recovery data.
- LR 1e-5, original upstream AdamW and warmup/cosine recipe, BF16 forward with
  all trainable expert parameters and Adam moments FP32. The normalized LR
  schedule is stretched to 4,000 global updates, rather than reusing an
  800-step absolute schedule. Frozen VLM and saved base normalization unchanged.
- The actor remains 16 measured positions plus two RGB cameras and instruction,
  producing 50 by 8 absolute targets. No phase/time/privileged pose, depth or
  torque input. No inference-time gripper-state adjustment or target filter.
- Select only the final step-4,000 weights, never a validation-loss winner.
  Save one approximately 381 MiB FP32 expert-only overlay; no optimizer-resume
  claim. Its base/model/processor dependencies are hash-bound.

This is a one-seed recipe-scaling pilot, not a factorial isolation of sampling
versus augmentation or proof that one factor caused the old failure.

## Qualification and evaluation sequence

1. Fresh exact expert-command pickup replays on all five selected resets.
   Require the existing physical pickup gate and zero replay qpos/qvel error.
2. Fresh live test of the saved successful single-parent overlay on
   `train_20002`. Stop before training if this positive control does not reproduce.
   Do not repeat attempts until a favorable result appears.
3. Twelve-update GPU preflight on the five-parent sampler; finite loss/gradients,
   FP32 storage and at least 1 GiB GPU headroom. Keep logs/probes but no new
   model copy. Its short LR schedule is not an exact reproduction/resume test
   of the subsequent 4,000-step run. Full training restarts from the frozen base.
4. Train all 4,000 updates. Audit every logged sample for parent order, quota,
   finite values and bounded offsets; require exactly 800 updates per parent.
   Probe nominal/perturbed observations on each parent with three explicit
   fixed latent seeds at initialization, every 800 updates and final. These
   are training probes, not full validation or autonomous success.
5. Fresh-process reload must reproduce saved physical-unit predictions within
   1e-8 and verify exact base-normalization bindings.
6. One live policy trial on each of the five training resets, prefix 25, seed
   1000, at most 1,200 commands. Actual simulation RGB/state only; no recorded
   input assistance, state injection or quantization. Independently recompute
   pickup from saved motion/contact/time traces and decode all video frames.
7. Only after **5/5 training-reset pickups**, evaluate the same predeclared
   five-parent validation bank with fresh expert controls, if at least 47
   minutes remain in the campaign allowance. Otherwise report validation as
   **not run**, with the reason. No training/validation reset substitution.

Pickup retains two-finger contact, physical grasp, at least 20 mm actual plug
lift and one second continuous carry, with no prior drops or unsafe aborts.
The pinned physical-contact-settle bookkeeping extension remains explicit;
physics and collision/contact thresholds are not relaxed. The source remains
the pinned `capstone_automateData` snapshot, not an asserted latest GitHub scene.

## Runtime, resources and status

Campaign: **`outputs/five_episode_openhold_20261020`**.
The detached local supervisor has a **three-hour total hard cap**; training has
a separate 100-minute outer cap and a soft completion reserve. It checks actual
heartbeat files every ten seconds, stops at 85 C, preserves at least 1 GiB
available RAM and **25 GiB free disk**, and never automatically retries or deletes
checkpoints. A process-scoped helper prevents sleep while this supervisor lives.

47 selected tests passed before launch. Launch free space was about 28.79 GiB.
GPU-memory and runtime expectations come from this machine's 8 GiB RTX 4060 Ti,
not a 4090. A model bundle is saved only after the fixed training budget passes.
Handoff status: the fresh expert replay on `train_20002` passed; the remaining
expert qualification is underway. The new 4,000-update training has not yet
started. No policy improvement is claimed from an expert replay.

- Selection/source receipts: `selection.json`, `plan.json` and hash archives.
- Live status: `health.json`; stages: `stages.json` and individual logs.
- Positive controls and outcomes: `conditions.json`.
- Qualification: `qualification.json`; final sample audit: `training_audit.json`.
- Training/physical probe logs: `training/updates.jsonl`, `training/report.json`.
  For the five-parent contract, use `binding.training_parents` and the selection
  receipt, not the legacy single-parent `episode`/`eligible_indices` fields.
  The former retains the original control record; the latter is the last
  loaded parent's local pool. Sampling and the independent audit use the
  correctly scoped per-parent pools, never those legacy summary fields.
- Terminal status: `completion.json` or `failure.json`. Neither a launch note
  nor finite training loss is evidence of successful pickup.
- After completion: `results/five_episode_openhold_20261020/training_pickup.png`,
  PDF and audited summary JSON.

Implementation: `learning/vla/five_episode.py`, opt-in `--selection` in
`scripts/train_gripper_feedback.py`, `scripts/run_five_episode_openhold.py`,
and `scripts/report_five_episode.py`. The original single-parent trainer source
is retained in its hash archive. Default single-parent behavior remains available.
Insertion, RL, new recovery collection and sim-to-real transfer are not started.
