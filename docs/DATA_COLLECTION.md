# First varied demonstration batch

Collected 2026-09-12 in `data/openarm_v1_varied10/`, separate from the original
20-episode pilot. All ten preflights and recorded episodes succeeded.
All ten command replays reproduce qpos/qvel exactly under the new recorded source
version. Dataset split, normalization round-trip and terminal-mask checks pass;
25 regression tests pass.

- 1,746 command transitions: 1,401 training, 345 validation (8/2 whole episodes).
- Requested reset perturbations: approach distance up to +/-2 mm, lateral offsets
  up to +/-0.8 mm, and rotations up to +/-0.5 degrees, including combined cases.
- These perturb the nominal robot grasp, not an ideal socket-aligned plug pose.
  Post-settle measured offsets differ; actual values are in the inspection report.
- Both OpenArm v1 arms remain present; right active, left parked. Physical finger
  grasp, controller, friction, images and command conventions are unchanged.
- No retry occurred. There are **zero recovery demonstrations** in this batch.
- 100 deterministic evaluation reset conditions are reserved in
  `evaluation_bank.json`. They have not been executed or used for expert tuning.
  They are an in-distribution test bank, not an OOD benchmark.

## Reproduction and validation

```powershell
# Output must not already exist; omit --record for physics-only preflight.
python scripts/collect_varied.py --record --output data/openarm_v1_varied_new
python scripts/replay_pilot.py --dataset data/openarm_v1_varied_new
python scripts/prepare_vla_data.py --dataset data/openarm_v1_varied_new
python scripts/inspect_varied.py --dataset data/openarm_v1_varied_new --output results/varied_new_inspection
python -m unittest discover -s tests -v
```

Banks are saved before collection. There is no silent replacement of failed resets.
A failed preflight stops image collection and leaves its diagnostics. A failed
recorded rollout leaves command/state traces separately from successful BC data.
Only a manifest with `complete=true` is a finished batch.

The reset API now accepts axial and angular offsets. This changed the environment
source hash. Original pilot files and reports were not rewritten; their exact
replay claim applies to their recorded source version. The strict replay guard
will reject that older dataset against the changed source. New recordings carry
the new source hash; do not bypass provenance checks to claim old/new equivalence.

## Visual inspection

See [wrist contact sheet](../results/varied10_inspection/wrist_rgb_sheet.png),
[scene contact sheet](../results/varied10_inspection/scene_rgb_sheet.png), and
[trajectory summary](../results/varied10_inspection/summary.json).

The wrist image shows the plug and slot entrances during approach, but housing
and gripper obscure the blade/slot interface near seating. The wide camera shows
the robot and fixture but offers little precision detail at 320x240. The 3x wrist
previews only enlarge recorded pixels; they add no information.

This visual review does not establish that the policy can infer sub-millimetre
alignment. Also, the exact-pose expert aligns before insertion, so most contact
trajectories converge to the same nominal behavior. Ten successful trajectories
are not proof of diverse precision-control coverage.

## Recovery demonstrations (injected faults)

`scripts/collect_random.py --faults ...` (2026-10-14, `controllers/faults.py`) collects episodes in
which the scripted expert makes **one deliberate mistake** and then recovers and completes the
task. These are scripted, injected faults. They are not failures observed from a policy, and no
training run has shown that they help yet.

| Fault | Mistake | Detected by | Recovery |
|---|---|---|---|
| `premature_close` | fingers close part-way down the descend, 20–60% of the way, above the housing | fingers at the closed command with no pad on the plug for 0.2 s (robot-side signals only) | reopen, rise back to the pregrasp, normal descend |
| `wrong_grasp` | **misses the plug entirely** (since 2026-10-15): the first pick goes to empty table 6.5–9.5 cm from the housing grasp point, past the pin tips or to either side (never toward the cable), and closes on nothing. The open fingers are checked to stay ≥ 10 mm from the true plug and the cable all the way down. | fingers at the closed command with no pad on the plug for 0.2 s (as premature_close); or the hand is blocked before the close | open, retreat 8 cm, re-plan the nominal grasp from the same detection (the plug was not touched) |
| `misaligned_insert` | the first align/insert aims 2–3 mm off the socket holes in the socket (y, z) plane | the existing socket-force (5 N) / stall rule | withdraw, re-align on the holes, insert |

- **One fault per scene.** The kind cycles with the seed, so `--faults all` is balanced across the
  three kinds. The parameters come from their own random stream, so the rest of the rollout is
  the same as a clean run's.
- **Without `--faults`, nothing changes.** A clean rollout is bit-identical to the pre-change code
  (row 6100: same qpos trace), and episodes have no `step_label`.
- **`step_label[i]`** describes `action[i]` and is one of `expert`, `fault` or `recovery`:
  - a step that touches the injected mistake is labelled `fault`;
  - the back-off is labelled `recovery`;
  - everything else is labelled `expert`.
- **Episode metadata** gains `fault` and `label_counts`; the manifest gains `fault_injection` and
  `training_note`. `validate_episode` checks the labels: a fault must be followed by recovery steps,
  and no episode may end on a fault step.
- **`fault_not_exercised`.** A faulted scene that succeeds without its fault and recovery happening
  is rejected under this outcome. Failed faulted scenes are kept, as usual, under
  `failed_preflight/` and `failed_attempts/`.
- **Mask fault steps before behaviour cloning.** The current VLA loaders (`action_chunk`,
  `vla_dataset.py`, `learning/vla/`) do **not** read `step_label`, so they would train on the
  injected mistakes. Mask `step_label == "fault"` actions as targets, and leave them out of the
  normalization fit, before training on these datasets. Keep recovery datasets in their own
  `--output`.

```powershell
python scripts/collect_random.py --workspace configs/workspace_v2_cable30.json --faults all --episodes 6 --record --headless --output data/openarm_v2_recovery_new
python scripts/collect_random.py --workspace configs/workspace_v2_cable30.json --faults premature_close --episodes 4 --video all --headless --output data/recovery_smoke_new   # preflight + video only
```

**First runs (2026-10-14/15, `workspace_v2_cable30`, noisy detector, fresh rows 7000–7403).** The
wrong_grasp numbers and frames here are for the earlier *off-centre pinch* version (shift/yaw),
replaced by the miss on 2026-10-15; see below.
- **24 faulted attempts with the final code: the injected fault was detected and recovered in
  24/24**, and the whole task then succeeded in 18/24:
  - premature_close 6/8;
  - wrong_grasp 5/9;
  - misaligned_insert 7/7.
- **Failures after the recovery:**
  - 3 were the known engaged-leaf insertion stall (not re-run clean);
  - 2 also fail clean on the same scene (7154, 7155);
  - 1 is attributable to the recovery: in wrong_grasp row 7401, the re-survey's fresh noisy
    detection (5.1 mm error, against 3.9 mm on the first look) left the re-grasp off-centre, so the
    seated pose was infeasible. The same scene succeeds clean.
- These are small samples. The rates are smoke-test evidence, not measured success rates.
- **Recorded:** 6 episodes in `data/openarm_v2_recovery_smoke_a` and `_b` (2 per kind), 130–223
  fault steps and 134–207 recovery steps each. All pass `validate_episode`, all retraced
  their preflight exactly (`trace_matches_preflight`), and all pass strict `replay_pilot.py`
  replay with 0.0 qpos/qvel error. Wrist-camera frames show the closed empty
  fingers, the grasp near the pins, and the pins on the socket face, each followed by the
  recovery and a seated plug.
- **Tuning from the smoke runs:**
  - wrong_grasp shifts go toward the pin end only. Toward the cable end, the hand landed on the
    cable, which the planner ignores (row 7101).
  - The insert aim error is capped at 3 mm, since 3.41 mm reached 3.78 mm lateral, close to
    PlaceInsert's 4 mm off-axis abort.
- **Visibility (old version):** a shift-only off-centre pinch was hard to see from `scene_rgb`
  because the hand hid the plug. That is why wrong_grasp became a clear miss.
- wrong_grasp masks the whole approach to the wrong spot, about 11–13 s, or about a quarter of
  the episode.

**2026-10-15: chest RGB-D camera, joint torque, wrong_grasp = miss.**
- **New arrays.** Episodes recorded in `workspace_v2_cable30` now also contain:

  | Array | Shape | Contents |
  |---|---|---|
  | `image_chest_rgb` | (T+1, 240, 424, 3) | chest RealSense D435-like RGB |
  | `depth_chest_mm` | (T+1, 240, 424) uint16 | depth aligned to that RGB, mm, 0 = no data |

  See `docs/WORKSPACE.md` for the camera and the depth model.
- **Joint torque in every new episode from any workspace.** `joint_torque_applied_nm`,
  `joint_torque_applied_peak_nm`, `joint_torque_gravity_nm` and `joint_torque_constraint_nm`, each
  (T, 14) in N m:
  - order: left joint1..7, then right joint1..7;
  - row i covers action i's interval;
  - mean over the 40 physics substeps (the peak array is the max |·|);
  - the same values are in `diagnostics.json`.
- **Torque is not a policy input.** The 16-d state is unchanged.
- **Strict replay of earlier datasets is lost.** The chest camera and the torque logging changed
  `envs/workspace.py` and `envs/openarm_insert.py`, which `env.manifest()` hashes, so
  `replay_pilot.py` refuses **every dataset recorded before 2026-10-15**, including the six recovery
  episodes above.
- **Physics did not change.** A clean row-6100 rollout reproduces the pre-change qpos trace
  bit-for-bit (1981×148 qpos and phases, compared with the trace recorded at HEAD bc72ba3).
- Recollect if strict replay of the old data is needed.
- **Cost:** with the chest camera an episode is about 2.5–2.9× larger on disk (404–585 MB, 0.21–0.24 MB/step, against 0.084 MB/step before).
  Recording one is verified end-to-end (4 kinds, strict replay 0.0 error; `docs/AUTOMATED_COLLECTION_LOG.md`).
- **Miss smoke runs.** Preflight + video, not recorded; rows 7500–7501, 7550–7551, 7600–7601;
  noisy detector.
  - **6/6 succeeded.** Every miss closed on empty table: 0 plug contacts, detected as an empty close.
  - **Clearance:** 7.1–8.7 cm from the housing grasp point, with the open fingers 13–42 mm from
    the plug and the cable.
  - **Timing:** the fault lasted 12.7–14.3 s (the trip to the miss point, then the close); the
    recovery 2.4–3.0 s.
  - **Direction fallback:** "left" was drawn twice and both times fell back to another direction,
    because no clear "left" miss existed in those scenes.
  - **Visible in `scene_rgb`:** the hand closes well away from where it later grasps the plug.

**2026-10-16: depth on the overhead and wrist cameras.**
- **New arrays.** Episodes recorded in `workspace_v2_cable30` now also contain:

  | Array | Shape | Contents |
  |---|---|---|
  | `depth_scene_mm` | (T+1, 240, 320) uint16 | overhead depth (D435 model) aligned to `image_scene_rgb`, mm, 0 = no data |
  | `depth_wrist_mm` | (T+1, 240, 320) uint16 | wrist depth (RealSense D405 model, 7 cm-1 m) aligned to `image_wrist_rgb`, mm, 0 = no data |

  `depth_chest_mm` is unchanged and its noise stream is too. Every depth array pairs with the
  `image_<name>_rgb` of the same size: `validate_episode` rejects a `depth_<name>_mm` without it.
  Other workspaces (v1, v2, v1_leaf5n) add no depth.
- **New folders.** 16-bit PNGs in `cameras/head_depth/` (`scene`), `cameras/wrist_depth/` and
  `cameras/chest_depth/`, T+1 each, numbered like the JPEGs. The metadata `depth_folders` maps
  folder to array name, and `depth_cameras` in the metadata holds the intrinsics and depth model.
- **Strict replay of earlier datasets is lost, again.** The depth streams changed
  `envs/openarm_insert.py`, `envs/d435.py` and the workspace JSON, which `env.manifest()` hashes, so
  `replay_pilot.py` refuses **every dataset recorded before 2026-10-16**, including
  `data/openarm_v2_e2e_sensors_test`. Physics is unchanged: the replayed states do not depend on
  rendering (qpos identical to the pre-change trace over reset + 20 hold steps; chest depth identical).
- **RGB is unchanged, but not bit-for-bit comparable.** The Windows GL renderer is not
  deterministic: two renders of the same state in one process differ by 1 grey level on up to about
  20 pixels per frame. Old-versus-new code differs no more than that (≤ 1 grey level, ≤ 14 px, over
  reset + 20 steps of row 6100).
- **Checked against ground truth** in a full expert rollout (row 6100, success, 1980 steps, nothing
  saved):
  - while grasping, the wrist sees the plug face at about 91 mm, with 0.96–1.00 of the plug patch valid;
  - `mj_ray` ground truth through those pixels agrees within 0.8 mm;
  - the overhead stream is about 0.9 valid throughout;
  - the wrist minimum is the D405's 70 mm cut-off, which the fingers sit at.
- **Cost (measured on 30-step `save_episode` slices, 2026-10-16):**
  - **Disk:** 0.37–0.38 MB/step on disk (was 0.21–0.24), so a 2000-step episode is about 0.75 GB
    and a 2700-step fault episode is about 1 GB.
  - **RAM while recording:** one observation is 1.22 MiB in memory (+32 %), and a recording keeps
    them all until the save. Saving a synthetic 2700-step episode peaked at 8.0 GiB working set
    (9.7 GiB commit), so plan on about 10 GB per recording process (`docs/DOCKER_COLLECTION.md`).
  - **Speed:** `observe()` is about 9 ms slower by median (12 → 21 ms; two extra 320×240 depth
    renders).
- The real-sensor limits (for example, the D405 has no IR projector) are in `docs/WORKSPACE.md`.

## 2026-10-27: data requirements pass (warm-up, posture, labels, quality, replay, release)

Collection-side changes so that a new `workspace_v2_cable30` dataset meets seven data requirements.
**No dataset has been collected with them yet**. The evidence below comes from audits and two
temporary recordings that were deleted afterwards: 5 episodes on intermediate code and 3 on the
final code. Nothing in `learning/` or `vla_dataset.py` changed.

- **Camera warm-up is in `reset`, not in the demonstration.** `configs/openarm_v1.json`
  `warmup_s` (0.88 s = 44 command steps) runs after the settle, through the same `_advance` code
  as `step()` (including the `mj_forward` at each boundary), plus one discarded render. Then the
  outcome, hold, elapsed time, peaks and grasp latch are reset. Tabletop mode only.
  - **No more still `survey` prefix.** `PickInsertExpert` latches its detection on the first
    `action()` (same `sense()`/rng order) and returns the first `transit_pick` action. Obs 0
    has `timestamp_s` 1.88 s.
  - **Regression (row 6100, zero posture offset):**
    - obs-0 state is bit-identical to the old trace's qpos[44];
    - the rollout takes 1935 = 1980 − 45 steps, with the same phase sequence minus `survey`;
    - final plug position differs by 0.013 mm.
- **Right-arm start posture varies.**
  - **Option:** the tabletop reset option `arm_offset_rad` (7 values) is added to
    `home_arm_rad`. The env validates |each| ≤ 0.15 rad and keeps the posture ≥ 0.05 rad inside
    the joint limits.
  - **Sampler:** draws it uniformly within ±`arm_offset_max_rad` (0.05) from its own rng
    stream (`POSTURE_STREAM`), so plug, socket and visual draws are unchanged for every seed.
  - **Checks:** the open-hand home-contact check, the pick check and the noise-margin check
    run from the perturbed posture. Up to 8 posture tries, then a zero offset; `diag["posture"]`
    records which.
  - **Measured on 30 audit rows:** per-joint std 0.022–0.032 rad, max 0.0498; 0 reset
    collisions; 30/30 success.
- **Fault commands are kept out of the expert labels** (data level). Episodes now carry
  `action_label` (T, 8), equal to `action_applied` except NaN on `step_label == "fault"` steps,
  and `label_mask` (T,), False exactly on those steps. Clean episodes have all-True masks.
  `validate_episode` enforces both. **The training loaders still read `action`**: use
  `action_label`/`label_mask` when they are updated.
- **Control contract recorded per episode.** Metadata `pilot_contract`:
  - cameras: `scene_rgb` + `wrist_rgb`;
  - state 16, action 8 (7 absolute rad + finger travel m);
  - 50 Hz, observations before actions;
  - left arm parked;
  - depth/torque/visibility listed as extra arrays, not policy inputs.

  The terminal step may be shorter than 0.02 s, because an episode ends mid-interval on
  success (`timing_note`; pre-existing). `tests/test_contract.py` checks a saved episode.
- **Physical quality gate** (`data_pipeline/quality.py`). A success is rejected as:
  - `quality_no_grasp` when there is no grasp latch;
  - `quality_low_lift` when the plug rose < 0.03 m by the end of `lift`;
  - `quality_drop` when a drop is counted;
  - `quality_contact_lost` when, from the first lift to the end, a step lacks two-finger
    contact or the latch.

  Every summary, accepted or failed, has `stage_outcomes` (`pickup` / `carry` /
  `insertion`), so a pickup-only success is distinguished from a full insertion, plus
  `original_outcome`.
- **Visibility gate** (`data_pipeline/visibility.py`): a success where neither pilot camera sees
  the plug on some frame from obs 0 to the first `lift` step becomes `camera_occluded`.
  - **Visible** means a housing sample point is in frame and its `mj_ray` first hits the plug;
    cable bodies are excluded.
  - **Per-frame record:** `visibility_pilot` (T+1, 2) is saved.
  - **Overhead pose:** the `scene_rgb` camera was moved for this to a raised right-side mount
    (`docs/WORKSPACE.md`).
  - **Fresh audit with the final pose** (36 rows, `scripts/audit_visibility.py`):
    - clean rows 41000–41014: 15/15 success, plug seen on 100% of pickup frames;
    - fault rows 41100–41120 (7 per kind): 16/21 success, 99.3% of pickup frames seen, the misses
      only in wrong_grasp miss paths;
    - the gate would reject 1 of 31 successes (3.2%);
    - random reset draws: 99.35% visible of 4000, with the misses at plug x 0.38–0.43 m nearest the
      robot. Those placements are slightly under-represented in gated data.
- **Every recorded episode is replayed in-process** (`data_pipeline/replay.py`, the same strict
  1e-8 / 1e-7 check as `replay_pilot.py`, which now imports it). A pass is written into the
  metadata `replay`; a mismatch moves the episode to `failed_attempts/` as `replay_mismatch`.
  `--no-verify-replay` skips it (replay roughly doubles physics time). The run manifest gains
  `collection_source_hashes`: the expert, place-insert, faults, sampler, quality, replay,
  visibility, episodes, collector and `openarm_v1.json` files.
- **Versioned release** (`scripts/make_release.py`): hard-links the episodes of one or more
  complete runs into `data/releases/<name>-v<N>/` and writes:
  - **`release.json`:**
    - git commit and dirty flag;
    - source, workspace and collection hashes;
    - camera calibration;
    - controller and robot config;
    - the pilot contract;
    - per episode: seed, reset options, checksum, fault, quality, replay and stage outcomes;
  - **`splits.json`:** grouped by scene seed (every episode of a seed, clean or faulted, in one
    split), with a hash rule for about 15% validation and at least one validation group;
  - **`heldout_resets.json`:** the first N `evaluation_bank` resets, asserted disjoint;
  - **`checksums.sha256`.**

  **It refuses:**
  - an existing output;
  - incomplete runs;
  - corrupt episodes;
  - runs missing replay verification (`--replay` replays them);
  - runs from mixed source, workspace or collection versions.
- **Temporary-recording check (deleted afterwards):**
  - **Episodes:** 5 episodes, 2 clean plus 1 of each fault kind.
  - **Faults:** every fault was detected and recovered.
  - **Labels:** NaN exactly on fault steps.
  - **Replay:** in-process and `replay_pilot.py` replay error 0.0 / 0.0.
  - **Release:** the test release split 3 train / 2 validation with groups kept and held-out
    resets disjoint; 64,356 checksums verified; it refused an overwrite and an old-version run.
  - **Final code (2026-10-27/28), 3 more episodes:**
    - clean rows 60000–60001, plus wrong_grasp. The gate rejected wrong_grasp row 60000 as
      `camera_occluded`, so row 60001 was recorded.
    - All replay with 0.0 / 0.0 error in-process and in `replay_pilot.py`.
    - Labels: NaN on exactly the 680 fault steps.
    - Obs 0 at 1.88 s, first phase `transit_pick`; offsets ≤ 0.046 rad; no unseen pilot frame before lift.
    - The 10 collection hashes match.
    - A `make_release --heldout 3` test release:
      - kept seed 1060001's clean and faulted episodes together in validation (1 train / 2 validation);
      - its held-out seeds were disjoint;
      - 37,236 checksums verified; it refused to overwrite.
    - Timing on the 16 GB Windows PC, one process alone: about 35 min per scene for preflight,
      record and in-process replay. `make_release` sampled 3 held-out resets in 262 s.
    - The wrong_grasp `fault_events` (`fault_start` / `detected` / `recovered`, with phase) are
      in the run's `progress.json` and `preflight_accepted.json`, not in the episode metadata.
- **Strict replay of every earlier dataset is lost again.** That includes the 30 recovery
  episodes and `v2c30_*`, because the env, sampler, expert and workspace changed. They also cannot
  join a release, because they lack the replay record and the collection hashes. Recollect.

```powershell
# clean + faults from the same rows, so each seed's variants share a split; then one release.
# --eval-scenes 100 samples the held-out bank once (about 3 min per scene) and caches it, so
# make_release's heldout_resets.json (default 100) is then a cache hit.
python scripts/collect_random.py --workspace configs/workspace_v2_cable30.json --start-row 20000 --episodes 50 --eval-scenes 100 --record --headless --video none --output data/v2c30_release_clean
python scripts/collect_random.py --workspace configs/workspace_v2_cable30.json --start-row 20000 --episodes 30 --faults all --record --headless --video none --output data/v2c30_release_faults
python scripts/make_release.py --runs data/v2c30_release_clean data/v2c30_release_faults --name v2c30 --version 1
```

## Next collection/training gate

1. Inspect a physically plausible close oblique external camera alongside the
   wrist camera, using the existing recordings/reset bank for development only.
   Measure visibility of housing/slot edges before versioning the camera schema.
   Do not alter these saved images or present an oracle overlay to the policy.
2. Build the **SmolVLA** data adapter, preserving physical units, train-only
   normalization and terminal masks. Validate its actual upstream preprocessing.
   Keep the pi0 adapter as a separate experiment, not an interchangeable adapter.
3. Run one-batch forward/backward and VRAM profiling on the local 8 GB GPU, then
   overfit one or two episodes if the runtime fits. No SmolVLA training has run yet.
4. After that integration check, collect toward the proposed 100-episode budget
   with measured coverage and genuine successful recoveries. Keep fault-injected
   bad commands out of successful BC labels and preserve failures for later RL.
5. Freeze the camera/controller/data bundle before evaluating held-out resets.
   Defer broad contact randomization until supported by calibration or justified
   bounds; retain the unpowered fixture requirement for physical transfer.

## Preview runs: --no-depth

`collect_random.py --no-depth` clears `env.depth_cameras` right after each env is built. No
`depth_*_mm` arrays or `*_depth` PNG folders are then saved, and `manifest()["depth_cameras"]` is
`null`. When the flag is set, `manifest.json` and the `--resume` run signature gain `no_depth: true`;
without it nothing changes.

It saves little time. Measured on one process (2026-10-31): rendering 3 RGB + 3 depth streams
takes 14 ms per frame, 3 RGB alone 6 ms, and one physics step 150 ms. That is about 13 s per
~1,650-frame episode. The main gain is memory, since depth is about 40% of the image data held
while recording. The time goes into the physics, which runs three times per saved episode
(preflight, record, replay check).

Preview runs usually also pass `--no-verify-replay`. `make_release` cannot package such runs, so
use a separate `--output` and keep them apart from `v2c30_*` data.
