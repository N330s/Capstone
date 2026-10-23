# Recovery collection format review

2026-10-19. This is an intake review, not a training run or a policy-success claim.
Follow-up recovery integration and completed pilot results are recorded in
[RECOVERY_TRAINING.md](RECOVERY_TRAINING.md); the intake status below is historical.
The original 101-episode export, normalization, checkpoints and source snapshot
were not modified. Three new recovery manifests have been retrieved from public
Drive links; the old `v2c30_a/b/c.rar` files were not downloaded again.

## Manifest inventory

| Collection | Episodes | Commands | Expert | Injected fault | Recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| Premature closing | 10 | 23,119 | 20,274 | 1,320 | 1,525 |
| Wrong grasp | 10 | 25,238 | 17,224 | 6,654 | 1,360 |
| Misaligned insertion | 10 | 21,715 | 18,111 | 1,937 | 1,667 |
| Total | 30 | 70,072 | 55,609 | 9,911 | 4,552 |

These are manifest counts, not independently replayed successes. All 30 episodes
are labeled `train`; their names/seeds/content hashes are distinct across these
three manifests. A recovery validation holdout has not yet been created.

## Important compatibility findings

- The declared robot contract remains OpenArm v1, right arm active, left parked,
  50 Hz, 16-element bimanual position state and eight absolute right-arm targets.
  The declared robot configuration matches the old collection.
- New metadata describes a chest RGB-D camera in addition to the overhead and
  wrist views, three aligned depth streams, and 14 joint torques ordered left
  joint 1..7 then right joint 1..7. Depth is declared in 1 mm units, with zero
  representing invalid/out-of-range pixels; it must not be treated as ordinary
  8-bit RGB or normalized as a color image.
- **Torque is recorded after the action, not at the preceding observation.**
  Each `(T, 14)` row summarizes physics substeps during `action[i]`: capped
  motor torque, peak absolute motor torque, `qfrc_bias` (gravity + Coriolis), and
  `qfrc_constraint` (contacts + joint limits). Appending row `i` to observation
  `i` would expose future information. Any later torque-conditioned actor needs
  a causal previous-interval feature or a newly recorded boundary measurement,
  with explicit reset handling and hardware-equivalent semantics.
- The scene, source hashes, workspace and camera calibration changed. The three
  recovery manifests share the new scene hash
  `5689c26ee022d7d4cae29c77d27bec2cf1c096d79a96f95e4afdeaeecd856422`.
  They must not silently be merged into the old source-bound export. Obtain the
  matching collector/environment source before replay or policy evaluation.
- GitHub's `automateData` branch still resolves to
  `b7297ba82701fe7e7814c32ea617a85f832b74a4` during this review. Its
  `envs/openarm_insert.py` and `envs/workspace.py` hashes differ from the new
  recordings, and `envs/d435.py` returns HTTP 404 at that commit. The environment
  used for this data is therefore not that pinned branch revision. Obtain the
  collector's exact new source revision (including depth/fault modules and
  workspace configuration) before replaying it. The existing local snapshot
  remains unchanged.

## Required training changes

Each manifest explicitly says:

> mask step_label == 'fault' before BC; current loaders do not

The current training reader only knows terminal padding and does not implement
this recovery-label contract. **Do not train these archives with the old loader
unchanged.** Injected faulty actions are not expert correction targets.

The next adapter should:

1. Reject `fault` anchor steps, and exclude every `fault` target inside future
   action chunks. Checking only the anchor is insufficient with horizon 50.
   A conservative first implementation ends a valid chunk at the next fault
   boundary, holds its last valid target for padding, and masks that padding.
   Resume sampling from the original timeline after the fault; do not delete
   fault rows and concatenate unrelated time intervals.
2. Retain actual observations following the disturbance and the `recovery`
   corrective commands. Preserve terminal flags, action units and label
   provenance. Fault/recovery labels and privileged diagnostics are not actor
   observations.
3. Preserve raw torque/depth/chest channels, but initially keep the baseline
   model's explicitly selected overhead/wrist RGB + 16-position inputs. Changing
   sensory architecture is a separate experiment. The simulator torque semantics
   are documented above; establish causal alignment and real-hardware equivalents
   before treating those values as actor features.
4. Define whole-episode train/validation splits within each recovery type; retain
   the original untouched validation set. Review scene compatibility deliberately
   and fit or preserve normalization with an explicit checkpoint/data binding.
   Do not silently reuse statistics or resume an old optimizer on a changed
   dataset under the identity of the original run.
5. Verify every downloaded episode's checksum, labels, timing, boundaries,
   camera/state/action schema and complete-release coverage before export.
   Qualify exact expert replay using the matching new environment, then run a
   bounded recovery fine-tune and compare closed-loop grasp success with the
   unchanged baseline and guards.

This is a more relevant next dataset for the observed gripper-feedback failure
than simply repeating the earlier clean demonstrations. It is not evidence that
recovery training has already fixed that failure.

## Inspection tools and storage

`scripts/inspect_drive_recovery.py` uses bounded Drive byte ranges and verified
RAR5 quick-open headers to fetch selected lossless recordings without all JPEG
previews. It runs in `.venv-download` and uses the installed WinRAR `UnRAR.exe`;
the Windows libarchive backend cannot unpack this archive dictionary setting.
No code from an archive is executed. CRC checks, path checks, remote identity,
transfer and free-space bounds remain enabled.

`scripts/audit_recovery_collections.py` runs in `.venv-smolvla` and writes a
read-only manifest/sample report plus actual RGB/depth contact sheets. The first
manifest-only report is `outputs/recovery_intake_20261019/metadata_audit/report.json`.

The three recovery archives total 27,953,973,687 bytes (26.03 GiB); there was only
about 42 GiB free at intake. Downloading every archive and extracting everything
would leave inadequate headroom. Selective sampling is intentional. A full
release download/export needs a separately checked storage allocation.

## Verified lossless sample

One episode per recovery type was downloaded successfully: `train_09000` (2,472
commands), `train_09100` (2,436), and `train_09200` (2,050). Archive CRC and the
recorded SHA256 checksums passed. All three pass the existing episode validator,
retain every legacy array, have aligned `T` actions / `T+1` position and RGB
observations, 50 Hz timing, valid terminal boundaries and matching label counts.
Passing that legacy validator alone does **not** make their fault targets safe
for behavior cloning.

Actual additions in all three NPZs:

| Field | Shape | Representation |
| --- | --- | --- |
| `step_label` | `(T,)` | `expert`, `fault`, `recovery` |
| `image_chest_rgb` | `(T+1, 240, 424, 3)` | uint8 RGB |
| `depth_scene_mm`, `depth_wrist_mm` | `(T+1, 240, 320)` | uint16 millimetres |
| `depth_chest_mm` | `(T+1, 240, 424)` | uint16 millimetres |
| Four `joint_torque_*_nm` arrays | `(T, 14)` each | float64 N m; post-action interval summaries |

In `train_09000`, the injected-fault interval is `[484, 616)` and recovery is
`[616, 780)`. Recorded finger travel changes from 34.99 to 11.42 mm during the
fault and returns to 34.99 mm during recovery. This is directly relevant
corrective supervision, although exact expert replay has not yet been qualified
for this new source revision.

Report: `outputs/recovery_intake_20261019/sample_audit_v2/report.json`.
Three actual RGB/depth review sheets are in that same directory. Depth ranges
and invalid-pixel fractions were inspected on one recovery frame per episode,
not exhaustively over all frames. Remaining 27 episodes have not been downloaded
or validated. No model-input, training-loader, normalization, checkpoint, physics,
or safety-guard changes were made, and no training was started.
