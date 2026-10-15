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
| `wrong_grasp` | the first pick is planned 6–12 mm toward the pin end and/or with 10–25° yaw, so the pads close on the wrong part of the housing | the close ends; or the hand is blocked before the close | open, retreat 8 cm, re-survey (new detection), re-plan the nominal grasp |
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

**First runs (2026-10-14/15, `workspace_v2_cable30`, noisy detector, fresh rows 7000–7403).**
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
- **Visibility:** a shift-only wrong grasp is hard to see from `scene_rgb` because the hand hides
  the plug; it is visible in `wrist_rgb`.
- wrong_grasp masks the whole approach to the wrong spot, about 11–13 s, or about a quarter of
  the episode.

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
