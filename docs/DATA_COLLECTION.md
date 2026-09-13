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
