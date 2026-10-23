# Recorded-input control diagnostic

2026-10-19. Test whether the original best SmolVLA checkpoint can complete a
physical simulation rollout when its inputs are supplied from demonstration data.
No training or source-recording changes are performed.

## Protocol

Use original held-out reset `train_20004`, original best `step_0009426_best`,
seed 1000, chunk 50 and execution prefix five. Inputs refresh at each policy
update (every five commands), matching the earlier closed-loop condition.
Recorded observation i corresponds to command i; the final T+1 observation is
not used as another action anchor. Stop at source exhaustion or normal guards.

| Condition | Actor numeric state | Actor RGB | Commands driving simulator |
| --- | --- | --- | --- |
| Existing closed-loop baseline | Actual simulator | Actual simulator | VLA predictions |
| Fresh expert replay control | Not used | Not used | Recorded expert applied targets |
| Recorded-state intervention | Recorded demonstration | Actual simulator | VLA predictions |
| Recorded-observations intervention | Recorded demonstration | Recorded demonstration | VLA predictions |

Only the actor's observation is replaced. Simulated qpos/qvel, object poses,
contacts, controller memory and task state evolve through ordinary `env.step`.
No recorded-state teleportation or relaxed collision guard is used. Each saved
video shows actual simulator frames, even when the actor sees recorded images.
The recorded-state/live-image condition is deliberately counterfactual once
the simulated trajectory diverges. Full recorded inputs are external guidance,
not an autonomous policy. Physical simulated success and autonomous success
are reported separately; replay success is not VLA success.

The prior live-input baseline is reused only after verifying checkpoint hash,
source episode, source/runtime manifest, prefix and sampling seed. It is the
fresh 322-command failure in `outputs/recovery_sampling_20261019/old_scene_transfer`.
Three new conditions run sequentially with a 900-second hard limit each and
2,400-second total limit, covering model load, rendering and artifact writes.
No GPU training or real robot command is executed.

## Implementation and checks

`scripts/rollout_smolvla.py` adds explicit recorded input modes. Actor inputs
remain only 16 positions, two RGB cameras and instruction; no recorded action,
phase, qpos or diagnostic is fed into the model. Source hashes, workspace and
robot checks remain unchanged. Each assisted prediction logs its recorded index,
actual actor state, input-image source and mismatch from the real simulator.

`scripts/test_recorded_observation_control.py` supervises the run, requires a
successful numerically matched expert control before proceeding, and rejects
assisted reports claiming autonomy or physical-state playback.

`scripts/report_recorded_observation_control.py` checks every logged assisted
input against its recording and command index, checks finite actual trajectories,
decodes all video frames and produces summary JSON, CSV and a gripper plot.
Assisted videos are labeled `ASSISTED SmolVLA`, never expert command replay.
Eight diagnostic/input-readback unit tests and six existing rollout feature /
source-provenance tests passed before interpretation.

## Interpretation

If recorded-state assistance permits completion but live inputs fail, drifting
proprioception is an important limiting factor. If full recorded observations
work but numeric-state assistance does not, actual image/context drift also
matters. If both fail while exact expert replay succeeds, approximate learned
actions may still be insufficient for physical grasp/contact precision; inspect
stage-specific joint/gripper errors before assuming that closed-loop recovery
data alone is enough. Any conclusion is limited to this reset and noise seed.

## Completed results

All three new conditions finished normally within 437.05 seconds total. The
ordinary closed-loop baseline below was reused from the immediately preceding
source-matched evaluation after binding checks; it was not rerun here.

| Condition | Commands | Grasp | Insertion | Outcome |
| --- | ---: | --- | --- | --- |
| VLA, actual state + actual images | 322 | No | No | Finger self-contact, `robot_collision` |
| Exact expert applied-command replay | 1,703 | Yes | Yes | `success` |
| VLA, recorded state + actual images | 497 | No | No | Finger pad/table contact, `table_collision` |
| VLA, recorded state + recorded images | 511 | No | No | Finger pad/table contact, `table_collision` |

Expert replay qpos/qvel errors are both exactly zero across the complete
trajectory. This freshly confirms that the original recording is executable
through the ordinary simulator interface and original guards.

Recorded numeric state removes the early empty-hand collapse in this test:
actual gripper travel at command 320 is 35.227 mm, rather than the near-2 mm
closure at the ordinary rollout's abort. Minimum actual opening over the
recorded-state and recorded-observations runs is 33.092 / 33.664 mm. Mean gripper
tracking error versus the demonstration falls from 11.033 mm in the shorter
live-input run to about 0.483 / 0.484 mm. These averages have unequal trajectory
lengths; the curves and explicit command-320 comparison are the better matched
evidence for suppression of early closure.

Both assisted conditions nevertheless contact the table during descent, before
the expert's first `close` command at 523. Peak support-contact forces are
30.041 / 30.035 N, crossing the unchanged 30 N abort threshold. Both have zero
peak unwanted robot contact, unlike the ordinary finger self-contact failure.
Both actors used exactly the intended recorded numeric observations at every
prediction anchor (100 and 103 verified chunks). Full recorded images therefore
did not rescue task completion on this reset.

Separate forward-kinematics readback of saved physical qpos (not state playback
during a rollout) gives the hand error at each assisted abort:

| Assisted condition | Position error vs recorded hand | Vertical delta | Orientation error |
| --- | ---: | ---: | ---: |
| Recorded state / actual images, command 497 | 9.211 mm | -1.960 mm | 2.235 degrees |
| Recorded state / recorded images, command 511 | 7.246 mm | -1.196 mm | 1.508 degrees |

The grasp site `robot_grasp` is computed using the source-hash-verified scene
and saved robot configurations at the same reference command index. In the
state-only run the contacted left finger-pad center is 1.938 mm lower than the
recorded pad center. Mean right-joint errors over the complete assisted traces
are only about 0.0072 / 0.0071 radians, illustrating why a small aggregate
joint error does not establish adequate contact-path accuracy.

## Conclusion and next gate

The answer separates expert execution from learned control: **the recorded
expert actions complete the task; the VLA still does not complete it when fed
recorded state, even with recorded images as well.** Recorded numeric inputs
suppress the original gripper-collapse failure, but a descent/contact error
then remains. Do not interpret this as proof that the model architecture cannot
learn the task, or that all implementation issues have been ruled out.

This is one **validation** episode (`train_20004` is its historical name), not a
seen-training-episode overfit test. Also, forcing all numeric observations to
the reference removes genuine joint-state feedback after physical divergence.
It is a useful diagnostic, not a deployable controller or definitive attribution
of the remaining error to data quantity alone.

Next isolate recorded **gripper only**, retaining real joint state and images,
then test a known training episode / tightly controlled overfit case if arm
accuracy remains insufficient. Keep nominal closing/holding examples, inspect
contact-path action error and the requested/applied target representation, and
require actual physical grasp progress before collecting a broad new release
or starting another long run. Do not weaken contact guards to make this pass.
Neither follow-up intervention nor new training was launched automatically.

## Artifacts and reproduction

- Reviewed plot, CSV and verified JSON: `results/recorded_input_control_20261019/`.
- Three new MP4s, actual trajectories, chunk/input logs, manifests and reports:
  `outputs/recorded_input_control_20261019/{replay,recorded-state,recorded-observations}/`.
- All four videos (including the reused baseline) decoded fully: 323, 1,704,
  498 and 512 frames, each matching actual commands plus the initial frame.
- Eight diagnostic tests, six existing rollout source/feature tests and four
  recovery-evaluation regression tests passed. Modified scripts compile and
  `git diff --check` passes. No new model/checkpoint or training run was created.

```powershell
.venv-smolvla\Scripts\python.exe scripts/test_recorded_observation_control.py `
  --output outputs/recorded_input_control_repeat

# PyAV verification uses the training environment; plotting uses system Python.
.venv-smolvla\Scripts\python.exe scripts/report_recorded_observation_control.py `
  --evaluation outputs/recorded_input_control_20261019 `
  --episode data/drive_v2c30_20261012/merged_all_v1/train_20004 `
  --output results/recorded_input_control_repeat --verify-only
python scripts/report_recorded_observation_control.py `
  --evaluation outputs/recorded_input_control_20261019 `
  --episode data/drive_v2c30_20261012/merged_all_v1/train_20004 `
  --output results/recorded_input_control_repeat --plot-only
```
