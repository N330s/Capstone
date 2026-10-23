# Gripper feedback investigation and controlled training change

Update: [TRAINING_PRECISION_FIX.md](TRAINING_PRECISION_FIX.md) supersedes the
pending paired comparison below. Its control completed 800 updates, but an
optimizer precision defect was then confirmed and the augmented condition was
stopped at 94 updates. No equal-budget paired result exists. The original source
files/artifacts are archived. A corrected FP32-storage run and explicit physical
grasp-bookkeeping qualification are now executing; no success is claimed yet.

The observed failure is **premature closure during approach**, before touching
the plug. It is not currently an insertion-precision failure. The original expert
replays successfully in the same source-matched simulator, while the VLA's measured
finger travel collapses from about 44 mm to 1.8 mm. Low aggregate flow loss did not
prevent that failure, including on a training reset.

## Findings checked before changing training

1. **Numerical inference was not a clear fix.** Same observations and Gaussian
   noise, BF16 versus FP32 and 10 versus 50 integration steps: nominal gripper MAE
   stayed about 0.75--0.79 mm; changing measured gripper position still changed the
   prediction almost proportionally. This was four anchors, three seeds, and two
   offline sensor perturbations, not a physical recovery test.
2. **Actuator-resolution filtering was insufficient.** Explicit one-mm rounding
   of the predicted finger target, without altering arm targets, still failed
   before grasp at command 331 (original continuous-target baseline: 322).
   This diagnostic is optional and off by default; it is not the training fix.
3. **A second environment limitation exists but does not explain early closure.**
   The source collector resets its grasp-settle accumulator on any finger target
   change, and treats opening the target by 0.1 mm after latching as release.
   Continuous noisy predictions can therefore prevent stable grasp bookkeeping
   even when contact is physically present. The current trials fail before plug
   contact. No environment code, guards or success definitions were changed.

## Paired training test

Both conditions start from the original best checkpoint, use `train_20002` only,
and use identical sampled indices, model-noise RNG sequence, batch eight, BF16,
and a **completed 800-update learning-rate schedule** (peak 1e-5, end 2.5e-7).
Each batch contains four open-hold anchors and four uniform same-parent anchors.
The open-hold eligibility test uses only recorded state/targets: the entire next
50-target window is constant and at least 30 mm open, and measured finger travel
is within one mm of the target. It excludes closing/contact-hold windows.

- Control: original observations and targets.
- Augmented: perturb only measured right-finger position of the four open-hold
  samples by a uniform amount in [-2, +2] mm, clipping to physical observation
  bounds. Keep RGB, all arm joints, instruction, future action targets and terminal
  masks unchanged. No augmentation is applied at rollout time.

This is artificial proprioceptive augmentation: unchanged RGB is deliberately
paired with a small modified sensor reading. It is not newly collected physical
recovery data. The hypothesis is that this teaches a restoring open command
instead of copying each slightly incorrect measured opening. It does not hardcode
when the deployed actor should open or close, and does not remove the gripper
feature or add privileged phase/contact information to actor inputs.

Probes at 0/200/400/600/800 use four open-hold anchors (nominal and +/-2 mm), five
close/hold/insertion anchors and three fixed Gaussian noise seeds. They report
physical prefix-five gripper/joint error and gripper sensor gain. A gain near one
means predictions copy measurement changes; a lower gain on open-hold probes
indicates more restorative behavior, not by itself successful grasping.

Both final candidates must reload in a fresh process with exactly identical
physical predictions. Each then runs in the original simulator from the same
initial reset, seed 1000 and prefix five, using live RGB and measured positions.
The rounding filter is disabled. Physics, contact limits and grasp success
definitions remain unchanged. The expert replay and original baseline are retained
from the immediately preceding source-matched one-parent test.

## Storage, supervision and implementation

`learning/vla/expert_overlay.py` saves only the 99,880,992 trainable parameters
(399.5 MB each) and binds them to exact base weights/config/processor files. These
are full trainable-expert replacements, **not LoRA or optimizer-resume bundles**.
Frozen VLM weights and normalization remain in the immutable original base. Exact
tensor names/shapes, file hashes and finite values are checked before loading.
This avoids duplicate frozen model weights and preserves the 25 GiB disk reserve.
Nothing from previous experiments is deleted or overwritten.

`scripts/run_gripper_feedback_test.py` enforces a one-hour total cap, 20-minute
training-stage caps, fixed update budgets and owned-descendant cleanup. A temporary
Windows keep-awake request follows the supervisor lifetime without changing power
settings or keeping the display on. The original baseline trainer is unchanged.

```powershell
.venv-smolvla\Scripts\python.exe -u scripts/run_gripper_feedback_test.py --output outputs/gripper_control_fix_20261019/paired_feedback
.venv-smolvla\Scripts\python.exe scripts/report_gripper_feedback.py --experiment outputs/gripper_control_fix_20261019 --output results/gripper_control_fix_20261019 --verify-only
python scripts/report_gripper_feedback.py --experiment outputs/gripper_control_fix_20261019 --output results/gripper_control_fix_20261019 --plot-only
```

The report independently audits all sample/noise records, identical batch
schedules/initial predictions, fresh reloads and fully decoded simulation videos.
No task success is claimed while execution is pending. Passing one training reset
would still require held-out resets, physical disturbance tests and eventually
real-hardware calibration before any sim2real claim.

The protocol above describes the superseded first comparison, not the corrected
run. Its BF16 expert bundle is 206.3 MB due to mixed tensor dtypes, not the 399.5 MB
FP32 upper bound stated above. Corrected FP32 expert storage is 399.5 MB. Requested
and applied arm-joint targets are identical throughout `train_20002`; changing
those columns would not fix its arm motion. Neither old candidate is promoted.
