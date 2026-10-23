# Training precision repair

## Confirmed defect

The pinned SmolVLA implementation constructs most of its action expert in BF16.
Our trainers enabled BF16 autocast but did not promote those **trainable parameter
values** before creating AdamW. AdamW consequently stored their moments in BF16
too. Small updates could round away even with finite nonzero gradients and a
decreasing aggregate loss. This is separate from using BF16 for forward compute.

The 800-update low-LR control supplies direct checkpoint evidence:

| Storage dtype | Parameters | Values changed after 800 updates |
| --- | ---: | ---: |
| BF16 expert | 96,607,440 | 5,555,952 (5.75%) |
| FP32 projections | 3,273,552 | 3,258,167 (99.53%) |

That comparison alone could include genuinely small gradients. A subsequent
**real-gradient GPU update test** isolates rounding on the first expert attention
query matrix: every gradient is nonzero; the same clipped gradient with the same
AdamW settings/lr=5e-6 changes **5.48%** of BF16 values versus **99.9994%** of FP32
values. Four corrected GPU updates completed with FP32 trainable gradients and
optimizer moments. A CPU regression also reproduces a BF16 value of ~0.02 ignoring
an AdamW step at LR=1e-5 while FP32 retains it.

Evidence: `outputs/gripper_control_fix_20261019/precision_issue.json` and
`fp32_qualification/precision_update_audit.json` beneath that experiment root.
This is a real precision defect, **not proof it is the only cause** of failed
grasping. Prior claims that finite-loss tests established bug-free learning were
too strong. Old interface/replay/loss records remain valid for what they measured,
but do not qualify effective action-expert learning.

## Implemented correction

- `learning/vla/training_precision.py` promotes only trainable weights to FP32
  **before optimizer construction**, and asserts FP32 gradients/Adam moments.
  Frozen VLM parameters are not promoted or optimized.
- Forward passes retain BF16 autocast. This separation follows the standard
  [PyTorch AMP approach](https://docs.pytorch.org/docs/stable/amp.html); lower
  precision computation is not the same as lower precision optimizer storage.
- Full-checkpoint loading restores saved trainable tensor dtype/bits, because
  constructing the expert in BF16 can silently downcast an FP32 checkpoint.
- Expert-overlay format v2 requires FP32 trainable values and restores their
  precision before inference. Legacy v1 overlays retain their historical dtype.
- The normal SFT, recovery SFT, single-episode fitting, smoke and resume-qualification
  entry points use the repair. New SFT bindings include the precision-helper hash
  and FP32 storage. Old optimizer state is not silently migrated/resumed; use
  old **weights** with a new optimizer for corrected training instead.
- Original source versions were preserved under `outputs/source_archives/<sha>/`.
  Historical reporting can verify those exact archived versions. Baseline weights,
  datasets, split membership and normalization tensors are not rewritten.

The corrected checkpoint format is CPU-unit-tested for tiny-update retention,
frozen-weight preservation, FP32 moment storage and exact recovery of FP32 bits
lost by a BF16 constructor. The four-update real GPU test passed. A complete
FP32 optimizer/RNG resume campaign has **not** yet been requalified on the GPU.
Old BF16 memory/batch-size qualifications must not be reused blindly: FP32 master
weights and moments increase memory use.

## Simulator bookkeeping repair and task test

The original collector requires an exactly unchanged finger target for 100 ms
before latching a grasp, resetting the accumulator on every target change. A
continuous VLA can produce different targets every 20 ms. A versioned optional
`--physical-grasp-metrics` adapter now accumulates **physical** still contact:
the same two-finger contact, velocity and squeeze thresholds must hold for 100 ms.
It also distinguishes harmless squeezing-target jitter from physically opening
and losing contact. Slip still causes drops. This changes bookkeeping, not
actuator dynamics, geometry, forces, collision guards or object state.

The collector checkout remains immutable. The extension revision/hash and exact
thresholds are declared in each evaluation manifest/report. It is **not silently
claimed to be the identical old success-definition contract**. Future collection
should integrate/pin the repair in a new collector revision rather than omitting
its provenance. Four physical-bookkeeping regression tests pass.

`scripts/run_precision_repair.py` first requires full expert action replay to
complete grasp/insertion with zero qpos/qvel error under the declared extension.
It then tests the old VLA, trains 800 corrected updates with the bounded open-hold
augmentation, verifies fresh reload and tests the trained model with live state
and RGB. Prefixes 5/25/50 are declared control tests, stopping early on full task
success. There is no reference-action assistance to the actor, no state forcing
after reset, no quantization filter and no weakened contact guard.

The supervisor caps the complete repair experiment at one hour, training at
25 minutes, and each simulation stage at ten minutes. It preserves a 25 GiB disk
reserve and writes one FP32 expert-only inference bundle, not optimizer state.

## Completed corrected run (2026-10-19)

The repair supervisor completed in 40.97 minutes. Training finished all 800
updates in 17.44 minutes, with 6,400 same-parent anchors, including 3,200
open-hold sensor perturbations; no other training or validation parent was used.
All losses/gradient norms were finite. Peak allocated GPU memory was 4.14 GB
(not total device usage). All 99,880,992 saved trainable parameters are FP32;
fresh-process physical probe predictions match exactly (maximum difference zero).
The saved expert tensor SHA256 is
`ad9214130d4e3fbc4ba7f029e5e3f9bdfcbd2196c7edc1495410391f64cb46c6`.

| Condition | Prefix | Commands | Pickup / hold | Insertion |
| --- | ---: | ---: | --- | --- |
| Expert command replay | 5 | 1,673 | Success; qpos/qvel errors exactly zero | Yes |
| Original VLA, fresh baseline | 5 | 336 | No grasp; finger self-contact | No |
| Corrected VLA | 5 | 546 | No grasp; finger self-contact | No |
| Corrected VLA | 25 | 2,000 | Grasp at 588; zero drops; still held at limit | No |
| Corrected VLA | 50 | 1,976 | Grasp at 660; three recorded drops; slip abort | No |
| Original VLA, matched control | 25 | 928 | No grasp; finger self-contact | No |

All policy trials used live RGB and positions, not recorded actor inputs.
The prefix-25 final lateral errors are -13.41 mm / -9.34 mm and orientation
error 8.33 degrees. This is a useful pickup/hold result on **one training reset**,
not a solved insertion task, held-out evaluation or real-time qualification.
The video's header label `expert` is the saved action-expert bundle directory,
not an expert controller: the title/report mode is SmolVLA closed-loop simulation.

On fixed training probes, perturbed open-hold gripper MAE falls
2.156 -> 1.057 mm and sensor gain 1.078 -> 0.522. Nominal joint MAE falls
0.006084 -> 0.004998 rad, but nominal gripper MAE does **not** improve
(0.692 -> 0.713 mm); nominal open-hold error rises 0.242 -> 0.521 mm. Do not
describe these subsets as full validation or uniform prediction improvement.

The prefix-25 original-weight control completed in 2.53 minutes under the same
metric extension, reset, seed and unfiltered interface. It never contacted the
plug and aborted at 928 commands on finger self-contact. Thus the pickup/hold
improvement survives a same-prefix original-weight comparison; it is not merely
the comparison of an old prefix-five run with a new prefix-25 run. This still
does not isolate augmentation from corrected optimizer storage or establish
held-out reliability. No candidate is promoted for general use or RL.

The combined supervised repair/control jobs took 43.49 minutes, within the
one-hour allowance. Both jobs are finished. The independent report audits all
800 update rows, parent/exposure isolation, saved FP32 storage and weight hash,
source/reset/seed/model/metric/filter bindings, and all six videos' full frame
counts (commands + 1). Forty-six selected regression tests pass.

Reviewed evidence:

- [Comparison plot](../results/gripper_control_fix_20261019/precision_repair.png),
  [PDF](../results/gripper_control_fix_20261019/precision_repair.pdf),
  [summary](../results/gripper_control_fix_20261019/summary.json),
  [conditions CSV](../results/gripper_control_fix_20261019/conditions.csv).
- [Repaired prefix-25 policy video](../outputs/gripper_control_fix_20261019/precision_repair/repaired_prefix25/rollout.mp4).
- [Original prefix-25 policy video](../outputs/gripper_control_fix_20261019/prefix25_control/rollout/rollout.mp4).

Reproduce verification and plotting (PyAV and matplotlib are in separate local
Python environments):

```powershell
.venv-smolvla\Scripts\python.exe scripts/report_precision_repair.py --experiment outputs/gripper_control_fix_20261019 --matched-control outputs/gripper_control_fix_20261019/prefix25_control --output results/gripper_control_fix_20261019 --verify-only
python scripts/report_precision_repair.py --experiment outputs/gripper_control_fix_20261019 --matched-control outputs/gripper_control_fix_20261019/prefix25_control --output results/gripper_control_fix_20261019 --plot-only
```

Verification deliberately refuses an existing output directory; use a new
`--output` when repeating the audit. Historical experiment reports are not
overwritten. Next gates are recorded in [plan.md](../plan.md): qualify corrected
GPU resume/memory, test a fixed reset bank, then correctly stored multi-parent
SFT before targeted alignment recovery or RL.
