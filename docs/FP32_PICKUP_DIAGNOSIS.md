# Why the corrected multi-parent VLA fails pickup

2026-10-20. Diagnostic only: no optimization, model/processor edits, controller
changes, source-scene changes, relaxed guards or physical robot commands.
The final checkpoint remains SHA-256
`9a6341162b9df3e8735a1e15d8bbde4acef4e11c79955d9a3b4a80cdfe5316f3`.
The completed original live validation result stays **0/5**, not superseded by
assisted-input results below.

## Findings

### 1. Finger-state errors are amplified instead of corrected

At command 300 on held-out `train_20004`, keep the exact same recorded images,
instruction, other 15 state values and explicit latent noise. Change **only**
the right-finger state:

| Input condition | Finger state | First predicted finger target |
| --- | ---: | ---: |
| All recorded expert state | 34.99 mm | 35.26 mm |
| Only finger replaced with actual failed-rollout state | 25.14 mm | 19.37 mm |

The expert target is 35 mm in both comparisons. The model responds to a partly
closed hand by closing **further**, even with unchanged correct images. This
confirms a state-channel effect within the model, not a difference in rendering
or sampled noise. Two fixed noise samples and 11 anchors on each of two resets
produce 44 three-way comparisons; the third variant replaces all numeric state.
These intentionally counterfactual combinations diagnose sensitivity; they are
not physical trials, recovery demonstrations or autonomous success evidence.

The same mechanism appears on the training reset: at command 200, substituting
only the 31.82 mm live finger state for the recorded 34.99 mm changes the first
target from 35.21 to 31.10 mm. At command 300, replacing it with 12.92 mm changes
the target from 34.87 to 5.74 mm. Other-state drift additionally affects arm
commands; the result is not exclusively a gripper issue.

### 2. Nominal training never covers these pre-grasp states

All **81 training NPZ hashes** were checked against the export receipt.
Across **41,594 command-aligned observations before the first close phase**,
the right finger ranges from **34.890 to 43.989 mm**. There are **zero** examples
below 34 mm, and zero below 30 mm. The failed policy reaches 25.14 mm while it
is still supposed to be approaching with a 35 mm target.

Inference: nominal expert trajectories provide little direct supervision for
reopening/re-aligning after these deviations. The measured non-corrective
state dependence and missing coverage support a closed-loop recovery-data
problem. They do not prove that dataset size, architecture or another software
component can never matter. More nominal trajectories need not fill this gap.

### 3. Small nominal arm errors can still cause unsafe descent

Predict complete 50-command chunks from recorded expert inputs at prefix 25,
then execute only each first 25 commands through the unchanged environment.
No expert action is supplied to the actor and no execution state is overwritten.
This is externally guided VLA-command playback, **not autonomous operation**.
Fresh expert command replay qualifies the headless environment on each reset,
with exactly zero qpos error and the same physical-contact-settle extension.

| Reset | Original live VLA | Recorded-input VLA command playback | Fresh expert |
| --- | --- | --- | --- |
| `train_20002` (training control) | No grasp, 1,200 commands | Physical pickup at 734; first grasp 576; 64.40 mm max lift and 1 s carry | Pickup at 739 |
| `train_20004` (validation) | No grasp, 1,200 commands | Table-contact abort at 479, before expert close begins at 523 | Pickup at 779 |

On the validation abort, the assisted hand TCP differs from the expert by
**8.72 mm** at the same command index: approximately -7.42 mm x, -0.73 mm y,
**-4.52 mm z**. Actual finger travel is still 34.88 mm, so this is not premature
closure. Recorded support contact is **44.69 N**, exceeding the unchanged
30 N guard. Do not weaken that guard to make the trial pass.

Recorded-input pre-close joint-target MAE is only 0.00743 rad on this reset
(mean over joints/commands), yet the local pose error is physically important.
One training reset succeeds with recorded inputs, but one validation reset does
not: restoring nominal observations is not by itself a complete solution.

## Pipeline checks

- Saved pre/post action normalization is identical. Saved versus export mean/std
  differs only at FP32 rounding scale (maximum about 4.94e-8).
- Action normalize/denormalize roundtrip error is at most 8.95e-8 in physical
  output units, not evidence of a meters-versus-millimeters mistake.
- Ten sampled raw/export comparisons have identical normalized state/action
  values and instruction tokens, including close-phase anchors. RGB is not
  bit-identical: mean absolute differences are about 0.007–0.018 on [0,1],
  consistent with the different encoded/raw image paths. This alone neither
  proves a visual-data bug nor proves encoding differences irrelevant.
- Training and live environment instructions both read
  `Insert the plug into the socket.` No prompt mismatch was found.
- Full recorded-input versus historical live rollouts use the same configured
  seed, not a demonstrated bit-exact paired latent stream: initial whole-chunk
  differences are nonzero. Do not treat those full-trajectory comparisons as a
  perfectly controlled state-only experiment. The separate explicit-noise,
  identical-RGB counterfactual above supplies that narrower causal control.
- FK measurements use a separate `MjData` from saved qpos. They do not teleport
  the simulator during physical execution. TCP errors at matching command index
  can include phase/speed drift; they are not all distances to the grasp goal.

These checks found no obvious normalization/action-index/prompt boundary defect
in the tested samples. They are not a claim that the entire training stack is
bug-free. Numeric training stability and task reliability remain separate gates.

## Most useful next experiment (not started here)

1. Collect a **small audited recovery pilot from training resets**, including
   the prematurely closed states actually visited before grasp. Use an expert
   that reopens toward 35 mm and re-plans from the actual displaced arm pose.
   Capture synchronized actual RGB/state and actual corrective commands. Do
   not train on the diagnostic's artificially mixed state/recorded images.
2. Include observed near-table approach errors of roughly 5–10 mm, corrected
   through a safe hover/re-alignment/descent trajectory with all guards intact.
   Ground-truth TCP/plug/contact information may diagnose/drive the simulation
   expert, but must not silently become privileged VLA inputs.
3. Preserve the original validation parents; do not convert `train_20004` or
   other reserved resets into training recovery data. Start from known training
   parents, then reserve separate recovery validation/test cases.
4. Run a small, bounded, phase-balanced correction experiment with FP32 training
   and the unchanged actor contract. Require live training-reset pickup, then
   the frozen validation bank, before another long nominal SFT run. Phase labels
   may guide sampling/loss diagnostics, not actor commands or privileged inputs.

This targets both measured problems: non-corrective finger/state feedback and
fragile approach precision. A bigger batch/model, longer training or RL is not
the evidence-backed first intervention. No collection/training/fix was launched
by this diagnostic request; implementation requires the next user instruction.

## Evidence and reproducibility

- Raw diagnostic: `outputs/fp32_pickup_diagnosis_20261020`.
- Main prediction/playback run: 297.83 seconds, within its 900-second watchdog.
  State-only probe: 63.69 seconds, within its 300-second watchdog. Full train
  coverage hash audit: 3.69 seconds, within its 120-second allowance.
- 31 selected regression tests pass, including prefix/terminal alignment and
  physical-unit/error-summary tests. No GPU training/evaluation job remains.
- `results/fp32_pickup_diagnosis_20261020/recorded_vs_live.png` and PDF: opening,
  arm-target and same-index TCP drift; CSVs retain full numeric comparisons.
- `.../state_sensitivity.png` and PDF: explicit same-noise state-only intervention.
- `.../diagnosis_summary.json`: bound diagnostic, sensitivity, coverage and
  alignment evidence. Sources are hash-archived; original data/models/live
  evaluation artifacts are preserved.
