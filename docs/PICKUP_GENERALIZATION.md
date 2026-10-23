# Pickup generalization gate

## Completed pilot (2026-10-20)

**Pickup generalization is not established.** The frozen VLA passes **1/5
held-out validation resets (20%)**, while source expert replay passes **5/5**.
The separate memorized-reset expert and VLA controls both pass and are excluded
from the held-out denominator. All twelve conditions are accounted for; no
training or evaluation job remains active.

| Held-out parent | Expert pickup | VLA pickup | VLA commands / stop |
| --- | --- | --- | --- |
| `train_20004` | Pass | Fail | 1,200; command limit, no pickup |
| `train_21009` | Pass | Pass | 655; pickup gate reached |
| `train_21022` | Pass | Fail | 685; guarded robot collision |
| `train_22000` | Pass | Fail | 1,200; command limit, no pickup |
| `train_22042` | Pass | Fail | 1,200; command limit, no pickup |

The successful held-out policy trial lifts the actual plug 50.13 mm and meets
the continuous one-second carry condition, with no drops. The training-reset
VLA control lifts 64.86 mm. None of the four failed validation trials reaches a
valid pickup. This is not an insertion failure being counted against pickup.
The descriptive reset-level Wilson 95% interval for 1/5 is **3.6--62.4%**; do not
present 20% as a precise deployment success probability or a blind-test score.

The independent audit recomputes pickup outcomes from qpos/contact/time traces,
checks source/split/reset/model/metric/filter bindings and continuation/recovery
provenance, and decodes **10,227 video frames** across all twelve conditions.
Fifty-six selected unit/regression tests pass. Cumulative active evaluation and
artifact recovery time is **32.29 minutes**, within the original 45-minute
allowance; the calendar wait for cleanup approval is not counted as compute time.

One failed trial (`train_22000_policy`) completed all 1,200 physical commands but
its recorder was killed during H.264 finalization. Its original trajectory,
chunks, diagnostics, GIF and incomplete MP4 are preserved. A separately labeled
`rollout_recovered.mp4` renders those saved states **post-hoc**, with no policy
rerun, additional control commands or outcome replacement. Its terminal summary
is explicitly reconstructed; unavailable original wall-time/final-contact detail
is not invented. This is the only reconstructed video. Its recovery time is
debited to the same cumulative budget. Later recording can receive at most 120
seconds of artifact-only grace **only after a complete terminal motion trace
exists**, never extra policy execution; the total 45-minute cap remains.

Reviewed evidence:

- [Pickup plot](../results/pickup_generalization_20261019/pickup_generalization.png),
  [PDF](../results/pickup_generalization_20261019/pickup_generalization.pdf),
  [audited summary](../results/pickup_generalization_20261019/summary.json),
  [conditions CSV](../results/pickup_generalization_20261019/conditions.csv).
- [Successful held-out VLA video](../outputs/pickup_generalization_20261019/train_21009_policy/rollout.mp4)
  and [its final frame](../results/pickup_generalization_20261019/train_21009_policy_last.png).
- [Failed held-out VLA video](../outputs/pickup_generalization_20261019/train_20004_policy/rollout.mp4).

## Evidence-driven next move

Keep insertion and RL deferred. The corrected checkpoint was fitted for 800
updates on **one** training episode; this pilot does not test a corrected
multi-parent training recipe. First qualify FP32 GPU optimizer/RNG exact resume
and current memory/batch headroom, with sufficient disk for full checkpoints.
Then run a controlled FP32 SFT baseline on the **existing 81 nominal training
parents**, with a new optimizer, unchanged saved feature contract/normalization
provenance and unchanged validation parents. Do not resume the legacy BF16
optimizer or expand collection as the first response to this result.

Use the same fixed pickup bank to measure progress, with pickup/contact/lift and
clean joint/gripper retention metrics alongside loss. More updates on the same
single example would not test pose coverage. A passing validation pilot still
needs the full validation bank and a fresh disjoint tabletop test bank before
broader claims. No new long training campaign is launched by this evaluation.

## Frozen protocol and historical execution notes

Priority: establish pickup beyond the memorized reset before tuning insertion
or beginning RL. The previous result is one successful training-reset pickup,
not evidence of generalization. Freeze the corrected FP32 expert and execution
prefix 25 for this pilot; do not retrain, select a new checkpoint or sweep prefixes
on these outcomes.

## Fixed validation pilot

- Positive control: `train_20002`, the single parent used for the corrected
  800-update diagnostic. Report it separately, never in the held-out denominator.
- Five validation parents: `train_20004`, `train_21009`, `train_21022`,
  `train_22000`, `train_22042`. These are indices 0/5/10/14/19 of the sorted
  original 20-parent validation split, chosen without consulting outcomes.
  The `train_` prefix is a collection identifier; split membership comes from
  the frozen export receipt, where these five parents are validation-only.
- Each reset uses the recorded plug/socket poses and visual randomization, the
  source-hash-matched collector scene and unchanged collision/force guards.
  First run source expert command replay, then the frozen VLA with live measured
  positions and both RGB images. Expert replay is a separate control, never actor
  input assistance. No simulator state is forced after reset.
- One fixed policy-noise seed (1000), prefix 25, at most 1,200 commands per trial.
  Stop early once pickup passes; no insertion-completion requirement.
- Pickup requires the actual free plug to rise at least 20 mm above its initial
  body-origin height, physical grasp registration and both finger contacts,
  continuously for one simulated second, with no prior drops or safety abort.
  Motion during that interval is allowed: this is lift/carry, not a scripted
  stationary hold. Evaluation does not generate a hold command or modify action
  targets, contact thresholds, forces or object motion.
- Count approach collisions, any grasp, lift height, continuous carry duration,
  drops and final outcome separately. A grasp flag alone is not pickup success.
  Recompute success from saved qpos/contact/time traces and fully decode videos.
- Pilot screen: five of five held-out pickups, all expert controls qualified and
  no pre-pickup safety aborts. A smaller result does not establish this gate.
  Do not replace failing parents; retain every outcome. If an expert control
  fails, mark benchmark validity separately rather than silently excluding it.

`scripts/evaluate_pickup_generalization.py` saves the complete frozen plan and
source/model hashes before executing. It caps each subprocess at five minutes
and the full pilot at 45 minutes, retains a 25 GiB disk reserve and owns its
process-scoped sleep guard/child cleanup. A failed positive control or runtime
failure produces a partial failure report, not a fabricated overall result.

```powershell
.venv-smolvla\Scripts\python.exe scripts/evaluate_pickup_generalization.py --output outputs/pickup_generalization_20261019 --prepare-only
.venv-smolvla\Scripts\python.exe -u scripts/evaluate_pickup_generalization.py --output outputs/pickup_generalization_20261019
```

Use a new output directory for a repeat; never overwrite the frozen protocol.
There are no optimizer updates, checkpoint selection or network/API-token calls.

## What the result can and cannot show

The 20 validation parents were not optimizer-training parents, but they influenced
the original best-checkpoint selection and earlier diagnostics. Therefore this
is **same-scene held-out validation generalization**, not an untouched blind
test. Five resets and one sampling seed are a small pilot. Even five successes
give a reset-level Wilson 95% interval of roughly 57--100%; video frames or action
commands must not be counted as independent trials.

After a passing pilot, run the complete original validation bank with fixed
sampling seeds and report per-reset outcomes. Additional noise seeds are repeats
on the same resets, not additional independent scene samples. Before a broader
generalization claim, reserve a fresh, source-matched **tabletop** test bank
(initially at least 20 new plug poses/yaws and visual conditions), disjoint from
all training/validation/recovery parents and reset seeds. Collect its reference
controls separately; never export it for training or use it to tune prefixes.
The old 100-condition insertion-only reserved bank starts with a held plug and
is not a pickup generalization test.

If this pilot fails while expert controls pass, retain insertion as a secondary
problem: first requalify FP32 optimizer resume/memory, then correct multi-parent
SFT and test nominal pickup coverage/retention. One-parent overfitting or more
batch size alone does not establish pickup coverage. Do not begin RL or order
a large unstructured collection based solely on the failed pilot.

Fresh geometry, camera mounts, robot dynamics and real hardware each require
separate domain-transfer/timing tests; this pilot cannot establish sim-to-real.

Status at the initial resource stop: six pickup-specific unit tests and 32 selected compatibility/regression
tests pass. The criterion also passes post-hoc on the preserved expert and VLA
positive-control trajectories (commands 739 and 697 respectively). The fresh
expert positive control passes at 739 with exact physical replay. The frozen
pilot has stopped safely at the disk guard; no overall validation-generalization
outcome is claimed yet.
The fresh VLA positive control also passes, at command 729. The first held-out
reset (`train_20004`) has passed expert control but failed policy pickup: 1,200
commands, zero plug/finger contacts, no lift. The second (`train_21009`) passes
policy pickup at command 655; its expert control passes at 758. The third
expert control passes at 745. Seven of twelve stages are preserved. The job
stopped after 936.047 seconds at its time/disk pre-stage check; the 45-minute
allowance was not exhausted, and current free disk is below the 25 GiB reserve.
No evaluation or training process remains active. One success in two completed
validation policy trials is partial evidence, not a five-reset result or a
passing generalization gate.

`--resume` rechecks the original plan/model/actor/metric hashes and every saved
condition/video, preserves all completed outcomes and continues only the ordered
unfinished stages. Original `plan.json` and `failure.json` remain immutable;
new supervisor code is separately hash-archived and declared in a continuation
manifest. Prior elapsed time is debited from the same 2,700-second allowance.
Three resume-specific regression tests pass; changed models/prefixes/actor code
or skipped/replaced completed conditions are rejected. Disk headroom must be
restored before launching.

With explicit user approval, only
`outputs/fullwindow_action_probe_preflight_v2/checkpoints/step_0000012_best`
was permanently removed: 1,610,577,735 bytes (1.50 GiB). The retained
`step_0000012_final/pretrained_model/model.safetensors` has identical weights,
SHA256 `c7758e6686a55849e3e2d522606b72e5fbe8f16c4d6d6d51b61facee8460058b`.
The removed checkpoint's optimizer/metadata files are not claimed identical to
the retained final checkpoint. Logs, current VLA models and evaluation artifacts
remain. The deletion bypassed the Recycle Bin; historical references to that
old preflight's best-checkpoint path now refer to a removed bundle.

After cleanup, free space was 25.54 GiB and continuation 001 launched. All seven
saved stages re-audited successfully; evaluation resumed at `train_21022_policy`
with 936.047 seconds debited from the original 2,700-second allowance. The same
model, actor/metric code, reset bank, prefix and sampling seed remain frozen.
No further deletion is authorized.

```powershell
.venv-smolvla\Scripts\python.exe -u scripts/evaluate_pickup_generalization.py --output outputs/pickup_generalization_20261019 --resume
```

After completion, independently audit and render the results:

```powershell
.venv-smolvla\Scripts\python.exe scripts/report_pickup_generalization.py --evaluation outputs/pickup_generalization_20261019 --output results/pickup_generalization_20261019 --verify-only
python scripts/report_pickup_generalization.py --evaluation outputs/pickup_generalization_20261019 --output results/pickup_generalization_20261019 --plot-only
```

The audit deliberately requires all 12 conditions and refuses an existing output
directory; a partial failed job must not masquerade as a completed pilot.
