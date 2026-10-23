# Gripper feedback: controlled evidence and next experiment

2026-10-13, checkpoint `step_0009426_best`. No weights, data, physics or guards
were modified. This extends [ROLLOUT_DIAGNOSIS.md](ROLLOUT_DIAGNOSIS.md).

## Controlled evidence

Rerendered expert/policy configurations at commands 0, 100, 200, 300 and 320,
on three held-out resets. Initial images match recordings within 6.7e-7 mean
absolute difference on a 0–1 scale. Seven input conditions use matched noise,
with three sampling seeds each: 315 predictions in total. The following means
are at command 320 across the three resets and three seeds:

| Input intervention | First predicted finger target |
| --- | --- |
| Expert state + expert images | 34.61 mm |
| Policy state + expert images | 2.12 mm |
| Expert state + policy images | 34.39 mm |
| Policy state + policy images | 1.95 mm |
| Change only right-arm joint positions | 34.05 mm |
| Change only measured right gripper position | 3.01 mm |
| Policy inputs, but restore expert gripper position | 33.88 mm |

This isolates gripper-position conditioning as the dominant contributor to the
closing prediction near failure in these cases. It does not establish that images
are irrelevant elsewhere, or that fixing the gripper alone fixes the arm path.
Hybrid observations are diagnostic counterfactuals, not deployable controllers.

The simulator rate limits targets to 0.01 rad per joint and 0.4 mm per finger per
50 Hz command; 195–206 of the 322 commands were modified. That establishes frequent
rate limiting, but does not prove it caused the learned feedback failure.

A matched `train_20004` rollout with prefix 25 (instead of 5) executes all 2,000
commands, 40 simulated seconds, without a safety abort. It never grasps or inserts;
finger travel still reaches 2.08 mm. This prefix change alone is not a task fix.
Only one reset and one alternative prefix were tested.

Logs: `outputs/v2c30_failure_diagnosis_20261013/input_swaps_v2/report.json` and
`prefix25/report.json`. Figure: `results/v2c30_input_ablation_20261013`.

## Recommended next experiment

1. Build a held-out gripper-state sensitivity benchmark. On the existing expert
   frames, perturb observed finger travel by -3, -2, -1, +1, +2 and +3 mm while
   retaining expert targets. Measure gripper errors separately from joint errors.
   These are observation-noise tests, not physical recovery demonstrations.
2. Prepare a small SFT candidate from step 9,426 with gripper-state observation
   noise on a minority of training batches: start at 20% of batches, uniform
   +/-2 mm, applied to physical state index 15 before normalization. Keep clean
   batches, original targets, train-only statistics and terminal masks; preserve
   noise RNG in checkpoints. This is a hypothesis test, not a proven correction.
   Use a bounded pilot and compare it before another long training run.
3. Collector owner: start with 10–20 additional corrective recovery episodes.
   During survey/transit/approach, briefly disturb opening by 2–5 mm and let the
   scripted expert restore the intended opening before closing around the plug.
   Record real state/images, corrective applied commands, disturbance seeds and
   outcomes. Do not disturb an already-held plug in this first test. Preserve
   episode-level holdout; failed disturbances remain explicitly labeled diagnostic
   data. Genuine recovery supervision is stronger than observation jitter alone.
4. Compare baseline/candidate on clean frames, noisy-state frames and the same
   three closed-loop resets, with prefix 5 and unchanged collision guards. Require
   reduced perturbed-state gripper error without worse clean-frame behavior.
   Require actual plug contact and grasp establishment to claim control progress;
   avoiding a collision alone is insufficient. Expand seeds and untouched resets
   before claiming generalization or proceeding to RL.

No candidate training or new collection was launched during diagnosis. The next
implementation is the sensitivity benchmark and bounded SFT candidate. Arm-path
drift remains a separate issue to evaluate even after gripper behavior improves.
