# Automated randomized collection: work log

Work log for fixing [scripts/collect_random.py](../scripts/collect_random.py),
the automated randomized table-top pick-and-insert collector. This file records
what was done, why, and the evidence. Results are filled in only after
independent verification. Anything not yet verified is marked **TODO**.

## 2026-09-23

### 1. Goal

`collect_random.py` had **0% success**. The robot could not even pick up the
plug, so no randomized full-task demonstration was recorded.

Approach: agents run in a **diagnose → fix → verify** loop. An orchestrator
reproduces the failure, diagnosis agents find causes, fixer agents change code,
and a separate agent verifies the merged result. If verification fails, the loop
goes back to the fix stage.

### Summary / current state

- `scripts/collect_random.py` works end to end (pick -> lift -> transport ->
  insert -> hold) for randomized tabletop scenes: about 95% of attempts
  succeed on fresh scenes, with 30-45 s episodes, on both detectors. Failed
  attempts are retained in `failed_preflight/` and never labelled as BC data,
  so every saved episode is a genuine success. Recorded episodes replay
  exactly (qpos/qvel error 0.0).
- Known limitation: about 5% of scenes (for example seeds 1000009, 1000113,
  1000116) lose the grasp during transport regardless of speed. Root cause
  not determined; accepted by the user as a known limitation (see "Remaining
  issues").
- The first run after any change to the sampler or controller code pays a
  one-time scene-cache warm-up of about 7 minutes; afterwards it drops to
  about 1.3 minutes.
- Nothing described in this log has been committed yet. Leftover agent git
  worktrees under `.claude/worktrees/` are awaiting cleanup.
- All three loop iterations are finished; see "Loop iterations" below for the
  full history and "Remaining issues" for what is still open.

### 2. Process and timeline

| Stage | Agent role | Parallel? | Status |
| --- | --- | --- | --- |
| 0 | Orchestrator: reproduce the failure | - | Done |
| 1a | Diagnose the controller side (`PickInsertExpert`) | Yes, with 1b | Done |
| 1b | Compare with the validated pipeline and scene sampling | Yes, with 1a | Done |
| 2a | Fix pickup + lift | Yes, with 2b | Done (merged) |
| 2b | Fix transport + insertion (separate git worktree, merged later) | Yes, with 2a | Done (merged) |
| 2c | Merge fixer 1 + fixer 2 into the main checkout | - | Done |
| Checkpoint 1 → 2d | scene_bank feasibility fix (user-approved) | - | Done |
| 3 | Full verification (20 scenes privileged + 20 noisy, record + replay) | Needs 2a + 2b merged | Done |
| - | Loop iteration 1: fixer A (transport grasp loss, `place_insert.py`, in a worktree) ∥ fixer B (noisy close failure + runtime insert-consistent grasp + dead-code cleanup, `pick_insert_expert.py`) | Yes | Done: fixer B merged and independently verified; fixer A's fix not merged (still failing) |
| - | Loop iteration 2: fixer A2 investigating the row-9 transport grasp loss, tail-slowed carry fix merged | - | Done: target not met on fresh seeds (row 113 grasp loss) |
| - | Loop iteration 3: fixer C (scene_bank noise-buffered feasibility margin, merged) ∥ fixer D (root-cause the grasp slip, worktree, stopped at the user's request) | Yes | Done |
| - | Startup regression fix: on-disk scene cache + lazy collection-bank generator | - | Done |
| - | Speed sweep and merge (`TRANSIT_SPEED_RAD_S = 0.15`) | - | Done |
| - | This log | Written alongside the work | Done (final) |

Stage 2b runs in its own git worktree so the two fixers do not edit the same
files at the same time. Its changes will be merged into the 2a branch before
stage 3. If stage 3 fails, the work returns to stage 2 with the failure evidence.

**Note:** from 2026-09-24, all subagents run on the Sonnet model at the user's
request.

**Note:** from now on, no bug fix is applied without the user's confirmation.
Each confirmation explains the likely cause, the next step, and why.

**Note:** on 2026-09-24, the user switched to **auto mode**: loop test → fix →
test until success, without per-fix confirmation, recording every iteration in
this log. Subagents still run on Sonnet. The success target is 20/20 accepted
with zero failed attempts for both the privileged and noisy detectors on a
fresh seed range, plus unit tests, the validated scripts and strict replay
passing.

### 3. Reproduction

```powershell
python scripts/collect_random.py --episodes 3 --early-abort 4 --headless --video none --detector privileged --output <scratch>
```

Result: **4/4 failed preflights**.

| Episode | Failure | Phase durations |
| --- | --- | --- |
| train_00000, train_00002, train_00003 | `expert_descend_timeout` | survey 0.9 s, transit_pick ~1.3 s, pregrasp ~1.9 s, descend 4.0 s (timeout) |
| train_00001 | `expert_pregrasp_timeout` | pregrasp 6 s (timeout) |

No table contact was recorded; the arm never reached the plug. The older
`data/openarm_v2_random/failed_preflight/` episodes fail the same way.

### 4. Diagnosis (why it failed)

#### 4.0 The validated path still works

- [scripts/validate_table_pickup.py](../scripts/validate_table_pickup.py): **8/8 PASS**
  (height gain 0.0736 m, drift ~0.13 mm, 2 finger contacts).
- [scripts/probe_table_insert.py](../scripts/probe_table_insert.py): nominal
  **PASS** (0.26 s hold, 16.0 mm depth).

So physics and the env are fine. The failure is in `PickInsertExpert` and the
scene distribution. Caveat: `--lower-fixture` crashes because the fixture geom
is commented out at [envs/scene.py:39](../envs/scene.py) (commit `bc15440`).

#### 4.1 Cause 1: descend is too slow

- SETTLE gain 0.15 × `pos_clip` 0.6 mm at 50 Hz ≈ **4.5 mm/s**
  ([controllers/pick_insert_expert.py:51](../controllers/pick_insert_expert.py)).
- `APPROACH_HEIGHT_M` = 90 mm (`:45`); descend timeout 4 s (`:53`).
- 90 mm at 4.5 mm/s needs ~20 s, so `descend_timeout` is certain.
- Measured 4.3 mm/s. With a 40 s timeout, descend took 23.7 s and reached `close`.

#### 4.2 Cause 2: position-only phase gates; servo trapped at joint limits

- `transit_pick` (`:228`), `pregrasp` (`:237`) and `descend` (`:246`) exit on
  position error only. `transit_pick` exits 76–94° off in orientation, and
  descend starts 27–51° off.
- One long-timeout run descended 51° tilted and ended with `table_collision`
  (30.1 N).
- `_servo` (`:185-201`) integrates a damped-least-squares step onto the joint
  target and ignores joint limits. The env clips at the limits
  ([envs/openarm_insert.py:418](../envs/openarm_insert.py)), so targets stay
  pinned (joint2 −10°, joint7 −90°). In scene 1, orientation error fell
  66° → 0.8° while position error grew 17 → 74 mm.
- Only one of the two symmetric gripper yaws is allowed
  (`GRASP_IN_PLUG_EULER_DEG`, `:41`). The 180°-rotated yaw is often feasible.

#### 4.3 Cause 3: scene sampling does not match top-down reach

- A top-down grasp puts joint7 near its limit even at the validated nominal
  (margin 0.001 rad).
- IK is feasible roughly for x ≤ 0.38 and y in [−0.30, −0.22] at plug yaw 0.
  x ≥ 0.42 is unreachable (42–85 mm short at x = 0.45).
- [data_pipeline/scene_bank.py:43-48](../data_pipeline/scene_bank.py) samples
  x 0.30–0.45, y −0.30–0.0, with plug yaw fixed at 0.
- The reach check measured from the base (0, 0, 0) instead of the right
  shoulder (0, −0.031, 0.698). `collect_random.py` never called
  `workspace_for_env`.
- `plug_half_height_m` is 0.012; the real value is 0.008.

#### 4.4 Cause 4: grasp target too low

- The target is the plug centre with a −0.010 m offset. The finger collision mesh
  extends 10.4 mm past `robot_grasp`, so the fingers reach **2.4 mm into the table**.
- The validated pickup uses `plug_grasp_frame` (−0.016 m) plus 3 mm in z
  ([controllers/table_pickup.py:95](../controllers/table_pickup.py)).

#### 4.5 Downstream bugs (hit once the first causes are bypassed)

| Bug | Evidence |
| --- | --- |
| `close` exits early | `grip_force_n` is the position-servo output (100 N/m × (command − actual)), not contact force. It reads 0.78 N at 0.4 s with fingers still open (`pick_insert_expert.py:252-259`, `:174-176`). Result: slip → `regrasp_budget_exhausted`. |
| Grip is weak | ~0.57 N when firm; 16.7 mm slip during approach. |
| Env tabletop success is impossible | Grasp slip/angle is measured against the side-grasp frame ([envs/openarm_insert.py:349-352](../envs/openarm_insert.py)); a top-down grasp reads 7.2 mm / 90°. |
| Socket pose is not physical | Sampled at table height and upside down (`socket_tilt_deg` 180), unlike the validated elevated, 90°-rolled socket. The multistart IK audit found no solutions for the original socket orientation. |

#### 4.6 Git history

- `bc15440` added `PickInsertExpert` and `scene_bank`, and removed the fixture.
- `9f873d0` switched to a top-down grasp and moved the sampling ranges.
- `e419d0c` fixed plug yaw at 0.
- None of them touched `table_pickup`, `carry_path` or physics.

### 5. Fix plan

**Fixer 1 (stage 2a): pickup + lift.** Rebuild the pickup on the validated recipe:

- Robust IK: restarts, joint-limit margin, contact check, both symmetric yaws.
- Smooth joint-space interpolation instead of the Cartesian servo.
- Correct grasp height (validated grasp frame).
- Close until the fingers settle and contacts exist; stronger grip.
- Fail fast with `ik_infeasible` instead of timing out.
- Tabletop-mode grasp latch in the env; feasible plug sampling.
- Acceptance: **≥90% lift on ≥20 scenes**.

**Fixer 2 (stage 2b): transport + insertion.**

- Physically sensible socket distribution: elevated/supported, orientation
  compatible with the grasp, feasibility rejection.
- Post-lift phases use `plan_carry`, smooth joint execution and a slow
  straight-line insertion.
- Acceptance: **≥80% success on ≥15 scenes**.

**Constraints for both fixers:**

- 8-D action contract unchanged.
- Free plug: no weld, no teleport.
- Legacy reset mode unchanged.
- Provenance and source-hash rules followed.
- Validated [controllers/table_pickup.py](../controllers/table_pickup.py) and
  `carry_path` left untouched.

### 6. Results

> Filled in as fixer results, merges and verification completed. See "Summary
> / current state" above for the final outcome.

#### Fix 1: pickup + lift

Changes in [controllers/pick_insert_expert.py](../controllers/pick_insert_expert.py),
[data_pipeline/scene_bank.py](../data_pipeline/scene_bank.py),
[envs/openarm_insert.py](../envs/openarm_insert.py) and
[scripts/collect_random.py](../scripts/collect_random.py):

- IK for both symmetric gripper yaws, with the best one chosen.
- Joint-space interpolation instead of the Cartesian servo.
- Corrected grasp height.
- Closing until the fingers settle and contacts exist, plus a firmer grip.
- Fail-fast reasons instead of timeouts.
- A tabletop-mode grasp latch in the env (hand-to-plug pose latched at grasp;
  accessor `latched_grasp()`).
- Plug sampling restricted to reachable poses.

Measured on 36 sampled scenes (collection bank rows 0-35) with a
stop-after-lift harness: **36/36 firm lifts with the privileged detector,
36/36 with the noisy detector.** Lift rise about 0.099 m, 2 finger pad
contacts, grasp drift mostly 0.05-0.1 mm (max 0.76 mm, noisy), no regrasps.
Grasp yaw options chosen per scene: -15 deg or -30 deg tool yaw.

Fixer 1's end-to-end run through `collect_random` with the OLD post-lift
phases then failed at approach (`approach_timeout`) and in regrasp
(`numerical_failure`). That was expected, because the post-lift phases are
fixer 2's scope.

#### Fix 2: transport + insertion

It worked in a separate git worktree. New file
[controllers/place_insert.py](../controllers/place_insert.py) with post-lift
phases `transit_place -> approach -> align -> insert -> hold`,
collision-checked carry with `plan_carry`, smooth joint-space execution, and
slow straight-line insertion. New tests
[tests/test_place_insert.py](../tests/test_place_insert.py). The socket
distribution in `scene_bank` was redesigned to put the socket elevated at
z ~= 0.335 with yaw about -10..+28 deg, instead of at table height and
upside down.

Measured on 30 scenes (seeds 1000000-1000029) with its own pick stand-in:
**24 successes**. 4 were `pick_standin_failed` (the stand-in's own pickup IK,
which the merge replaces with fixer 1's pickup). 2 were `expert_grasp_lost`:

- seed 1000009: during approach, drift 1.18 mm and 2.9 deg;
- seed 1000026: during transit_place, drift 1.71 mm and 2.5 deg.

The limit is 1 mm. So the rate is **24/26 = 92% where pickup succeeded**. In
successful runs: final depth 15.99-16.0 mm, hold 0.22-0.24 s, peak socket
force 0.85-1.81 N, 0 N robot contact and 0 N support contact, grasp drift
about 0.11-0.24 mm. The carry used a direct segment in most cases; RRT was
needed in 2 cases (seeds 1000009 and 1000025).

#### Merge

2026-09-24.

- [controllers/place_insert.py](../controllers/place_insert.py) and
  [tests/test_place_insert.py](../tests/test_place_insert.py) were copied from
  fixer 2.
- In [controllers/pick_insert_expert.py](../controllers/pick_insert_expert.py),
  fixer 1's pickup was kept wholesale. The old Cartesian-servo post-lift block
  was replaced by a hand-off to `PlaceInsert` (`_begin_place`/`_place_action`).
- In [data_pipeline/scene_bank.py](../data_pipeline/scene_bank.py), fixer 1's
  plug sampling and fixer 2's socket sampling (`sample_socket`,
  `InsertFeasibility` screen) were combined.
- The hand-off uses `env.latched_grasp()`. It is the same formula and the same
  `robot_grasp` frame as fixer 2's measured transform, so env success checks
  and the expert share one reference.
- Unused helpers from the old servo remain in `pick_insert_expert.py`
  (`_servo`, `_socket_frame`, and so on). Cleanup is pending.
- 30/30 unit tests pass.

#### Checkpoint 1: plug orientation mismatch (fixed with user approval)

Symptom: the merged smoke run failed every scene at `transit_place` with
`plan_seated_unreachable`, after a successful pick and lift.

Cause: fixer 1 sampled plug yaw 150-210 deg, a convention tuned for the old
upside-down socket. Fixer 2's upright socket faces -10..30 deg. A top-down
pinch cannot change the plug's yaw about world z. The socket's half-turn
symmetry is a roll about the horizontal mating axis, not a turn about world z,
so a plug 180 deg off cannot be inserted. The Workspace comment claiming
otherwise was wrong and has been corrected. `require_insert_feasible` was
False, so such scenes were still generated.

The user's decision: change the scene_bank config so impossible scenes can't
be generated, and investigate further.

Investigation: a kinematic sweep of 900 combinations (plug yaw -180..180 deg
in 15 deg steps x a 3x3 grid of plug x/y x 4 socket poses). Insert-consistent
fraction: about 0% beyond +-90 deg and 0-6% in the old 150-210 deg band, while
pick feasibility there was 33-72%. The peak is at 0-15 deg (72-75%).

| Plug yaw (deg) | Insert-consistent fraction |
| --- | --- |
| -45 | 0.22 |
| -30 | 0.33 |
| -15 | 0.56 |
| 0 | 0.72 |
| 15 | 0.75 |
| 30 | 0.61 |
| 45 | 0.25 |
| 60 | 0.17 |
| 75 | 0.08 |
| 90+ | ~0 |
| 150 | 0.06 |
| 180 | 0.00 |

Change: `plug_yaw_range_deg` (150, 210) -> (-30, 60), and
`require_insert_feasible` False -> True. No sampling logic changed.

Results:

- Sampler: 1.08 s per scene, 23.8 draws per scene on average (max 83, limit
  200), deterministic per seed.
- Banks are still disjoint and repeatable. 30/30 unit tests pass.
- Pickup benchmark: 12/12 firm lifts, all insert-consistent.
- End-to-end smoke: 3/3 successes, each on its first attempt (it was 0/2
  before).

#### Verification

2026-09-24.

| Check | Result |
| --- | --- |
| Unit tests | 30/30 |
| `validate_table_pickup.py` | 8/8 |
| `probe_table_insert.py` | pass (hold 0.26 s, peak socket force 0.84 N) |
| `collect_random` 20 eps, privileged | 20/20 accepted in 21 attempts (95.2%), 41.8 min |
| `collect_random` 20 eps, noisy | 20/20 accepted in 22 attempts (90.9%), 43.2 min |
| Record pass, 3 eps | 3/3 recorded, manifest `complete=true`, no `failed_attempts` |
| `replay_pilot` on the record pass | 3/3, max qpos/qvel error 0.0 (exact) |
| `validate_episode` | all pass |

Failures found:

- **A. `expert_grasp_lost` (`transit_place`):** 2/43 attempts, the same scene
  (seed 1000009, row 9) with both detectors. Drift 1.16 / 1.85 mm against the
  1 mm limit, angle 1-2 deg, grasp tilt -30 deg, direct joint segment (no RRT),
  joint margin 0.16 rad. In the last ~60 ms the actual right joint3 speed
  jumped from 0.03 to 1.1 rad/s, while the planned (commanded) per-joint speed
  is capped at about 0.3 rad/s by the synchronous `JointSegment`. The
  verifier's "near-singularity" explanation was rejected for that reason; the
  cause must be outside the plan (contact, torque saturation or slip). Under
  investigation in iteration 1.
- **B. `penetration_abort` (`close`):** 1/22, noisy detector only (row 1).
  Detector error 2.86 mm against about 1.5 mm finger clearance. One-sided
  finger contact, 7.8 N, 4.9 mm penetration, plug flung.
- Note: failed attempts were retained in `failed_preflight/` and never
  labelled as BC data, so collection was already usable at this point. The
  remaining work is about acceptance rate.

### Loop iterations

#### Iteration 1 (2026-09-24/25)

Two fixers ran in parallel:

- Both fixers were cut off by the API usage limit. Their work was preserved.
- **Fixer B** (controllers/pick_insert_expert.py):
  - It found that a noisy detector error (≈2.9 mm) exceeded the finger clearance, causing a one-sided pad contact while closing. The fix: if only one pad touches for a grace period, or penetration exceeds a threshold, the fingers back off and fold the offset into a robot-side grasp correction for the retry. The plug pose is never overwritten.
  - The runtime planner now REQUIRES an insert-consistent grasp and fails fast with `ik_infeasible:no_insert_consistent_grasp`.
  - The old Cartesian-servo code was removed. `grasp_pose_source` other than "top_down" now raises ValueError.
  - Verified in iteration 2 by an independent verifier:

| Check | Result |
|---|---|
| Unit tests | 30/30 |
| validate_table_pickup.py | 8/8 (output redirected to scratch, because the committed results/ evidence must not be overwritten) |
| probe_table_insert.py | pass |
| Pickup benchmark, privileged | 12/12 firm lifts, no pick failures |
| Pickup benchmark, noisy | 12/12; rows 1 and 8 recovered from one-sided contact on retry |
| Row 1, noisy | full success through hold, no penetration_abort |
| 10-scene noisy regression | 10/10 accepted in 11 attempts; the only failure was the known row-9 transport grasp loss |

  The back-off path never fired with the privileged detector, so privileged behaviour is unchanged.
- **Fixer A** (transport grasp loss, row 9):
  - Measured root-cause evidence: the commanded joint motion is smooth (joint3 command-to-actual error under 0.002 rad). At t≈28.5 s, near the end of a long direct carry, the actual right joint3 velocity jumped from 0.016 to −6.8 rad/s within one 20 ms command step. The actuator saturated, the finger–housing contacts vanished and then returned at 5 N, and drift went 0.09 → 1.7 → 4.5 mm.
  - No contact other than the fingers on the held plug was involved.
  - Halving or doubling the transit speed still triggered it at about the same path fraction.
  - Conclusion: a controller, physics or contact instability at a specific arm configuration while holding the plug, not a planning error.
  - Its attempted fix (splitting the carry into short legs with stops) still failed (numerical_failure, drift 2.85 mm), so it was NOT merged.
- The user ran `python scripts/collect_random.py --episodes 5 --detector privileged --output data/test_auto_collection` themselves: 5/5 accepted in 5 attempts, 444 s.

#### Iteration 2 (2026-09-25)

Fix B was verified (already described in Iteration 1).

Fixer A2 investigated the row-9 transport grasp loss.

- Measured: the static holding torque at the carry goal is only 9–15% of the
  actuator forcerange. A coupled 7×7 linear PD stability check (true mass
  matrix) was stable at every pose tried, including next to the crash.
  Holding still at the goal from a clean start is stable.
- Ruled out: changing the grip squeeze during carry, a quintic (C²) time-law
  (worse), uniformly halving or doubling the transit speed, and fine
  stop-and-go legs (worse).
- Fix applied in controllers/place_insert.py: `tail_slowed_legs` slows the
  final 25% of the carry's arc length by 10x (TAIL_FRACTION=0.25,
  TAIL_SLOWDOWN=10, TAIL_CHECKPOINT_S=0.3), and the post-lift descent is
  uniformly slowed (DESCENT_CARRY_SPEED_RAD_S=0.02, 5x slower). No physics or
  env change. The chunked-transit attempt was removed.
- Results: row 9 succeeded with both detectors; 10-scene regressions 10/10
  with 0 failed attempts, privileged and noisy; unit tests 30/30. Merged into
  the main checkout, with the previous place_insert.py backed up.

**Final verification on FRESH scenes (bank rows 100–119, never used for tuning):**

| Check | Result |
|---|---|
| Unit tests | 30/30 |
| Fresh, privileged | 19/20 (row 113: expert_grasp_lost in transit_place, 1.22 mm / 3.5°) |
| Fresh, noisy | 18/20 (row 113: grasp_lost 1.13 mm / 3.0°; row 105: expert_ik_infeasible, joint margin 0.003–0.048 < 0.05, detector error 3.8 mm) |
| Recorded 3 eps (noisy) + replay_pilot | 3/3, complete=true, qpos/qvel error 0.0, validate_episode pass |
| Episode length | mean 62.5 s (3123 steps), range 50.8–74.3 s; it was 33–37 s before the slowdown, so about 2x |

**Target NOT met.** Lessons:

1. The iteration-2 slowdown was over-fitted to one seed. Row 113 slipped
   inside the already-slowed tail, so speed is not the root cause, and the
   slowdown doubles episode length (a throughput and BC data-quality cost).
   Hypothesis for iteration 3: a marginal grip. Pad normal force is ≈0.78 N,
   μ=0.4, and the plug is 30 g, so the friction capacity is only ≈2× the
   weight. Pad stick-slip may shake the arm rather than the reverse.
2. Row 105: scene_bank certifies feasibility at the exact pose with the same
   margin floor as runtime, so marginal scenes fail under detector noise.

#### Iteration 3 (2026-09-25) — finished

- **Fixer C (done, merged)**,
  [data_pipeline/scene_bank.py](../data_pipeline/scene_bank.py): a
  noise-buffered pick feasibility check. `_detector_noise_bounds` reads the
  Detector's default noise kwargs (bias 1.5 mm + noise 0.8 mm position, 1.2°
  + 0.4° yaw, combined in quadrature) and uses 3σ bounds of ≈5.1 mm / 3.8°.
  `_pick_noise_margin_check` re-runs `PickFeasibility` at 12 perturbed poses
  ({plug, socket} × {x, y, yaw} × ±). Workspace flags: `pick_noise_margin=True`,
  `pick_noise_sigma=3.0`. Row 105's original draw is now rejected and the seed
  resamples to a robust scene. It also caught a latent case (seed 1000116,
  socket yaw). Yaw, not translation, is the dominant sensitivity. Result on
  fresh rows 100–119 with the noisy detector: zero `ik_infeasible`. Cost:
  scene generation went from 1.08 s to 5.25 s per scene.
- **Fixer D (stopped at the user's request)**: root-causing the transport
  grasp slip. Its partial finding: a firmer grip (finger target travel 0.003
  or 0.006 m) did NOT prevent the slip on row 113 (drift 1.12–1.22 mm). The
  user decided that roughly 5% failures are acceptable and to skip this task.
- **Slowdown removed (user request)**:
  [controllers/place_insert.py](../controllers/place_insert.py) was reverted
  to the pre-iteration-2 speeds (TRANSIT 0.2, DESCEND 0.1 rad/s), and the
  slowdown version was backed up. Verification on fresh rows 100–119:
  privileged 17/20 and noisy 17/20 (85%). Rows 107, 113 and 116 lost the
  grasp in `transit_place` with both detectors. Episodes averaged 34.8 s
  (30.3–41.4). 3 recorded episodes replayed exactly; 30/30 unit tests.
  **Correction:** the earlier "~5%" estimate given to the user was too
  optimistic for the current scene distribution; it was really about 15%.
- **Startup regression found and fixed**: fixer C's slower scene generation
  had made `collect_random` spend about 25 min building the 100-scene
  evaluation bank plus about 14 min on a 64-scene collection batch before the
  first rollout (≈41 min). Fix, in `scene_bank.py` and `collect_random.py`:
  - A deterministic on-disk scene cache at
    `data/.scene_cache/<workspace_hash>_<source_hash>.json`. The key is
    sha256 of the Workspace config and sha256 of `scene_bank.py`,
    `pick_insert_expert.py` and `place_insert.py`, so any config or code
    change misses the cache rather than serving a stale scene. Cached
    results were verified identical to uncached ones.
  - A lazy `iter_collection_bank()` generator, so only the rows actually
    attempted are generated. The seed → scene mapping is unchanged.
  - Measured: cold cache, first rollout at about 7.3 min; warm cache, about
    1.3 min, including the first rollout's own physics time. `--resume`
    works, manifests are valid, 30/30 tests pass.
- **Speed sweep (user chose "middle ground")**, from the bad rows 9, 107,
  113, 116 with the privileged detector:

| TRANSIT speed | row 9 | 107 | 113 | 116 |
| --- | --- | --- | --- | --- |
| 0.20 | fail | fail | fail | fail |
| 0.15 | fail | pass | fail | fail |
| 0.12 | fail | pass | fail | fail |
| 0.10 | fail | pass | fail | fail |
| 0.08 | fail | pass | fail | fail |

  Conclusion: below 0.15, slowing further gives no gain. Rows 9, 113 and 116
  slip at every speed down to 0.08 (drift 1–5 mm), so it is a scene-specific
  grasp-quality issue, not speed. Validation at 0.15:

| Set | Detector | Success | Episode length mean (range) |
| --- | --- | --- | --- |
| rows 100–119 | privileged | 18/20 (fails 113, 116) | 36.4 s |
| rows 120–139 (unseen) | privileged | 20/20 | 37.0 s (34.5–43.5) |
| rows 120–139 (unseen) | noisy | 20/20 | 38.0 s (33.6–45.5) |
| combined privileged 100–139 | | 38/40 = 95% | 36.7 s |

  **Merged**: `TRANSIT_SPEED_RAD_S = 0.15` in
  [controllers/place_insert.py](../controllers/place_insert.py) (DESCEND
  unchanged at 0.1). 30/30 unit tests pass after the merge.

#### Remaining issues

- Grasp lost during transport in about 5% of scenes (for example seeds
  1000009, 1000113, 1000116), regardless of transit speed down to 0.08 rad/s.
  Root cause not established (the order of pad friction saturation vs. the
  joint spike was never confirmed); accepted by the user as a known
  limitation.
- The fixture geom is commented out in [envs/scene.py:39](../envs/scene.py),
  which breaks `probe_table_insert.py --lower-fixture`.
- Scene generation is about 5x slower when the on-disk scene cache is cold
  (1.08 s → 5.25 s per scene, from fixer C's noise-buffered feasibility
  check). The cache (`data/.scene_cache/`) makes this a one-time cost per
  workspace/code-hash combination; a cold cache still means a roughly
  7-minute warm-up before the first rollout of a fresh run.

### 7. How to reproduce / review

```powershell
# Replay a failed preflight episode (inspect where the arm stops)
python scripts/replay_episode.py data/<out>/failed_preflight/train_00003.npz

# Small failure-reproduction run (physics only, no video)
python scripts/collect_random.py --episodes 3 --early-abort 4 --headless --video none --detector privileged --output <scratch>

# Validated reference paths
python scripts/validate_table_pickup.py
python scripts/probe_table_insert.py
```

Collection output must go to a new directory. Failed preflights are kept under
`failed_preflight/` for diagnosis and are never counted as training data.
