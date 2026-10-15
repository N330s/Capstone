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

## 2026-10-01: workspace_v2_cable30 (after the setup_env merge)

### Context
A collaborator merged automateData into the new workspace scene: a real table, spring-leaf socket on a fixture post, a cable, and the connector catalog. See [docs/MERGE_AUTOMATEDATA.md](MERGE_AUTOMATEDATA.md) (Thai). Their results:

| Scene | Result | Notes |
|---|---|---|
| v1 automation | 10/16 | 4 insertion stalls at the spring leaves, unresolved |
| v2 Type O | 3/3 | only 3 scenes, about 45 min each with the 60-segment cable |

### User requests (2026-10-01)
1. Socket always attached to its board: already implemented; `carry_fixture()` in envs/openarm_insert.py moves the post with the socket by the same rigid transform on every tabletop reset; tests/test_place_insert.py checks it; renders of sampled v1 and v2_cable30 scenes confirm. No change needed.
2. Wire from under the table: already exists in configs/workspace_v2.json and workspace_v2_cable30.json (box on floor, cable over near table edge to plug). Not visible because collect_random.py defaults to workspace_v1 (box on table). Use `--workspace configs/workspace_v2_cable30.json`.
3. "Sometimes OpenArm can't push the plug into the socket" = insertion stall. User chose workspace_v2_cable30 as target (Type O, 30-segment cable, ~3.8x faster than v2; the CLAUDE.md collection scene).

### Plan
Auto-mode loop (test -> fix -> test) with Sonnet subagents. Step 1 (in progress): baseline of ~16 scenes with privileged detector on v2_cable30, plus per-step diagnosis of each insertion stall (plug sag in hand vs wrist deflection). Prior v1 evidence (MERGE section 4.9): aligned < 30 um until leaf contact at ~9.9 mm, then 0.36 mm z sag within 0.6 s as push rises 12 -> 14 N; already failed: lateral gain x2, target lead <= 1 mm, 20 N cap.

### Results

#### Baseline (privileged detector, collection-bank rows 0–15)

| Outcome | Count | Notes |
|---|---|---|
| success | 10/16 | depth 18.87–19.00 mm, peak socket force 7.8–19.7 N |
| `expert_plan_seated_unreachable` (transit_place) | 5 | rows 3, 10, 13, 14 (and 28 in rows 16–31): the sampler accepted the scene and the plug was picked, but PlaceInsert could not plan the seated pose |
| insertion stall at the leaves | 1 | row 6 (seed 1000006): stops at ~10.15 mm (leaf contact), axial 21–24 N, ends as `grasp_lost` (drift 1.02 mm) or `insert_stalled` |

Row-6 variants (lateral/vertical pre-offset, grip changes) all still stall at ~10.15 mm, so it is a deterministic leaf stall. Rows 16–31 are in progress; so far rows 16, 21 and 29 succeeded, row 20 failed with `plan_insert_line_infeasible` and row 28 with `plan_seated_unreachable`.

So on v2_cable30 the dominant failure is a **planner/sampler mismatch**, not the push.

#### Full baseline, rows 0–31 (privileged)

**20/32 success (62.5%)**. Planning failures make up 8/32: `plan_seated_unreachable` on rows 3, 10, 13, 14, 27, 28, 31 and `plan_insert_line_infeasible` on row 20. Pickup failures make up 2/32 (row 23 `grasp_no_contact`, row 30 `numerical_failure`). Insertion failures make up 2/32 (row 6 `grasp_lost`, row 17 `force_limit_abort`), so 20/22 succeed once insertion is reached. Successful episodes average 34.0 s simulated, with peak axial force of 17.0 N on average (max 19.2 N).

#### Stall diagnosis (per-step traces)

- **The sag is arm compliance, not grasp slip.** In successes the in-hand pose stays within 52 µm and 0.2°. The commanded hand leads the actual hand by 12–22 mm axially. Joint 7 tracking error is 48–69 mrad. No torque saturates. The same ~130 µm z sag appears in successes, so it does not cause failures.
- **Controller wind-up.** After the 15 N cap trips, the advance freezes but the pose servo keeps integrating at 5 mm/s, so the push ramps about 5 N/s to 20–24 N and the cap is never enforced. Enforcing a literal 15 N cap breaks successes, which need 16–19 N; row 9 then stalled at 12.9 mm.
- **Row 6.** An unexplained free-flight y/z drift (~0.5 mm/s from 8.8 mm depth, with no contacts) makes one pin hit its leaf early (9.59 mm instead of 10.04 mm). That sets up a one-sided wedge (12 N wall load). The grip then slips once axial force exceeds 15.7 N (drift 1.0 mm at 19.9 N). These variants all failed: retrying with a wind-up reset (it jams the same way), a ±100 µm bias after 9 mm, slower insertion, a finer servo clip, and a firmer squeeze (no slip, but still stalls at 10.19 mm / 24 N).
- **Row 17.** The plug creeps 10 → 14.2 mm with about 22 mm of stored wind-up, then jumps 0.37 mm in one step. The wall force spike of 30.7 N goes over the 30 N abort.
- **Ranked fixes.** (1) Anti-wind-up force control (bounded target lead, about 2 N/s ramp, 20–22 N cap) plus an engaged-stall recovery that resets the wind-up and changes the approach. (2) A firmer squeeze, used together with (1). (3) A y/z correction before contact, from about 6 mm depth. Wrist stiffness is not recommended before anti-wind-up is in place.

#### Fixer S: planning failures (done, merged 2026-10-02)

- **Root cause.** When the plug starts close to the table edge where the cable drops (first cable waypoint x = 0.235), the cord pulls taut during the lift and swings the friction-held plug about 25° around the pinch axis before handoff. Measured handoff pitch: −23.0 to −28.6° on failing rows, versus 0.2° elsewhere. The sampler and `insert_feasible` model only the nominal, unswung grasp. From the swung grasp, 0 of 29 random sockets are plannable, against 27 of 29 for the unswung grasp. Predicting the swing therefore cannot recover these scenes, so they have to be rejected.
- **Fix.** In `data_pipeline/scene_bank.py`, new `Workspace` fields `cable_edge_waypoint_m` (read from the workspace's first `cable.rest_waypoints_world`) and `cable_edge_min_m = 0.06`. `_cable_check` now rejects `cable_taut_on_lift` when the plug's cable attachment is within 0.06 m of that waypoint in xy (failing rows sat at 0.049, good ones at 0.059 or more). v1 has no waypoint, so v1 sampling is unchanged. Three asset-free tests were added to `tests/test_workspace.py`.
- **Result.** Rows 1, 3, 4, 10, 13, 14, 20, 27, 28 and 31 were re-run (failing rows resample to new scenes): **10/10 success, zero plan failures**. Unit tests 46/46. The effective plug x range shrinks to about ≥ 0.337 m at yaw 0. Cold scene generation is slower under contention (about 28 s per row).
- **Environment note.** Each simulation process uses about 1.5 GB of RAM; running more than about 4 at once caused `MemoryError: bad allocation`.

#### Fixer R: insertion stall (done, merged 2026-10-02)

- **Shipped (controllers/place_insert.py only; no physics or env change):** a force-controlled push once the leaves touch.
  - The target is the plug x plus a bounded lead (0.1 mm/N of force error, clamped to −0.5…+1 mm), so the integrating servo can no longer wind up.
  - The force setpoint ramps at 4 N/s to FORCE_CAP_N = 21 N, with `push_force_cap_n` as the floor. It holds when leaf force exceeds 0.6 × the leaf abort.
  - The engaged push has its own 15 s budget.
  - Behaviour before leaf contact is unchanged.
- **Engaged-stall recovery** (pull back, re-align, re-push, with guards) is implemented but **ships OFF** (`MAX_RECOVERIES = 0`). It rescued none of the stall scenes, and on v1 00000 and v2 row 6 the pull-back ended in `force_limit_abort`. The reason: the arm's lateral stiffness at the hand is only about 1 N/mm, so releasing the stored load swings the plug through the clearance.
- **Results:**

| Case | Before | After |
|---|---|---|
| v2 row 17 | force_limit_abort | **success** |
| v2 row 6 | grasp_lost at ~10.15 mm | unchanged |
| v2 rows 0, 1, 2, 4, 5, 7, 8, 9 | – / success | success |
| v1 previously passing (00001, 00002, 00005, 00008) | success | success |
| v1 stalls 00000, 00003, 00007, 00009 | insert_stalled | still stalled (3 of them now push 0.3–1.3 mm deeper) |

- **Measured dead ends:** the leaf pair pins the plug's y regardless of where it is aimed, so y bias does nothing. A z pre-bias only shifts the stall depth slightly; at +300 µm it hit the leaf abort. Dither at 2–3 Hz and a slower engaged push changed nothing. A firmer squeeze stops the row-6 slip but it still stalls, and it turns row 17 into an abort.
- **Remaining:** row 6's one-sided wedge comes from an unexplained y/z drift that starts at 8.8 mm depth, before contact. That has to be fixed upstream; retrying does not help.
- Unit tests after merging both fixes: 46/46.

#### Final verification (2026-10-02, unseen rows 40–55, both fixes merged)

| Detector | Accepted / attempted | Failures |
|---|---|---|
| privileged | **14/16 (87.5%)** | row 48 `grasp_lost` in insert (drift 1.00 mm at 5.2 mm depth, the same family as row 6); row 53 `force_limit_abort` at 14.5 mm (passes with the noisy detector) |
| noisy | **13/16 (81.3%)** | row 48 `grasp_lost` (as above); row 50 `ik_infeasible` (noisy pose estimate near a joint limit; fails fast in 18 s); row 54 `plan_seated_collision` (noisy grasp estimate, finger hits the socket frame) |

These are different rows from the baseline, so the comparison with 62.5% (rows 0–31) is indicative only. There were zero planning failures with the privileged detector (the baseline had 25%) and no reset rejections. Successful episodes reach 18.8–19.0 mm depth and run 33–41 s simulated.

`collect_random.py --workspace configs/workspace_v2_cable30.json --eval-scenes 2 --episodes 2 --record --detector noisy` accepted and recorded 2/2. The manifest has `complete=true`, `validate_episode` passes, and strict `replay_pilot.py` gives qpos/qvel error 0.0. Unit tests 46/46.

**Remaining:** a one-sided leaf wedge with grip slip (rows 6 and 48) that begins as a y/z drift before contact; one late `force_limit_abort` (row 53); and two noisy-detector misses (rows 50 and 54) that the scene generator's noise check does not cover for the Type O plug. Leaf-force maxima of 28–53 N appear larger than the 21 N axial cap; this was not checked further.

#### Iteration 1 (done)
- Diagnosis: the row-6 stall mechanism, step by step (plug slip in the hand vs. wrist deflection), and the full 32-row breakdown.
- Fixer R (worktree, insert phase of `controllers/place_insert.py`): recovery for stalls once the leaves are engaged (currently terminal). It backs off, re-aligns with a sag pre-bias, and retries; primary case is row 6.
- Fixer S (worktree, planning/feasibility code and `data_pipeline/scene_bank.py`): make the sampler reject exactly the scenes the runtime cannot plan (`plan_seated_unreachable`, `plan_insert_line_infeasible`).

## 2026-10-02: overhead scene camera and per-frame JPEG export

Collection tuning is **paused** at the user's request. The state above (v2_cable30: 87.5% privileged, 81% noisy on rows 40–55) is where it stopped.

- **Overhead camera.** `cameras.scene_rgb` was moved to an overhead mount in every workspace file (`workspace_v1`, `v1_leaf5n`, `v2`, `v2_cable30`). The camera sits at (0.12, −0.21, 0.98) m, 0.66 m above the table top, looks at (0.40, −0.21, 0.36) (24° from vertical) with a 48° field of view. It was chosen from rendered candidates: a straight-down view cannot see the socket slots, and wider views leave the task area small at 320×240. The plug spawn area, gripper, cable and socket face are in view; the hand occludes the plug from above during the grasp. The per-scene camera jitter (±4 mm, ±1.5°) still applies. Editing the workspace JSON changes its hash, so earlier datasets no longer replay strictly. [docs/WORKSPACE.md](WORKSPACE.md) is updated.
- **JPEG frames.** `data_pipeline/episodes.py` `save_episode()` now also writes `<episode>/cameras/head/<i>.jpeg` (from `scene_rgb`) and `<episode>/cameras/wrist/<i>.jpeg` (from `wrist_rgb`), with i = 1…T+1 and file i = observation i−1, at JPEG quality 95. The folders and frame count are recorded in `metadata.json` under `camera_frames`. This applies to every collector that uses `save_episode`. `episode.npz` (lossless) is unchanged and remains what checksums and training use.
- Unit tests 46/46.

## 2026-10-08: per-scene preflight → record, exact resume

**Why.** `collect_random.py --record` used to preflight every scene first and record afterwards. A stop during preflight made `--resume` treat the partial accepted list as final, so the dataset came out smaller than asked. A kill while an episode was being saved left a half-written folder, and the resume then crashed because `save_episode` refuses to overwrite. The user asked for one scene at a time (record only after a successful preflight), and for a resume that redoes exactly the interrupted stage.

**What changed (only `scripts/collect_random.py`).**
- `collect_pass` replaces `record_pass` for `--record` runs. For each row it preflights (no images); if that succeeds, it records straight away. The run ends when `--episodes` episodes are recorded or the attempt budget (`episodes × attempt_ratio`) runs out. A preflight-only run (no `--record`) still uses `preflight_pass`, unchanged.
- One env serves both stages. `env.images` is switched on only for the recording, since `observe()` checks it on every call. That keeps one model in memory (about 1.5 GB per process, as before), and no hashed env source changed.
- `progress.json` is written after every stage. It holds the next row to preflight (`cursor`), any accepted row whose recording has not finished (`pending_record`), the accepted/rejected/recorded/failed lists, and the run signature (workspace, detector, validation fraction). All JSON outputs are now written atomically (temp file + rename).
- `--resume`:
  - stopped while recording → delete the partial `<id>/` folder and any stale `failed_attempts/<id>*`, then re-record that row;
  - stopped during preflight → preflight row `cursor` again.

  It refuses to resume under a different signature. A higher `--episodes` is allowed, to extend a run.
- Rows are still drawn from the same 64-row blocks (`iter_collection_bank(start=block)` + skip), so ids, seeds and splits do not depend on where a run stopped.
- Each recorded episode stores `trace_matches_preflight`: whether the recorded qpos trajectory equals its preflight bit for bit. It is null for a recording redone after a resume, because the preflight trace is gone by then.
- `manifest.json` is rewritten after each stage. `complete` now means "recorded ≥ `--episodes`". There is also a new `collection_mode` field.

**Verification** (v2_cable30, privileged detector, `--episodes 2 --record --headless --video none --eval-scenes 1`, three runs in parallel; driver `scratchpad/resume_test.py` kills the process with TerminateProcess, like a crash):

| Run | What happened | Result |
|---|---|---|
| reference | uninterrupted | 2/2 recorded, `trace_matches_preflight` true for both |
| kill_record | killed 1 s after `train_00000/` appeared (mid `save_episode`, 1 file written). `progress.json` showed `pending train_00000`, records [] | resume removed the partial folder, re-recorded `train_00000`, then preflighted and recorded `train_00001`: 2/2 |
| kill_preflight | killed 5 s into the preflight of `train_00001`. `progress.json`: cursor 1, nothing pending, records [`train_00000`] | resume restarted `train_00001`'s preflight and recorded it: 2/2 |

- In both resumed datasets, all 11 non-image arrays (qpos, qvel, actions requested/applied, reward, flags, timing, phase) are **bit-identical** to the reference. Both have 1797/1810 JPEGs per camera and no `.tmp` or `failed_*` leftovers.
- `data_sha256` differs between runs, and so do the image arrays: at most 1/255 on about 0.005% of pixels. **This is not caused by resume.** Re-rendering the same state four times in one process alternates between two pixel hashes per camera (OpenGL driver on this Windows machine). The rendered images are therefore not bit-reproducible; the physics is. `replay_pilot.py` checks only the physics.
- Strict `replay_pilot.py` on both resumed datasets: all 4 episodes qpos/qvel error **0.0**, outcome success. Unit tests **46/46** OK.
- **Docker** (`openarm-collect` 507 MB and `:cpu` 796 MB, both rebuilt with this script):
  - The first in-container record failed with `module 'mujoco' has no attribute 'Renderer'`. The image set `PYOPENGL_PLATFORM=egl`, which conflicts with `-e MUJOCO_GL=osmesa`; mujoco then swallows the ImportError and drops `Renderer`. Rendering had never been reached in the container before, because the 2026-10-06 run was killed with exit 137. Fix: do not set `PYOPENGL_PLATFORM` in the Dockerfile (mujoco derives it from `MUJOCO_GL`).
  - After the fix, the `:cpu` image (OSMesa) recorded 1/1 with `complete: true` and `trace_matches_preflight: true`, 1794 + 1794 JPEGs, 155 MB for the episode. Strict `replay_pilot.py` in the container gave error 0.0.
  - The episode has 1793 steps against 1796 on Windows for the same seed. This is the known cross-platform floating-point difference, so do not mix platforms in strict replay.
  - The exit-137 kill: `docker stats` peaked at 2.96 GiB for one collector, so two in parallel on the 7.6 GiB Docker VM ran out of memory. `docs/DOCKER_COLLECTION.md` now says about 3 GB per container.
  - EGL on an NVIDIA GPU is still unverified; it needs the Linux machine.

## 2026-10-10: wider scene variance + success loop (v2_cable30, noisy detector)

**Request.** Make plug and socket placement as wide as the arm can really reach, add `--start-row` so runs draw fresh scenes, and raise success to ≥ 90% with the noisy detector.
- **Loop:** Opus plans from code analysis and test reports, Sonnet codes, Sonnet tests; work runs in parallel where possible.
- **Constraints:** Docker is untouched. `envs/*.py` may change. Changes to `configs/workspace_v2_cable30.json` need the user's confirmation.

**Starting point.**
- **Accepted scenes span:**
  - plug x 0.28–0.38, y −0.30…−0.12, yaw −20…60°;
  - socket x 0.38–0.46, y −0.36…−0.26 (nominal −0.36…−0.20, but the cable and hand-home checks reject y > −0.257), yaw −10…30°.
- **Repeats:** every run starts at row 0, so the same scenes repeat.
- **Success:** 81% noisy / 87.5% privileged (rows 40–55).

**Stop criterion.** Noisy ≥ 90% on 32 fresh rows at the widest kept ranges, confirmed on a second fresh 48-row set; otherwise stop after 6 iterations.

### Iteration 0 (parallel)
- A (Sonnet, worktree): `--start-row` for `collect_random.py`.
- B (Sonnet): sharded test harness `run_rows.py`; baseline on rows 5000–5031, noisy.
- C (Opus): reach map, i.e. which sampler checks bind where, and a staged widening proposal.
- **A done** (commit a653bba, fast-forwarded onto `automation_new_environment`).
  - `--start-row N` (default 0; attempt k draws row N+k, seed `COLLECTION_SEED_BASE + N + k`). Blocks and splits stay aligned relative to N, so default 0 reproduces earlier runs. The helper `row_block()` is shared by both passes.
  - `start_row` is stored in `run_signature` (so `--resume` refuses a different value), in the manifest and in `preflight_stats`. A guard keeps seeds inside `SEED_NAMESPACE_WIDTH`.
  - Tests 50/50.
- **Incident:** removing coder A's worktree also deleted the main `third_party/openarm_mujoco`. `git worktree remove` followed the `third_party` directory junction the coder had created. It was re-fetched with `scripts/fetch_openarm.ps1` (pinned 161039cd); `scene_sha256` and `source_hashes` match the 10-08 recordings exactly. B and C were told to rerun anything that ran while the files were missing (about 7 minutes). From now on: remove a junction itself (`rmdir`) before removing its worktree.
- **C done** (reach map: `scratchpad/loop/iter0_reach_map.md` + maps).
  - **Physical bands:** socket insertion reach caps x at ≤ 0.51; the feasible y band shifts with yaw (centre ≈ −0.27 + 0.0051·yaw, 0.24–0.30 m wide). Plug pick reach is x ≤ 0.48 at yaw 0 (≤ 0.45 at ±45°).
  - **Cable:** `cable_taut_on_lift` makes x ≥ 0.337 at yaw 0, so 44% of today's draws are wasted because `plug_x_range` starts at 0.28.
  - **Conservative rules:** `socket_at_hand_home` (0.10 m) is what cuts socket y at −0.257. The reach annulus is redundant with IK, and keepout and cable reach never fire on v2.
  - **Settle drift:** the 1 s reset settle moves the plug 0.1–8.4 mm and −17…+23°, but the sampler certifies the pre-settle pose.
  - **Proposal:** Stage 1 (plug x 0.32–0.43, y −0.38…−0.06, yaw −40…60; socket x 0.37–0.49, y −0.44…−0.10, yaw −15…35; hand keep-out 0.06 + exact home-collision check; settled-pose certification), then Stage 2 at full reach. Two optional JSON proposals (move the cable edge waypoint; lower the fixture post) need the user's approval and are not needed for Stages 1–2.
- Iteration 1 code started early (Sonnet, worktree): Stage 1 in `scene_bank.py`, in parallel with baseline B.
- **Stage 1 coded** (worktree branch, commit c14119c, not merged yet so the baseline stays clean).
  - `V2_STAGE1` ranges apply to `workspace_v2*` only; v1 is unchanged (tested).
  - New checks: `_home_collision_check` (exact open-hand vs socket/fixture contact at the home posture, scratch data, ~3 ms), and settled-pose certification (pick check + noise sweep on the plug pose after the reset settle; drift recorded in the diag).
  - New reject reasons: `socket_home_collision`, `settled_pick_infeasible:*`, `settled_noise_margin:*`, `settled_pose_unavailable`. Tests 54/54.
  - **Smoke test** (rows 6000–6011): 7.0 draws/scene. Accepted spans: plug x 0.338–0.428, y −0.348…−0.082; socket x 0.404–0.489, y −0.390…−0.110. Settle drift 0.2–8.1 mm, −7…+12°. One noisy rollout succeeded.

### PAUSED (2026-10-10, at the user's request): state for resuming
Scratch dir: `C:\Users\Psukj\AppData\Local\Temp\claude\c--C-robotic-Capstone\eca5e112-cad1-4b10-b17a-f06854798ffd\scratchpad\loop` (LOOP).
- **Baseline (current code, noisy, rows 5000–5031):** 30/32 finished, **23 success (77%)**; rows 5030 and 5031 not run (the tester was stopped). Per-row JSONs are in `LOOP/iter0_baseline/rows/`, failure traces in `LOOP/iter0_baseline/failures/`.
  - **Failures:**
    - 5006 `plan_seated_unreachable`
    - 5008, 5025 `plan_insert_line_infeasible`
    - 5010, 5013, 5027 `force_limit_abort`
    - 5023 `grasp_lost`
  - **Resume:** `LOOP/run_rows.py --rows 5030,5031 --out LOOP/iter0_baseline_rest`, then merge with the existing rows to write the report (`run_rows.py` has the merge step).
- **Test harness:** `LOOP/run_rows.py` (`--rows`, `--shards`, `--repo` to test a worktree, `--detector`). Max 3 sims in parallel (RAM).
- **Reach map:** `LOOP/iter0_reach_map.md` + PNGs (done).
- **Merged:** `--start-row`, commit a653bba on `automation_new_environment`.
- **Stage 1 sampler:** done but NOT merged. Worktree `.claude/worktrees/agent-a11955fff299b3681`, branch `worktree-agent-a11955fff299b3681`, commit c14119c, tests 54/54. That worktree holds a real *copy* of `third_party` (not a junction).
- **Next steps:**
  1. Finish baseline rows 5030–5031 and write the report.
  2. Merge c14119c.
  3. Test Stage 1 on fresh rows (e.g. 6100–6131, noisy).
  4. Opus plans iteration 1 from both reports.

  Stop criterion: ≥ 90% noisy on 32 fresh rows, confirmed on 48.

### Iteration 1 (resumed 2026-10-13; the user authorized iteration 1 only)
- **Baseline complete** (old code a653bba, noisy, rows 5000–5031): **25/32 = 78.1%** (Wilson 95% [61%, 89%]).
  - **Failures:**
    - 5006 `plan_seated_unreachable`
    - 5008, 5025 `plan_insert_line_infeasible`
    - 5010, 5013, 5027 `force_limit_abort` at 14.5–15.0 mm depth
    - 5023 `grasp_lost` (drift 1.02 mm in insert)
  - **Spans:** plug x 0.337–0.379, y −0.295…−0.134, yaw −10…54; socket x 0.395–0.460, y −0.359…−0.243, yaw −9…24.
  - **Sampler:** 9.2 draws/scene; `cable_taut_on_lift` caused 142 of 262 rejects.
  - **Timing:** scene generation 54 s, rollout 943 s mean (3 in parallel).
- Stage 1 merged (fast-forward to c14119c); tests 54/54.
- The Opus planner is analysing the baseline failures on top of Stage 1, output `LOOP/iter1_plan.md`.
- **Opus plan** (`LOOP/iter1_plan.md`; reproductions matched the baseline exactly).
  - **Planning failures (5006, 5008, 5025):** `plan_grasp` picks the least-tilted grasp whose insertion plan only just clears the 0.05 rad joint margin (joint 7). The pick-time check uses the noisy socket and the nominal grasp, the runtime uses the true socket and the measured grasp, so marginal plans flip. Tilt −15 grasps: 3/9 success vs tilt −30: 20/21. Stage 1's settled-pose check does not close this gap.
  - **Insertion failures (5010, 5013, 5027 force abort; 5023 grasp_lost):** during the leaf push the arm's compliance stores ~18 mm axial + 5–6 mm lateral + 4.4° pitch offset. The cross-coupled stiffness turns that into lateral finger force that pins the blades on the walls, so the wall force spikes to 31–44 N when the leaf load drops at 14.5–15 mm. A lateral force admittance (1e-4 m/N, branched from first leaf contact) turned all 4 into successes at 19 mm, peak wall force 3–7 N.
- **Coders A/B/C launched in parallel** (worktrees with copied `third_party`):
  - A: robust insertion margin (`plan_grasp` prefers margin ≥ 0.10; sampler requires 0.09 in `V2_STAGE1`).
  - B: lateral force admittance in the insert branch of `PlaceInsert.action`.
  - C: runtime planning retry at 0.03 margin.
- **A done** (commit 8803ab6). `insert_feasible` returns the minimum joint margin; `plan_grasp` prefers insert margin ≥ 0.10 (falls back to the best available); the sampler floor is `pick_insert_margin_rad` = 0.09 (exact and settled pick checks only). Tests 57/57.
  - Kinematics scan: 5006, 5008, 5025, 5004, 5016 all switch from tilt −15 (margin 0.056–0.072) to tilt −30 (margin 0.10–0.17 at the detected and true socket).
  - Rows 6200–6207: 5.9 draws/scene; `insert_margin_low` caused 2 of 40 rejects.
  - One rollout (row 6200) failed with `expert_grasp_misaligned` in the pick (38° plug error, never reached insertion). Not investigated yet; watch for it in the test.
- **C done** (commit 7d6ca83). `plan_insertion(min_margin=0.05)` by default; `PlaceInsert.start` retries once at `FALLBACK_JOINT_MARGIN_RAD` = 0.03 and records `insert_plan_fallback` and `insert_plan_margin_rad`. The screens stay strict. Tests 55.
  - With the old tilt −15 grasps: 5006 and 5008 become plannable (margin ~0.044); 5025 stays infeasible (a continuity limit).
  - Full rollouts on the baseline scenes: 5008 succeeded **using the fallback** (margin 0.0437, 18.9 mm); 5006 succeeded too (fallback use not confirmed).
- **B done** (commit f9654b0). Lateral finger-force admittance when the leaves are engaged (`LATERAL_ADMIT_M_PER_N` = 1e-4, cap 25 mm, attitude still servoed to the socket; stat `peak_finger_lateral_n`).
  - Full noisy rollouts, all success at 18.9–19.0 mm: 5027, 5013, 5023 and 5004 (no regression).
  - Peak wall force 3.1–3.5 N (baseline failures: 31–44 N).
  - 5023 is still marginal on grasp drift (0.983 mm vs the 1.0 mm limit).
- **Merged** A, B, C by cherry-pick onto the branch (fb6f6cd, 7be098f, bc72ba3); tests **58/58**.
- **Iteration 1 test launched:** `run_rows.py --detector noisy --rows 6100-6131 --shards 3 --out LOOP/iter1_test` (fresh rows, Stage 1 ranges + A/B/C).
- **Iteration 1 result** (HEAD bc72ba3, noisy, fresh rows 6100–6131, Stage 1 ranges; `LOOP/iter1_test/summary.md`): **30/32 = 93.8%** (Wilson 95% [79.9%, 98.3%]) vs baseline 25/32 = 78.1% [61.2%, 89.0%]. The sets are different rows at wider ranges, so the confidence intervals overlap.
  - **Gone:** none of the baseline families recurred (0 planning failures, 0 `force_limit_abort`, 0 `grasp_lost`, 0 `grasp_misaligned`).
  - **Remaining failures:** 2× `insert_stalled` (6113 at 4.7 mm, 6123 at 9.0 mm). In both, the axial push saturates at the ~19.5 N cap while the leaf load is 21.6–24.5 N, only one leaf compresses, and the socket yaw is near the top of the range (30.9°, 27.1°). This is a one-sided leaf load at high socket yaw.
  - **Per-row:**
    - All rows chose tilt −30; the 0.03 fallback fired 0/32.
    - Pick-plan insert margin min 0.069. 3 rows are below the sampler's 0.09 floor yet succeeded: the runtime plans on the noisy detection, the sampler on the exact pose.
    - Peak wall force in successes ≤ 11.8 N; finger lateral force 5.5–14.7 N.
    - Grasp drift > 0.8 mm in 2 successes (0.95 and 0.84 mm; limit 1.0).
  - **Variance** (span, baseline in brackets):
    - plug x 0.095 m [0.042], y 0.300 m [0.161], yaw 82° [64];
    - socket x 0.112 m [0.064], y 0.322 m [0.116], yaw 46° [33].

    That is 1.3×–2.8× wider.
  - **Sampler:** 6.6 draws/scene (baseline 9.2); scene generation 36 s mean.
  - **Rollout time:** 713 s mean; total 2.25 h with 3 shards.
- **Status:** iteration 1 is done and the loop is **stopped at the user's request** (no iteration 2).
  - The loop's stop criterion is not fully met yet: Stage 1 is not the widest range set (Stage 2 = full reach), and the ≥ 90% result still needs confirmation on a fresh 48-row set.
  - Obvious next target: high-socket-yaw one-sided leaf stalls (6113, 6123).
  - Docker image not rebuilt; it still has the older collector code.

## 2026-10-14/15: recovery demonstrations (injected faults)

**Request.** Episodes in which the robot makes a mistake and then recovers and completes the
task:
- the gripper closes before reaching the plug;
- it grips the plug in the wrong place;
- it puts the plug off the socket hole.

Only collection changes; the training-side masking is out of scope. Design and results are in
`docs/DATA_COLLECTION.md` ("Recovery demonstrations").

- **Code.** New `controllers/faults.py`. Fault phases and per-step `label` in `PickInsertExpert`.
  `PlaceInsert(probe_offset_m=(y, z))`. `collect_random.py --faults`. `step_label` in
  `save_episode` / `validate_episode`. 6 new tests; suite 64/64. No `envs/` or manifest-hashed
  file changed.
- **Clean mode unchanged.** Row 6100 without `--faults` gives a bit-identical qpos trace
  (1981×148) and the same phases and diagnostics keys as HEAD bc72ba3.
- **Smoke runs** (preflight + video; rows 7000–7004 premature_close, 7100–7102 and 7150–7156
  wrong_grasp, 7200–7203 misaligned_insert):
  - premature_close: 4/5. Closed 34–42 mm above the table with no contact; detected in 2.6 s;
    recovered in 3.0–3.3 s.
  - misaligned_insert: 4/4. Face contact detected at 4.9–5.2 N (wall abort 20 N).
  - wrong_grasp: the first three rows used the original code.
    - **Bug (row 7101):** a shift toward the cable end put the hand on the cable, which the
      planner ignores (5.3 N), and the expert ended `close_tracking_timeout`. **Fix:** shift toward
      the pin end only, and treat an arrive timeout during the fault as detection.
    - With the fix: 4/7.
  - **Range change:** row 7201 drew a 3.41 mm aim error and measured 3.78 mm lateral, against
    PlaceInsert's 4 mm off-axis abort. The cap went from 3.5 to 3 mm.
- **Recorded sets.** `data/openarm_v2_recovery_smoke_a` (rows 7300–7303) and `_b` (7400–7403),
  `--faults all`:
  - 6 recorded in 8 attempts;
  - all pass `validate_episode` and all have `trace_matches_preflight`;
  - strict replay (`replay_pilot.py`) passes for all 6 with **0.0** max qpos/qvel error (`replay_report.json` in each set).
- **Totals with the final code:** 24 faulted attempts; the fault was detected and recovered in
  **24/24**; the task then succeeded in **18/24**.
- **Failures after the recovery:**
  - 7000, 7151, 7301 `insert_stalled` (known leaf-stall family, not re-run clean);
  - 7154 and 7155 also fail clean on the same scene;
  - 7102 (original code) also fails clean;
  - **7401 is attributable to the recovery:** the re-survey's fresh detection (5.1 mm error vs
    3.9 mm on the first look) gave an off-centre re-grasp and `plan_seated_collision`; the same
    scene succeeds clean. A possible fix, not applied: average several detections in the
    re-survey. That reduces the noise part of the error, not the per-episode bias.
- **Open:**
  - the training loaders must mask `step_label == "fault"` before any BC;
  - a shift-only wrong grasp is hard to see from `scene_rgb` (visible in `wrist_rgb`);
  - the Docker image does not have this code.

## 2026-10-15: wrong_grasp = miss, chest D435 RGB-D camera, joint torque in collected data

**Request.**
1. wrong_grasp should grip in a totally wrong place. The user chose: miss the plug entirely.
2. Add a chest camera like the real OpenArm v1, stereo depth type. The official OpenArm 01
   Release No.2 chest camera is a RealSense **D435**. The user chose rendered depth with D435
   noise, in `workspace_v2_cable30` only.
3. Store joint torque in future data collection.

Working rules from the user:
- keep `claude_progress.md` (task checklist);
- run tests in parallel;
- no recording for tests;
- logs and results stay as temp files in the session scratchpad.

- **Miss.**
  - **Planner:** `plan_table_pick(grasp_offset=...)` moves only the grasp point. The default is
    unchanged, so the sampler is unaffected.
  - **Fault:** the expert plans 6.5–9.5 cm past the pins or to either side (never toward the
    cable), with fallbacks in direction and distance. The open fingers must stay ≥ 10 mm from the
    true plug and the live cable along the whole descend, checked with `mj_geomDistance` on
    scratch data.
  - **Detection and recovery:** empty-close detection, as premature_close. Then
    open → retreat → re-plan from the same detection. The re-survey is removed, along with the
    row-7401 failure mode it caused.
- **Chest camera.**
  - **Geometry:** `cameras.chest_rgbd` with a massless visual module on `openarm_body_link0`, at
    (0.095, 0, 0.64) estimated from the torso mesh. 42° vertical FOV, 424×240.
  - **Depth:** `envs/d435.py` degrades rendered z-depth to uint16 mm with Intel's RMS noise, range
    limits, edge holes and dropout.
  - **Aim:** facing forward, the socket was in view 0/18 and the plug 12/18. The user chose to aim
    it at the workspace, (0.42, −0.25, 0.36), giving 18/18 for both.
  - **Fix:** the first aimed render had a rolled horizon (`camera_frame` assumes downward
    cameras). Fixed with `level_camera_frame`.
- **Torque.** `info["joint_torque_{applied,applied_peak,gravity,constraint}_nm"]` (14 values), saved
  by `save_episode` as (T, 14) arrays.
- **Tests:** 76/76 (+12):
  - `test_d435`: units, range, determinism, the noise formula, holes, intrinsics;
  - `test_faults`: miss parameters, saving and validating the sensor arrays;
  - `test_sensors`: chest RGB-D shapes, depth vs geometry, reproducibility, torque ≈ gravity at
    rest and within limits, v1 unchanged, the planner offset.

  One existing test mocked `_candidate` with the old signature; the offset is now passed only when
  set.
- **Physics unchanged.** Clean row 6100 is bit-identical to the pre-change trace (1981×148).
  The hashed sources changed, so **datasets recorded before this no longer pass strict replay**;
  documented.
- **Miss smoke:** 6/6 succeeded (rows 7500/7501/7550/7551/7600/7601), 0 plug contacts, clearance
  13–42 mm, recovery 2.4–3.0 s.
- **Not done:** no episode was recorded with the new arrays (by instruction); saving is covered by
  synthetic-episode tests. Docker not rebuilt.

### End-to-end test (2026-10-15, recorded episodes; moved to `data/openarm_v2_e2e_sensors_test/` at the user's request:
`A_clean`, `B2_premature`, `C2_misaligned`, `D_miss` + `e2e_report.json` and `sheet_*.png`; logs stayed in the scratchpad)
- **Bug found and fixed.** `collect_random.py --faults ... --record` crashed in `build_manifest`
  before the first recording: it still referenced the removed `WRONG_GRASP_SHIFT_M`/`_YAW_DEG`.
  - Missed earlier because the unit tests never built a fault manifest and the miss smoke runs were
    preflight-only.
  - Fixed: the manifest lists `wrong_grasp_directions` and `wrong_grasp_distance_m`.
  - Regression test `FaultManifestTests`: fails on the old code, passes on the fix. Suite 77/77.
- **Recorded:** one episode each, rows 7700 clean, 7720 premature_close, 7740 misaligned_insert and
  7760 wrong_grasp miss. **All 4 succeeded.** For every episode:
  - `validate_episode` passes and `trace_matches_preflight` is true;
  - strict `replay_pilot.py` gives 0.0 qpos/qvel error.
- **Data checks:**
  - **Shapes:** chest RGB (T+1, 240, 424, 3) uint8; depth (T+1, 240, 424) uint16; torque
    (T, 14) ×4.
  - **Depth:** 66–75% valid, median 487–528 mm.
  - **Exported frames:** chest JPEG and chest_depth PNG counts equal T+1.
  - **Labels** line up with the phases.
- **Torque:**
  - right-arm peaks ≤ 11.5 N m (limits 40/27/7);
  - at rest applied − gravity = 0.000;
  - contact part about 0.11 N m in free motion (including carrying the plug) vs about 0.36 N m in
    insert/hold;
  - the parked left arm is constant.
- **Depth at the plug's pixel vs geometry:** −21 to −57 mm when the plug's top or the hand is in
  front. At the home pose the right arm can hide the plug from the chest camera entirely (row 7760:
  −274 mm, i.e. the arm). That is real occlusion, not a bug.
- **Cost:**
  - 404–585 MB per episode (0.21–0.24 MB/step) against 184 MB (0.084 MB/step) without the chest
    camera, so about 2.5–2.9×;
  - wall time 1000–1875 s per scene (preflight + record) with 3 runs in parallel.
