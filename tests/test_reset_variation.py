"""Tabletop reset warm-up (camera warm-up outside the demonstration) and start-posture variation.

Sampler-only tests run without the robot assets; the env tests build the v2_cable30 scene (slow).
"""
import json
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from data_pipeline import scene_bank as sb

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = "configs/workspace_v2_cable30.json"
HAS_ASSETS = (ROOT / "third_party/openarm_mujoco").is_dir()
# Pre-change captures (baseline_6100.npz, baseline_options.json); those tests skip when it is unset.
BASELINE_DIR = Path(os.environ.get("RESET_VARIATION_BASELINE_DIR", "__unset__"))
BASELINE_SEEDS = ("1006100", "1006101")      # sampled fresh (slow with an empty scene cache)


class PostureSamplerTests(unittest.TestCase):
    """_sample_posture with the feasibility checks stubbed out: pure rng/bookkeeping."""

    def patched(self, **kw):
        names = {"home": "_home_collision_check", "pick": "_pick_check", "noise": "_pick_noise_margin_check"}
        stack = [mock.patch.object(sb, names[k], **({"return_value": None} | kw.get(k, {}))) for k in names]
        return stack

    def draw(self, seed, ws=None, **kw):
        ws = ws or sb.Workspace()
        patches = self.patched(**kw)
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        return sb._sample_posture(seed, {}, ws)

    def test_deterministic_bounded_and_seed_dependent(self):
        ws = sb.Workspace(arm_offset_max_rad=0.05)
        offsets = {}
        for seed in (1, 2, 1006100, 1006101):
            a, diag = self.draw(seed, ws)
            b, _ = sb._sample_posture(seed, {}, ws)
            self.assertEqual(a, b)
            self.assertEqual(len(a), 7)
            self.assertLessEqual(np.abs(a).max(), 0.05)
            self.assertGreater(np.abs(a).max(), 0.0)
            self.assertEqual(diag["tries"], 1)
            offsets[seed] = tuple(a)
        self.assertEqual(len(set(offsets.values())), len(offsets))

    def test_zero_max_disables(self):
        offset, diag = self.draw(5, sb.Workspace(arm_offset_max_rad=0.0))
        self.assertEqual(offset, [0.] * 7)
        self.assertEqual(diag["tries"], 0)

    def test_redraws_posture_only_then_falls_back_to_zero(self):
        ws = sb.Workspace(arm_offset_max_rad=0.05)
        offset, diag = self.draw(7, ws, home={"return_value": "socket_home_collision"})
        self.assertEqual(offset, [0.] * 7)
        self.assertEqual((diag["tries"], diag["fallback_zero"]), (sb.POSTURE_TRIES, True))
        self.assertEqual(len(diag["rejected"]), sb.POSTURE_TRIES)
        # first two draws rejected, third accepted: a later draw of the same stream
        calls = iter([True, True] + [False] * 20)
        with mock.patch.object(sb, "_home_collision_check",
                               side_effect=lambda *a, **k: "x" if next(calls) else None):
            third, diag3 = sb._sample_posture(7, {}, ws)
        self.assertEqual(diag3["tries"], 3)
        stream = np.random.default_rng([7, sb.POSTURE_STREAM])
        stream.uniform(-0.05, 0.05, 7), stream.uniform(-0.05, 0.05, 7)
        np.testing.assert_array_equal(third, stream.uniform(-0.05, 0.05, 7))

    def test_checks_receive_the_offset(self):
        ws = sb.Workspace(arm_offset_max_rad=0.05)
        patches = self.patched()
        mocks = [p.start() for p in patches]
        self.addCleanup(lambda: [p.stop() for p in patches])
        offset, _ = sb._sample_posture(3, {}, ws)
        home, pick, noise = mocks
        np.testing.assert_array_equal(home.call_args[0][2], offset)
        np.testing.assert_array_equal(pick.call_args.kwargs["arm_offset"], offset)
        np.testing.assert_array_equal(noise.call_args.kwargs["arm_offset"], offset)

    def test_cache_key_follows_the_sampler_source(self):
        self.assertIn("data_pipeline/scene_bank.py", sb._SAMPLING_SOURCES)
        self.assertIn("controllers/pick_insert_expert.py", sb._SAMPLING_SOURCES)


@unittest.skipUnless(HAS_ASSETS, "needs third_party/openarm_mujoco")
class ResetVariationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from envs.openarm_insert import OpenArmInsertEnv
        cls.env = OpenArmInsertEnv(images=False, workspace=WORKSPACE)
        cls.ws = sb.workspace_for_env(cls.env)
        cls.home = np.array(cls.env.config["home_arm_rad"])
        row = next(iter(sb.iter_collection_bank(n=1, validation_fraction=0.1, ws=cls.ws, start=6100)))
        cls.seed = row["seed"]
        cls.sampled = sb.as_reset_options(row["options"])      # includes arm_offset_rad
        cls.zero = {k: v for k, v in cls.sampled.items() if k != "arm_offset_rad"}

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.warmup_s = self.env.config["warmup_s"]
        self.addCleanup(self.env.config.__setitem__, "warmup_s", self.warmup_s)

    # --- sampler on the real scene
    def test_sampled_options_only_add_the_offset(self):
        path = BASELINE_DIR / "baseline_options.json"
        if not path.exists():
            self.skipTest("baseline_options.json not available")
        baseline = json.loads(path.read_text())
        for seed in BASELINE_SEEDS:
            options, diag = sb.cached_sample_scene(int(seed), self.ws)
            self.assertEqual(set(options) - set(baseline[seed]), {"arm_offset_rad"})
            for key, value in baseline[seed].items():
                self.assertEqual(options[key], value, f"seed {seed} key {key}")
            offset = np.array(options["arm_offset_rad"])
            self.assertLessEqual(np.abs(offset).max(), self.ws.arm_offset_max_rad)
            self.assertEqual(options["arm_offset_rad"], sb.cached_sample_scene(int(seed), self.ws)[0]["arm_offset_rad"])
            self.assertLessEqual(diag["posture"]["tries"], sb.POSTURE_TRIES)
        offsets = [sb.cached_sample_scene(int(s), self.ws)[0]["arm_offset_rad"] for s in BASELINE_SEEDS]
        self.assertNotEqual(offsets[0], offsets[1])

    # --- env: warm-up
    def test_warmup_bookkeeping(self):
        obs, info = self.env.reset(seed=self.seed, options=self.zero)
        e = self.env
        self.assertEqual(e.warmup_steps, 44)
        self.assertEqual((e.elapsed, e.hold, e.outcome), (0., 0., None))
        self.assertEqual((e.peak_force, e.peak_robot_contact, e.peak_support_contact), (0., 0., 0.))
        self.assertAlmostEqual(obs["timestamp_s"], e.config["settle_s"] + 44 * e.dt, places=9)
        self.assertEqual(info["outcome"], "running")
        self.assertNotIn("executed_dt_s", info)
        np.testing.assert_array_equal(e.reset_state, e.data.qpos)
        np.testing.assert_array_equal(e.target[:7], e.last_action[:7])
        # grasped resets are unchanged: no warm-up
        e.reset(seed=0)
        self.assertEqual(e.warmup_steps, 0)

    def test_warmup_equals_the_old_survey_holds_bit_exact(self):
        e = self.env
        e.reset(seed=self.seed, options=self.zero)
        warm_qpos, warm_qvel = e.data.qpos.copy(), e.data.qvel.copy()
        e.config["warmup_s"] = 0.0                        # old behaviour: obs 0 straight after the settle
        e.reset(seed=self.seed, options=self.zero)
        self.assertEqual(e.warmup_steps, 0)
        for _ in range(44):                               # the old expert's survey holds
            e.step(np.r_[e.target[:7], e.config["grip_open_travel_m"]])
        np.testing.assert_array_equal(e.data.qpos, warm_qpos)
        np.testing.assert_array_equal(e.data.qvel, warm_qvel)

    def test_warmup_state_matches_pre_change_baseline(self):
        path = BASELINE_DIR / "baseline_6100.npz"
        if not path.exists():
            self.skipTest("baseline_6100.npz not available")
        base = np.load(path)
        options = json.loads(str(base["reset_options"]))
        self.assertNotIn("arm_offset_rad", options)
        self.env.reset(seed=int(base["seed"]), options=options)
        np.testing.assert_array_equal(self.env.data.qpos, base["qpos"][44])
        np.testing.assert_array_equal(self.env.data.qvel, base["qvel"][44])

    # --- expert
    def test_first_action_is_a_pick_action_not_a_hold(self):
        from controllers.pick_insert_expert import PickInsertExpert
        e = self.env
        e.reset(seed=self.seed, options=self.sampled)
        expert = PickInsertExpert(e, rng=np.random.default_rng(self.seed), detector_mode="noisy",
                                  probe_offset_y_m=0.0, fault=None)
        action = expert.action()
        self.assertEqual(expert.phase, "transit_pick")
        self.assertIsNone(expert.failure)
        self.assertIsNotNone(expert.detection_error)
        self.assertFalse(np.array_equal(action, e.target))
        self.assertAlmostEqual(expert.phase_time, e.dt)

    def test_expert_without_warmup_keeps_the_timed_survey(self):
        from controllers.pick_insert_expert import PickInsertExpert
        e = self.env
        e.config["warmup_s"] = 0.0
        e.reset(seed=self.seed, options=self.zero)
        expert = PickInsertExpert(e, rng=np.random.default_rng(self.seed), detector_mode="noisy",
                                  probe_offset_y_m=0.0, fault=None)
        phases = []
        for _ in range(46):
            action = expert.action()
            phases.append(expert.phase)
            self.assertTrue(np.array_equal(action, np.r_[e.target[:7], e.config["grip_open_travel_m"]])
                            or phases[-1] != "survey")
        self.assertEqual(phases[:44], ["survey"] * 44)
        self.assertEqual(phases[44], "transit_pick")        # latched on the 45th call, which still holds

    def test_wrong_grasp_fault_starts_on_the_first_action(self):
        from controllers import faults
        from controllers.pick_insert_expert import PickInsertExpert
        e = self.env
        e.reset(seed=self.seed, options=self.sampled)
        spec = faults.draw_fault(self.seed, ("wrong_grasp",))
        expert = PickInsertExpert(e, rng=np.random.default_rng(self.seed), detector_mode="noisy",
                                  probe_offset_y_m=0.0, fault=spec)
        expert.action()
        self.assertEqual(expert.fault_state, "active")
        self.assertEqual(expert.fault_events["fault_start"]["step"], 0)
        self.assertEqual(expert.fault_events["fault_start"]["phase"], "transit_pick")
        self.assertEqual(expert.phase, "transit_pick")
        self.assertEqual(expert.label, "fault")

    # --- env: start posture
    def test_start_posture_is_home_plus_offset_and_collision_free(self):
        e = self.env
        offset = np.array(self.sampled["arm_offset_rad"])
        self.assertGreater(np.abs(offset).max(), 0.)
        e.reset(seed=self.seed, options=self.sampled)        # raises on a reset collision
        np.testing.assert_allclose(e.reset_qpos[e.qa["right"]], self.home + offset, atol=1e-12)
        self.assertLessEqual(e.info["robot_unwanted_contact_n"], e.config["robot_contact_abort_n"])
        # the left arm is untouched and the target started at the offset posture
        np.testing.assert_array_equal(e.reset_qpos[e.qa["left"]], np.zeros(7))
        np.testing.assert_allclose(e.target[:7], self.home + offset, atol=1e-12)
        e.reset(seed=self.seed, options=self.zero)
        np.testing.assert_allclose(e.reset_qpos[e.qa["right"]], self.home, atol=1e-12)

    def test_invalid_offsets_raise(self):
        e = self.env
        base = dict(self.zero)
        bad = {"too_big": [0.2] + [0.] * 6, "wrong_len": [0.] * 6, "nan": [np.nan] + [0.] * 6,
               "joint_limit_margin": [0., -0.14] + [0.] * 5,        # joint2 home 0, lower limit -0.1745
               "inf": [np.inf] + [0.] * 6}
        for name, value in bad.items():
            with self.assertRaises(ValueError, msg=name):
                e.reset(seed=self.seed, options={**base, "arm_offset_rad": value})
        e.reset(seed=self.seed, options={**base, "arm_offset_rad": [0.05] * 7})


if __name__ == "__main__":
    unittest.main()
