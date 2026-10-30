"""Pilot control contract of saved episodes, label arrays and strict replay (robot scene, cable30;
needs third_party assets)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_pipeline.episodes import save_episode, validate_episode, TORQUE_KEYS
from data_pipeline.replay import replay_episode
from data_pipeline.scene_bank import as_reset_options, iter_collection_bank, workspace_for_env
from envs.openarm_insert import OpenArmInsertEnv

CABLE30 = "configs/workspace_v2_cable30.json"
STEPS = 3


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = OpenArmInsertEnv(images=True, workspace=CABLE30)
        ws = workspace_for_env(cls.env)
        cls.row = next(iter(iter_collection_bank(n=1, validation_fraction=0.1, ws=ws, start=6100)))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def rollout(self):
        env = self.env
        obs, info = env.reset(seed=self.row["seed"], options=as_reset_options(self.row["options"]))
        hold = np.r_[env.target[:7], env.target[7]]
        d = dict(observations=[obs], infos=[info], states=[env.data.qpos.copy()], velocities=[env.data.qvel.copy()],
                 requested=[], applied=[], rewards=[], terminated=[], truncated=[], durations=[])
        for i in range(STEPS):
            obs, reward, term, trunc, info = env.step(hold)
            d["observations"].append(obs); d["infos"].append(info)
            d["states"].append(env.data.qpos.copy()); d["velocities"].append(env.data.qvel.copy())
            d["requested"].append(hold.copy()); d["applied"].append(info["applied_action"].copy())
            d["rewards"].append(reward); d["durations"].append(info["executed_dt_s"])
            d["terminated"].append(i == STEPS - 1)       # validate_episode wants a finished, successful episode
            d["truncated"].append(False)
        d["infos"][-1] = dict(d["infos"][-1], outcome="success")
        return d

    def save(self, d, path, **kw):
        return save_episode(path, d["observations"], d["requested"], d["applied"], d["states"], d["velocities"],
                            d["rewards"], d["terminated"], d["truncated"], d["durations"], ["hold"] * STEPS,
                            d["infos"], self.env.manifest(), **kw)

    def test_contract_arrays_and_metadata(self):
        d = self.rollout()
        vis = np.array([[True, True]] * (STEPS + 1))
        with tempfile.TemporaryDirectory() as tmp:
            ep = Path(tmp) / "ep"
            meta = self.save(d, ep, visibility=vis, quality={"ok": True}, stage_outcomes={"pickup": {}},
                             replay={"passed": True})
            with np.load(ep / "episode.npz", allow_pickle=False) as a:
                self.assertEqual(a["image_scene_rgb"].shape[0], STEPS + 1)
                self.assertEqual(a["image_wrist_rgb"].shape[0], STEPS + 1)
                self.assertEqual(a["observation_state"].shape, (STEPS + 1, 16))
                self.assertEqual(a["action_applied"].shape, (STEPS, 8))
                self.assertEqual(a["action_requested"].shape, (STEPS, 8))
                dt = a["executed_dt_s"]
                np.testing.assert_allclose(dt[:-1], 0.02)          # terminal step may end mid-interval
                self.assertLessEqual(dt[-1], 0.02 + 1e-12)
                left = a["observation_state"][:, :8]
                self.assertLess(np.ptp(left, axis=0).max(), 1e-3)             # parked
                for key in ("depth_scene_mm", "depth_wrist_mm", "image_chest_rgb") + TORQUE_KEYS:
                    self.assertIn(key, a.files)
                self.assertEqual(a["joint_torque_applied_nm"].shape, (STEPS, 14))
                self.assertEqual(a["observation_state"].shape[1], 16)       # torque/depth are not in the state
                self.assertTrue(a["label_mask"].all())
                self.assertNotIn("step_label", a.files)                     # clean episodes stay unlabelled
                np.testing.assert_array_equal(a["action_label"], a["action_applied"])
                self.assertEqual(a["visibility_pilot"].dtype, bool)
                self.assertEqual(a["visibility_pilot"].shape, (STEPS + 1, 2))
            c = meta["pilot_contract"]
            self.assertEqual(c["cameras"], ["scene_rgb", "wrist_rgb"])
            self.assertEqual((c["state"], c["action"], c["rate_hz"]), (16, 8, 50))
            self.assertEqual(len(c["action_names"]), 8)
            self.assertIn("terminal", c["timing_note"])
            self.assertEqual(c["label_arrays"], ["action_label", "label_mask"])
            self.assertEqual(meta["visibility_pilot_columns"], ["scene_rgb", "wrist_rgb"])
            self.assertEqual(meta["replay"], {"passed": True})
            validate_episode(ep)

    def test_fault_labels_masked(self):
        d = self.rollout()
        labels = ["expert", "fault", "recovery"]
        with tempfile.TemporaryDirectory() as tmp:
            ep = Path(tmp) / "ep"
            self.save(d, ep, labels=labels, fault={"kind": "wrong_grasp"})
            with np.load(ep / "episode.npz", allow_pickle=False) as a:
                self.assertEqual(a["label_mask"].tolist(), [True, False, True])
                self.assertTrue(np.isnan(a["action_label"][1]).all())
                np.testing.assert_array_equal(a["action_label"][[0, 2]], a["action_applied"][[0, 2]])
                np.testing.assert_array_equal(a["step_label"], labels)
            validate_episode(ep)
            # tamper: a non-NaN label on a masked step must be caught
            npz = ep / "episode.npz"
            with np.load(npz, allow_pickle=False) as a:
                arrays = {k: a[k] for k in a.files}
            arrays["action_label"] = arrays["action_applied"].copy()
            np.savez_compressed(npz, **arrays)
            import hashlib
            meta = json.loads((ep / "metadata.json").read_text())
            meta["data_sha256"] = hashlib.sha256(npz.read_bytes()).hexdigest()
            (ep / "metadata.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                validate_episode(ep)

    def test_replay_episode_reproduces_saved_commands(self):
        d = self.rollout()
        env = self.env
        with tempfile.TemporaryDirectory() as tmp:
            ep = Path(tmp) / "ep"
            meta = self.save(d, ep)
            # The saved flags mark the last step terminal (validate_episode needs that); make the replay
            # env report the same flag so only the physics is compared.
            real_step, count = env.step, [0]

            def step(action):
                out = list(real_step(action))
                count[0] += 1
                out[2] = count[0] == STEPS
                return tuple(out)
            env.step = step
            try:
                result = replay_episode(env, ep)
            finally:
                del env.step
            self.assertLessEqual(result["max_qpos_error"], 1e-8)
            self.assertLessEqual(result["max_qvel_error"], 1e-7)
            self.assertEqual(meta["steps"], STEPS)


if __name__ == "__main__":
    unittest.main()
