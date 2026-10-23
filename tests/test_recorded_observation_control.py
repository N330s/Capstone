import unittest
import numpy as np

from scripts.rollout_smolvla import policy_observation
from scripts.test_recorded_observation_control import check_report
from learning.vla.rollout_video import video_title
from scripts.report_recorded_observation_control import verify_input_trace


class RecordedObservationTests(unittest.TestCase):
    def setUp(self):
        self.live = {"state": np.zeros(16), "instruction": "Pick and insert",
                     "images": {camera: np.zeros((2, 3, 3), np.uint8) for camera in ("scene_rgb", "wrist_rgb")}}
        self.recording = {"observation_state": np.ones((4, 16)),
                          **{"image_" + camera: np.full((4, 2, 3, 3), 123, np.uint8)
                             for camera in ("scene_rgb", "wrist_rgb")}}

    def test_live_mode_is_unchanged(self):
        self.assertIs(policy_observation(self.live, None, 0, "policy"), self.live)

    def test_recorded_state_preserves_live_images_and_actual_state(self):
        actor = policy_observation(self.live, self.recording, 1, "recorded-state")
        np.testing.assert_array_equal(actor["state"], np.ones(16))
        self.assertIs(actor["images"], self.live["images"])
        actor["state"][:] = 3
        np.testing.assert_array_equal(self.live["state"], np.zeros(16))
        np.testing.assert_array_equal(self.recording["observation_state"], np.ones((4, 16)))

    def test_full_teacher_forcing_uses_only_recorded_actor_features(self):
        self.recording["action_applied"] = np.full((3, 8), 999.)
        self.recording["qpos"] = np.full((4, 100), 999.)
        actor = policy_observation(self.live, self.recording, 2, "recorded-observations")
        self.assertEqual(set(actor), {"state", "images", "instruction"})
        for camera in actor["images"]:
            self.assertTrue(np.all(actor["images"][camera] == 123))
            actor["images"][camera][:] = 0
            self.assertTrue(np.all(self.recording["image_" + camera] == 123))

    def test_terminal_observation_and_invalid_modes_rejected(self):
        for index in (-1, 3, 4, True):
            with self.assertRaises(IndexError):
                policy_observation(self.live, self.recording, index, "recorded-state")
        with self.assertRaises(ValueError):
            policy_observation(self.live, self.recording, 0, "teleport")

    def test_assisted_video_cannot_be_labeled_expert_replay(self):
        for mode in ("recorded-state", "recorded-observations"):
            self.assertIn("ASSISTED", video_title(mode))
            self.assertNotIn("expert command replay", video_title(mode))

    def test_assisted_report_must_not_claim_autonomy_or_teleportation(self):
        report = {"mode": "recorded-state", "checkpoint_weights_sha256": "w", "source_data_sha256": "d",
                  "source_episode": "ep", "execution_prefix": 5, "policy_seed": 1000,
                  "pipeline_passed": True, "simulator_state_forced_to_recording": False,
                  "recorded_input_assistance": True, "autonomous_task_success": False}
        check_report(report, "recorded-state", "w", "d", "ep", 5, 1000)
        for key in ("autonomous_task_success", "simulator_state_forced_to_recording"):
            with self.assertRaises(ValueError):
                check_report({**report, key: True}, "recorded-state", "w", "d", "ep", 5, 1000)

    def test_expert_control_must_physically_succeed_and_match_recording(self):
        report = {"mode": "replay", "checkpoint_weights_sha256": "w", "source_data_sha256": "d",
                  "source_episode": "ep", "execution_prefix": 5, "policy_seed": 1000,
                  "pipeline_passed": True, "simulator_state_forced_to_recording": False,
                  "recorded_input_assistance": False, "autonomous_task_success": False,
                  "task_success": True, "grasp_established": True, "replay_errors": {"qpos_max": 0., "qvel_max": 0.}}
        check_report(report, "replay", "w", "d", "ep", 5, 1000)
        with self.assertRaises(ValueError):
            check_report({**report, "task_success": False}, "replay", "w", "d", "ep", 5, 1000)

    def test_report_readback_rejects_shifted_or_incomplete_inputs(self):
        reference = np.arange(64).reshape(4, 16)
        chunks = [{"recorded_observation_index": 0, "start_step": 0,
                   "policy_input_state": reference[0].tolist(), "executed_steps": 3}]
        self.assertEqual(verify_input_trace(chunks, reference, 3), 1)
        with self.assertRaises(ValueError):
            verify_input_trace(chunks, reference, 2)
        chunks[0]["policy_input_state"] = reference[1].tolist()
        with self.assertRaises(ValueError):
            verify_input_trace(chunks, reference, 3)


if __name__ == "__main__":
    unittest.main()
