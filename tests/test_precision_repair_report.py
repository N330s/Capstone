import copy
import unittest
import numpy as np

from scripts.report_precision_repair import audit_rollout, motion_landmarks


class PrecisionReportTests(unittest.TestCase):
    def setUp(self):
        self.grasp = {"revision": "physical-contact-settle-v1", "extension_sha256": "extension",
                      "physics_or_guard_thresholds_changed": False, "settle_s": .1}
        self.report = {"recorded_input_assistance": False, "simulator_state_forced_to_recording": False,
            "pipeline_passed": True, "reset_state_max_error": 0., "mode": "policy",
            "policy_seed": 1000, "execution_prefix": 5, "checkpoint_weights_sha256": "base",
            "source_data_sha256": "source", "grasp_metrics": self.grasp.copy(),
            "actuation_filter": {"gripper_resolution_mm": 0., "reference_or_phase_inputs": False,
                                 "arm_targets_changed": False}, "expert_overlay_weights_sha256": "overlay"}

    def audit(self, report, label="repaired_prefix5"):
        return audit_rollout(report, label=label, base_sha="base", source_sha="source",
                             overlay_sha="overlay", grasp_metrics=self.grasp)

    def test_repaired_condition_and_declared_prefix(self):
        self.audit(self.report)
        with self.assertRaises(ValueError):
            self.audit(self.report, "repaired_prefix25")
        self.audit({**self.report, "execution_prefix": 25}, "repaired_prefix25")
        self.audit({**self.report, "execution_prefix": 25, "expert_overlay_weights_sha256": None}, "original_prefix25")

    def test_assistance_wrong_reset_or_model_is_rejected(self):
        for key, bad in (("recorded_input_assistance", True), ("reset_state_max_error", 1e-7),
                         ("policy_seed", 1), ("source_data_sha256", "different"),
                         ("expert_overlay_weights_sha256", None)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit({**self.report, key: bad})

    def test_metric_or_command_filter_changes_are_rejected(self):
        for key, bad in (("grasp_metrics", {**self.grasp, "extension_sha256": "other"}),
                         ("actuation_filter", {**self.report["actuation_filter"], "arm_targets_changed": True})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit({**self.report, key: bad})

    def test_expert_requires_exact_successful_replay_without_overlay(self):
        expert = copy.deepcopy(self.report)
        expert.update(mode="replay", expert_overlay_weights_sha256=None, task_success=True,
                      grasp_established=True, replay_errors={"qpos_max": 0., "qvel_max": 0.})
        self.audit(expert, "expert")
        with self.assertRaises(ValueError):
            self.audit({**expert, "replay_errors": {"qpos_max": 1e-8}}, "expert")

    def test_motion_landmarks_do_not_equate_command_index_with_progress(self):
        reference = np.zeros((4, 16))
        reference[:, 8:15] = np.arange(4)[:, None]
        reference[:, 15] = [.035, .035, .035, .011]
        actions = np.zeros((3, 8))
        actions[:, 7] = .035
        state = reference[[0, 1, 1]].copy()
        state[-1, 15] = .029
        commands = actions[:2].copy()
        commands[-1, 7] = .028
        result = motion_landmarks({"observation_state": state, "action_requested": commands}, reference, actions)
        self.assertEqual(result["first_measured_below_30mm"]["trace_index"], 2)
        self.assertEqual(result["first_measured_below_30mm"]["nearest_expert_right_joint_index"], 1)
        self.assertAlmostEqual(result["first_measured_below_30mm"]["nearest_expert_target_gripper_mm"], 35.)
