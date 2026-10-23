import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from learning.vla.gripper_feedback import open_hold_anchors, perturbed_gripper, quantize_finger_command
from learning.vla.expert_overlay import BASE_FILES, FORMAT, verify_binding, save_overlay, load_overlay
from learning.vla.smolvla_data import sha256
from scripts.report_gripper_feedback import audit_updates


class FeedbackTests(unittest.TestCase):
    def test_only_steady_open_targets_are_eligible(self):
        actions = np.zeros((100, 8))
        actions[:70, 7] = .035
        actions[70:, 7] = .011
        states = np.zeros((101, 16))
        states[:, 15] = .035
        self.assertEqual(open_hold_anchors(states, actions, 50), list(range(21)))
        states[0, 15] = .02
        self.assertEqual(open_hold_anchors(states, actions, 50), list(range(1, 21)))

    def test_sensor_noise_is_bounded_and_does_not_force_open(self):
        self.assertAlmostEqual(perturbed_gripper(.035, -.002), .033)
        self.assertAlmostEqual(perturbed_gripper(.035, .002), .037)
        self.assertAlmostEqual(perturbed_gripper(.044, .002), .044)
        with self.assertRaises(ValueError):
            perturbed_gripper(.035, .003)

    def test_resolution_filter_changes_only_gripper_and_retains_close_decision(self):
        command = np.arange(8, dtype=float)
        command[7] = .0354
        result = quantize_finger_command(command, .001)
        np.testing.assert_array_equal(result[:7], command[:7])
        self.assertAlmostEqual(result[7], .035)
        command[7] = .0112
        self.assertAlmostEqual(quantize_finger_command(command, .001)[7], .011)
        self.assertAlmostEqual(command[7], .0112)
        with self.assertRaises(ValueError):
            quantize_finger_command(command, .002)

    def test_overlay_binding_requires_exact_base_and_completed_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            for name in BASE_FILES:
                (base / name).write_bytes(name.encode())
            manifest = {"format": FORMAT, "complete": True,
                        "base_sha256": {name: sha256(base / name) for name in BASE_FILES}}
            verify_binding(manifest, base)
            with self.assertRaises(ValueError):
                verify_binding({**manifest, "complete": False}, base)
            (base / "model.safetensors").write_bytes(b"different")
            with self.assertRaises(ValueError):
                verify_binding(manifest, base)

    def test_overlay_roundtrip_reuses_frozen_parameters_and_detects_corruption(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "base"
            base.mkdir()
            for name in BASE_FILES:
                (base / name).write_bytes(name.encode())
            model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
            for parameter in model[1].parameters():
                parameter.requires_grad_(False)
            original = {name: p.clone() for name, p in model.named_parameters()}
            target = Path(folder) / "overlay"
            manifest = save_overlay(model, base, target)
            self.assertEqual(set(manifest["shapes"]), {"0.weight", "0.bias"})
            with torch.no_grad():
                model[0].weight.add_(1)
                model[0].bias.add_(2)
            load_overlay(model, base, target)
            self.assertTrue(all(torch.equal(p, original[name]) for name, p in model.named_parameters()))
            with (target / "expert.safetensors").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaises(ValueError):
                load_overlay(model, base, target)

    def test_independent_batch_audit_rejects_wrong_parent_or_noise_in_control(self):
        rows = [{"step": i + 1, "indices": [1] * 8, "loss": .001, "grad_norm": .1,
                 "gripper_sensor_offsets_m": [0.] * 4} for i in range(800)]
        self.assertEqual(audit_updates(rows, [1], 10, False)["anchors"], 6400)
        rows[0] = {**rows[0], "indices": [11] * 8}
        with self.assertRaises(ValueError):
            audit_updates(rows, [1], 10, False)
        rows[0] = {**rows[1], "step": 1, "gripper_sensor_offsets_m": [.001] * 4}
        with self.assertRaises(ValueError):
            audit_updates(rows, [1], 10, False)
        self.assertEqual(audit_updates(rows, [1], 10, True)["perturbed_anchors"], 4)
