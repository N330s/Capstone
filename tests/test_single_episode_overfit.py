import unittest
from scripts.overfit_smolvla_episode import episode_span, probe_indices
from learning.vla.smolvla_data import CONTRACT, LEROBOT_VERSION
from scripts.report_single_episode_overfit import audit_updates


class SingleEpisodeTests(unittest.TestCase):
    def receipt(self):
        return {"contract": CONTRACT, "lerobot_version": LEROBOT_VERSION, "datasets": {
            "train": {"episodes": [{"episode": "a", "steps": 7}, {"episode": "b", "steps": 11}]},
            "validation": {"episodes": [{"episode": "heldout", "steps": 4}]}}}

    def test_global_span_is_exact(self):
        self.assertEqual(episode_span(self.receipt(), "b")[:2], (7, 18))

    def test_validation_cannot_be_overfit_target(self):
        with self.assertRaises(ValueError):
            episode_span(self.receipt(), "heldout")

    def test_split_leakage_rejected(self):
        receipt = self.receipt()
        receipt["datasets"]["validation"]["episodes"][0]["episode"] = "a"
        with self.assertRaises(ValueError):
            episode_span(receipt, "b")

    def test_probes_stay_inside_selected_parent(self):
        self.assertEqual(probe_indices(7, 9), [7, 8])
        self.assertTrue(all(7 <= i < 18 for i in probe_indices(7, 18)))

    def test_exposure_audit_rejects_other_parent_or_nonfinite(self):
        row = {"type": "update", "step": 1, "indices": [7, 8], "loss": .01, "grad_norm": .2}
        self.assertEqual(audit_updates([row], 7, 18, 1, 2)["sampled_anchors"], 2)
        for bad in ({**row, "indices": [6, 7]}, {**row, "loss": float("nan")}, {**row, "step": 2}):
            with self.assertRaises(ValueError):
                audit_updates([bad], 7, 18, 1, 2)
