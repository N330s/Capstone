import unittest

from learning.vla.pickup_evaluation import PickupGate, select_validation_parents, wilson_interval, audit_pickup_trace


class PickupEvaluationTests(unittest.TestCase):
    def sample(self, gate, **changes):
        values = dict(plug_z_m=.35, grasp=True, contacts=2, drops=0, safe=True, dt=.02)
        return gate.update(**(values | changes))

    def test_table_contact_or_airborne_without_grasp_cannot_pass(self):
        for values in ({"plug_z_m": .325}, {"grasp": False}, {"contacts": 1}, {"drops": 1}, {"safe": False}):
            gate = PickupGate(.32)
            for _ in range(100):
                self.assertFalse(self.sample(gate, **values))

    def test_requires_one_full_second_after_threshold_crossing(self):
        gate = PickupGate(.32)
        for _ in range(50):
            self.assertFalse(self.sample(gate))
        self.assertTrue(self.sample(gate))
        self.assertAlmostEqual(gate.maximum_s, 1.)

    def test_contact_loss_resets_continuity_and_real_dt_counts(self):
        gate = PickupGate(.32)
        for _ in range(40):
            self.sample(gate)
        self.sample(gate, contacts=0)
        for _ in range(50):
            self.assertFalse(self.sample(gate))
        self.assertTrue(self.sample(gate))

    def test_selection_is_order_independent_and_rejects_split_leakage(self):
        heldout = [dict(episode=f"e{i:02}", split="validation") for i in range(20)]
        receipt = {"datasets": {"train": {"episodes": [{"episode": "train"}]}, "validation": {"episodes": heldout}}}
        self.assertEqual([r["episode"] for r in select_validation_parents(receipt)], ["e00", "e05", "e10", "e14", "e19"])
        heldout.reverse()
        self.assertEqual(select_validation_parents(receipt)[0]["episode"], "e00")
        receipt["datasets"]["train"]["episodes"].append({"episode": "e00"})
        with self.assertRaises(ValueError):
            select_validation_parents(receipt)

    def test_small_bank_is_not_strong_generalization_proof(self):
        low, high = wilson_interval(5, 5)
        self.assertLess(low, .6)
        self.assertAlmostEqual(high, 1.)

    def test_saved_motion_audit_rejects_false_positive_report(self):
        import numpy as np
        n = 52
        trace = {"qpos": np.zeros((n + 1, 7)), "observation_state": np.zeros((n + 1, 16)),
            "action_requested": np.zeros((n, 8)), "policy_action_raw": np.zeros((n, 8)),
            "executed_dt_s": np.full(n, .02)}
        trace["qpos"][:, 2] = .32
        diagnostics = [dict(grasp_established=True, plug_finger_contacts=2, drops=0, outcome="running") for _ in range(n + 1)]
        report = {"steps": n, "outcome": "step_budget_exhausted", "pickup_evaluation": {
            **PickupGate(.32).definition(), "plug_qpos_address": 0, "passed": False,
            "maximum_consecutive_carry_s": 0., "maximum_lift_m": 0.}}
        self.assertFalse(audit_pickup_trace(report, trace, diagnostics)["pickup_success"])
        report["pickup_evaluation"]["passed"] = True
        report["outcome"] = "pickup_success"
        with self.assertRaises(ValueError):
            audit_pickup_trace(report, trace, diagnostics)
