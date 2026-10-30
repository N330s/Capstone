"""quality_check on synthetic step lists (robot-free)."""
import unittest

from data_pipeline.quality import quality_check


def step(phase, contacts=2, latch=True, drops=0, z=0.0, label="expert"):
    return {"phase": phase, "plug_finger_contacts": contacts, "grasp_established": latch,
            "drops": drops, "plug_z": z, "label": label}


def clean(n_lift=10, lift=0.08):
    s = [step("transit_pick", 0, False), step("descend", 0, False), step("close", 2, False)]
    s += [step("close", 2, True)]
    s += [step("lift", z=lift * (i + 1) / n_lift) for i in range(n_lift)]
    s += [step("transit_place", z=lift)] * 5 + [step("insert", z=lift)] * 5 + [step("hold", z=lift)] * 3
    return s


class QualityTests(unittest.TestCase):
    def test_clean_passes(self):
        q = quality_check(clean(), "success", start_z=0.0)
        self.assertTrue(q["ok"], q)
        so = q["stage_outcomes"]
        self.assertTrue(so["pickup"]["success"] and so["carry"]["success"] and so["insertion"]["success"])
        self.assertEqual(so["pickup"]["latch_step"], 3)
        self.assertAlmostEqual(so["pickup"]["lift_height_m"], 0.08)

    def test_start_z_defaults_to_first_step(self):
        self.assertTrue(quality_check(clean(), "success")["ok"])

    def test_drop(self):
        s = clean(); s[-4] = step("transit_place", 2, False, drops=1, z=0.08)
        q = quality_check(s, "success", start_z=0.0)
        self.assertFalse(q["ok"]); self.assertIn("quality_drop", q["reasons"])
        self.assertEqual(q["stage_outcomes"]["carry"]["drops"], 1)

    def test_contact_lost_during_carry(self):
        s = clean(); s[12] = step("lift", 1, True, z=0.04)
        q = quality_check(s, "success", start_z=0.0)
        self.assertEqual(q["reasons"], ["quality_contact_lost"])
        self.assertEqual(q["stage_outcomes"]["carry"]["contact_loss_steps"], 1)

    def test_contact_lost_at_hold(self):
        s = clean(); s[-1] = step("hold", 0, True, z=0.08)
        self.assertIn("quality_contact_lost", quality_check(s, "success", start_z=0.0)["reasons"])

    def test_low_lift(self):
        q = quality_check(clean(lift=0.01), "success", start_z=0.0)
        self.assertEqual(q["reasons"], ["quality_low_lift"])
        self.assertFalse(q["stage_outcomes"]["pickup"]["success"])

    def test_no_latch(self):
        s = [dict(x, grasp_established=False) for x in clean()]
        q = quality_check(s, "success", start_z=0.0)
        self.assertEqual(q["reasons"][0], "quality_no_grasp")

    def test_abort_outcome(self):
        q = quality_check(clean(), "wall_force_abort", start_z=0.0)
        self.assertFalse(q["ok"])
        self.assertFalse(q["stage_outcomes"]["insertion"]["success"])
        self.assertEqual(q["stage_outcomes"]["insertion"]["outcome"], "wall_force_abort")
        self.assertTrue(q["stage_outcomes"]["pickup"]["success"])

    def test_fault_empty_closes_before_lift_are_fine(self):
        pre = [step("descend", 0, False, label="expert"), step("fault_close", 0, False, label="fault"),
               step("fault_close", 0, False, label="fault"), step("reopen", 0, False, label="recovery"),
               step("rise", 0, False, label="recovery")]
        q = quality_check(pre + clean(), "success", start_z=0.0)
        self.assertTrue(q["ok"], q)

    def test_no_lift_phase(self):
        s = [x for x in clean() if x["phase"] != "lift"]
        q = quality_check(s, "success", start_z=0.0)
        self.assertIn("quality_low_lift", q["reasons"])
        self.assertFalse(q["stage_outcomes"]["carry"]["success"])


if __name__ == "__main__":
    unittest.main()
