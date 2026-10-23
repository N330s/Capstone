import unittest

from scripts.resume_smolvla_campaign import remaining_budget, planned_updates


class CampaignRestartTests(unittest.TestCase):
    def test_restart_debits_prior_training_not_a_fresh_four_hours(self):
        self.assertEqual(remaining_budget(1000, 100), 13240)
        self.assertLessEqual(1000 + 100 + remaining_budget(1000, 100), 14400)
        eight_hour_budget = remaining_budget(1400, 500, ceiling=28800)
        self.assertEqual(eight_hour_budget, 26840)
        self.assertLessEqual(1400 + 500 + eight_hour_budget, 28800)
        for previous, retry in ((-1, 0), (float("nan"), 0), (11000, 100)):
            with self.assertRaises(ValueError):
                remaining_budget(previous, retry)

    def test_cosine_horizon_matches_measured_remaining_window(self):
        steps, median = planned_updates([2.] * 100, 13000)
        self.assertEqual(steps, 6200)
        self.assertEqual(median, 2.)
        for bad in ([], [0.], [float("nan")]):
            with self.assertRaises(ValueError):
                planned_updates(bad, 13000)


if __name__ == "__main__":
    unittest.main()
