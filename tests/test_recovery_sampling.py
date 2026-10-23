import copy
import random
import unittest

import numpy as np

from learning.vla.recovery_sampling import CorrectionSampler, correction_categories, correction_probes


class CorrectionSamplingTests(unittest.TestCase):
    def test_only_real_recovery_corrections_selected(self):
        labels = np.array(["expert", "fault", "recovery", "recovery", "recovery"])
        states, actions = np.zeros((6, 16)), np.zeros((5, 8))
        states[:, 15] = .02
        actions[:, 7] = [.035, .035, .035, .011, .0201]
        self.assertEqual(correction_categories(labels, states, actions).tolist(),
                         ["expert", "fault", "active_reopening", "other_recovery", "other_recovery"])
        for threshold in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                correction_categories(labels, states, actions, threshold)

    def test_every_batch_has_quota_and_rng_resume_is_exact(self):
        pools = {"active_reopening": [0, 1], "other_recovery": [2, 3], "expert": list(range(4, 104))}
        sampler, rng = CorrectionSampler(pools, 104), random.Random(3)
        for size in (4, 7, 8, 16):
            saved = rng.getstate()
            batch = sampler.sample(rng, size)
            exposure = sampler.exposure(batch)
            self.assertGreaterEqual(exposure["active_reopening"], size / 4)
            self.assertEqual(exposure["other_recovery"], size // 4)
            self.assertEqual(sum(exposure.values()), size)
            rng.setstate(saved)
            self.assertEqual(batch, sampler.sample(rng, size))

    def test_missing_duplicate_or_noninteger_partition_rejected(self):
        pools = {"active_reopening": [0], "other_recovery": [1], "expert": [2]}
        for key, values in (("expert", []), ("expert", [1]), ("expert", [2.])):
            changed = copy.deepcopy(pools)
            changed[key] = values
            with self.assertRaises(ValueError):
                CorrectionSampler(changed, 3)
        with self.assertRaises(ValueError):
            CorrectionSampler(pools, 3).sample(random.Random(1), 3)

    def test_probes_include_all_available_parents_and_only_requested_category(self):
        records = [{"episode": "a0", "parent_episode": "a", "steps": 3},
                   {"episode": "b0", "parent_episode": "b", "steps": 4}]
        index = {"partitions": {"validation": {"active_reopening": [0, 1, 4, 6]}}}
        probes = correction_probes(index, records, "validation", "active_reopening")
        self.assertEqual({r["parent_episode"] for r in probes}, {"a", "b"})
        self.assertEqual({r["index"] for r in probes}, {0, 1, 4, 6})


if __name__ == "__main__":
    unittest.main()
