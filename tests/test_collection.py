import unittest
import numpy as np
from data_pipeline.collection_bank import collection_bank, evaluation_bank
from envs.openarm_insert import OpenArmInsertEnv


class CollectionTests(unittest.TestCase):
    def test_reserved_bank_is_disjoint_and_repeatable(self):
        train, heldout = collection_bank(), evaluation_bank()
        self.assertEqual(len(train), 10)
        self.assertEqual(len(heldout), 100)
        self.assertEqual(heldout, evaluation_bank())
        self.assertFalse({r["seed"] for r in train} & {r["seed"] for r in heldout})
        self.assertFalse({tuple(sorted(r["options"].items())) for r in train}
                         & {tuple(sorted(r["options"].items())) for r in heldout})

    def test_varied_reset_repeatability_and_validation(self):
        env = OpenArmInsertEnv(images=False)
        try:
            options = collection_bank()[-1]["options"]
            env.reset(seed=17, options=options)
            qpos = env.data.qpos.copy()
            env.step(env.target.copy())
            env.reset(seed=17, options=options)
            np.testing.assert_array_equal(env.data.qpos, qpos)
            for options in ({"roll_deg": 3}, {"offset_x_m": .004}, {"yaw_deg": np.nan}):
                before = env.data.qpos.copy()
                with self.assertRaises(ValueError):
                    env.reset(options=options)
                np.testing.assert_array_equal(env.data.qpos, before)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
