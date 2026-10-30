"""Ray-cast plug/socket visibility probe (robot scene; needs third_party assets)."""
import unittest

import mujoco
import numpy as np

from data_pipeline.scene_bank import as_reset_options, iter_collection_bank, workspace_for_env
from data_pipeline.visibility import PILOT_CAMERAS, VisibilityProbe, pickup_coverage
from envs.openarm_insert import OpenArmInsertEnv

CABLE30 = "configs/workspace_v2_cable30.json"


class VisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = OpenArmInsertEnv(images=False, workspace=CABLE30)
        ws = workspace_for_env(cls.env)
        cls.row = next(iter(iter_collection_bank(n=1, validation_fraction=0.1, ws=ws, start=6100)))
        cls.env.reset(seed=cls.row["seed"], options=as_reset_options(cls.row["options"]))
        cls.probe = VisibilityProbe(cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_reset_returns_bools_per_pilot_camera(self):
        for flags in (self.probe.plug(), self.probe.socket()):
            self.assertEqual(set(flags), set(PILOT_CAMERAS))
            self.assertTrue(all(isinstance(v, bool) for v in flags.values()))
        self.assertIsInstance(self.probe.any_plug(), bool)

    def test_plug_out_of_every_view_is_not_visible(self):
        env = self.env
        qpos = env.data.qpos.copy()
        try:
            # place_plug also lays the cable and refuses poses out of its reach, so move the free
            # joint directly: only the pose matters for visibility.
            env.data.qpos[env.plug_qadr:env.plug_qadr + 3] = (-0.6, 0.0, 0.5)
            mujoco.mj_forward(env.model, env.data)
            self.assertEqual(self.probe.plug(), {c: False for c in PILOT_CAMERAS})
            self.assertFalse(self.probe.any_plug())
        finally:
            env.data.qpos[:] = qpos
            mujoco.mj_forward(env.model, env.data)

    def test_probe_does_not_write_state(self):
        qpos, qvel = self.env.data.qpos.copy(), self.env.data.qvel.copy()
        self.probe.plug()
        self.probe.socket()
        np.testing.assert_array_equal(self.env.data.qpos, qpos)
        np.testing.assert_array_equal(self.env.data.qvel, qvel)

    def test_pickup_coverage(self):
        out = pickup_coverage({"survey": [True, False], "descend": [True, True]})
        self.assertEqual(out["survey"]["fraction"], 0.5)
        self.assertEqual(out["_all"]["covered"], 3)
        self.assertFalse(out["_complete"])
        self.assertTrue(pickup_coverage({"a": [True]})["_complete"])


if __name__ == "__main__":
    unittest.main()
