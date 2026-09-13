import unittest
import numpy as np
from envs.openarm_insert import OpenArmInsertEnv
from controllers.table_pickup import PickupProbe


class TableRestTests(unittest.TestCase):
    def test_downward_rest_and_free_table_support(self):
        env=OpenArmInsertEnv(images=False)
        try:
            probe=PickupProbe(env)
            probe.reset()
            for side in ('left','right'):
                forward=env.data.xmat[env.model.body(f'openarm_{side}_hand').id].reshape(3,3)[:,2]
                self.assertLess(forward[2],-.999)
            self.assertLess(probe.peak_robot_force,.01)
            self.assertAlmostEqual(env.data.xpos[env.plug,2],.328,places=5)
            self.assertEqual(probe.finger_contacts(),[])
            np.testing.assert_array_equal(env.data.xfrc_applied,0)
            before=env.data.qpos.copy()
            probe.reset()
            np.testing.assert_array_equal(env.data.qpos,before)
        finally: env.close()


if __name__=='__main__':unittest.main()
