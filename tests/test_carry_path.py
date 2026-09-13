import unittest
import numpy as np
from controllers.carry_path import plan_carry
from envs.openarm_insert import OpenArmInsertEnv


class CarryPathTests(unittest.TestCase):
    def test_planning_never_moves_live_robot_or_payload(self):
        env=OpenArmInsertEnv(images=False)
        try:
            data=env.data
            saved_qpos=data.qpos.copy();saved_qvel=data.qvel.copy()
            rotation=data.site_xmat[env.grasp_site].reshape(3,3)
            relative_position=rotation.T@(data.xpos[env.plug]-data.site_xpos[env.grasp_site])
            relative_rotation=rotation.T@data.xmat[env.plug].reshape(3,3)
            start=data.qpos[env.qa['right']].copy()
            goal=start.copy();goal[-1]+=.001
            path,report=plan_carry(env,start,goal,relative_position,relative_rotation)
            np.testing.assert_array_equal(path[-1],goal)
            np.testing.assert_array_equal(data.qpos,saved_qpos)
            np.testing.assert_array_equal(data.qvel,saved_qvel)
            self.assertGreater(report['collision_queries'],0)
        finally:env.close()


if __name__=='__main__':unittest.main()
