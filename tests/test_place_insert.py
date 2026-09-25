import unittest
import numpy as np
from controllers.place_insert import (ArmKinematics, InsertFeasibility, PlaceInsert, nominal_grasp,
                                      plan_insertion, REFERENCE_POSTURE)
from data_pipeline.scene_bank import WORKSPACE, SOCKET_BASE_BELOW_ENTRY_M, sample_socket
from envs.openarm_insert import OpenArmInsertEnv

TABLE = 0.32


def scene(socket_xy, socket_yaw):
    return {"plug_pos_m": [0.50, 0.25, TABLE + 0.008], "plug_yaw_deg": 0.,
            "socket_pos_m": [socket_xy[0], socket_xy[1], TABLE + SOCKET_BASE_BELOW_ENTRY_M],
            "socket_yaw_deg": socket_yaw, "socket_tilt_deg": 0., "table_height_m": TABLE}


class PlaceInsertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = OpenArmInsertEnv(images=False)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_sampled_socket_is_upright_on_the_table(self):
        rng = np.random.default_rng(3)
        for _ in range(5):
            pos, yaw, tilt = sample_socket(rng, WORKSPACE)
            options = scene(pos[:2], yaw)
            options["socket_tilt_deg"] = tilt
            self.env.reset(seed=0, options=options)
            m, d = self.env.model, self.env.data
            socket = self.env.socket
            lowest = min(d.geom_xpos[g][2] - abs(d.geom_xmat[g].reshape(3, 3)[2]) @ m.geom_size[g]
                         for g in range(m.ngeom) if m.geom_bodyid[g] == socket)
            self.assertTrue(TABLE <= lowest < TABLE + 2e-4, lowest)   # resting on, not in, the table
            axis = d.site_xmat[self.env.socket_site].reshape(3, 3)
            self.assertAlmostEqual(axis[2, 0], 0., delta=1e-9)      # horizontal mating axis
            self.assertGreater(axis[2, 2], 0.999)                   # upright
            self.assertGreater(axis[0, 0], 0.)                      # face looks back at the robot

    def test_planning_never_moves_live_state(self):
        env = self.env
        env.reset(seed=0, options=scene((0.32, -0.22), 10.))
        qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
        kin = ArmKinematics(env.model)
        entry = env.data.site_xpos[env.socket_site].copy()
        basis = env.data.site_xmat[env.socket_site].reshape(3, 3).copy()
        plan = plan_insertion(kin, env.data.qpos.copy(), REFERENCE_POSTURE, entry, basis,
                              *nominal_grasp(0, -30.), table_z=TABLE)
        self.assertIsNone(plan.failure, plan.detail)
        self.assertGreaterEqual(kin.margin(plan.q_seated), 0.05)
        np.testing.assert_array_equal(env.data.qpos, qpos)
        np.testing.assert_array_equal(env.data.qvel, qvel)
        # An empty hand is reported before any motion, not timed out.
        place = PlaceInsert(env, kin=kin)
        self.assertEqual(place.start(), "handoff_not_held")
        np.testing.assert_array_equal(env.data.qpos, qpos)

    def test_insert_feasibility_screen(self):
        screen = InsertFeasibility()
        reason, grasps = screen(scene((0.32, -0.22), 10.))
        self.assertIsNone(reason)
        self.assertTrue(grasps)
        reason, grasps = screen(scene((0.60, 0.25), 10.))   # far out of reach
        self.assertIsNotNone(reason)
        self.assertEqual(grasps, [])


if __name__ == "__main__":
    unittest.main()
