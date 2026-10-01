import unittest
import numpy as np
from controllers.place_insert import (ArmKinematics, InsertFeasibility, PlaceInsert, nominal_grasp,
                                      plan_insertion, REFERENCE_POSTURE)
from data_pipeline.scene_bank import sample_socket, workspace_for_env
from envs.openarm_insert import OpenArmInsertEnv

TABLE = 0.32
# Workspace scene: the plug stays within reach of its 0.35 m cable (anchored at the appliance
# near (0.40, -0.42)), and the socket keeps its fixture-pedestal height above the table.
PLUG_XY = (0.34, -0.25)


def scene(socket_xy, socket_yaw, entry_z):
    return {"plug_pos_m": [*PLUG_XY, TABLE + 0.008], "plug_yaw_deg": 0.,
            "socket_pos_m": [socket_xy[0], socket_xy[1], entry_z],
            "socket_yaw_deg": socket_yaw, "socket_tilt_deg": 0., "table_height_m": TABLE}


class PlaceInsertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = OpenArmInsertEnv(images=False)
        cls.ws = workspace_for_env(cls.env)
        cls.entry_z = TABLE + cls.ws.socket_entry_above_table_m

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_sampled_socket_stays_mounted_on_its_pedestal(self):
        env = self.env
        home_offset = env.fixture_home_pos - env.socket_home_pos
        rng = np.random.default_rng(3)
        mounted = 0
        for _ in range(20):
            if mounted == 5:
                break
            pos, yaw, tilt = sample_socket(rng, self.ws)
            self.assertAlmostEqual(pos[2], self.entry_z, delta=1e-12)   # workspace mounting height
            options = scene(pos[:2], yaw, pos[2])
            options["socket_tilt_deg"] = tilt
            try:
                env.reset(seed=0, options=options)
            except RuntimeError:
                # The raised socket/pedestal can overlap the right hand's home pose; reset refuses
                # the scene (the sampler rejects it as insert_infeasible:reset_failed).
                continue
            mounted += 1
            m, d = env.model, env.data
            np.testing.assert_allclose(d.site_xpos[env.socket_site], pos, atol=1e-9)
            # The pedestal moved by the same rigid transform, so the socket is still mounted on it.
            rot = d.site_xmat[env.socket_site].reshape(3, 3)
            np.testing.assert_allclose(m.geom_pos[env.fixture_geom] - m.body_pos[env.socket],
                                       rot @ home_offset, atol=1e-9)
            axis = d.site_xmat[env.socket_site].reshape(3, 3)
            self.assertAlmostEqual(axis[2, 0], 0., delta=1e-9)      # horizontal mating axis
            self.assertGreater(axis[2, 2], 0.999)                   # upright
            self.assertGreater(axis[0, 0], 0.)                      # face looks back at the robot
        self.assertEqual(mounted, 5)
        # A grasped-mode reset restores the workspace pose of socket and pedestal.
        env.reset()
        np.testing.assert_array_equal(env.model.body_pos[env.socket], env.socket_home_pos)
        np.testing.assert_array_equal(env.model.geom_pos[env.fixture_geom], env.fixture_home_pos)

    def test_planning_never_moves_live_state(self):
        env = self.env
        env.reset(seed=0, options=scene((0.42, -0.30), 0., self.entry_z))
        qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
        kin = ArmKinematics(env.model)
        entry = env.data.site_xpos[env.socket_site].copy()
        basis = env.data.site_xmat[env.socket_site].reshape(3, 3).copy()
        plans = [plan_insertion(kin, env.data.qpos.copy(), REFERENCE_POSTURE, entry, basis,
                                *nominal_grasp(flip, -30.), table_z=TABLE) for flip in (0, 1)]
        plan = next((p for p in plans if p.failure is None), plans[0])
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
        reason, grasps = screen(scene((0.42, -0.30), 0., self.entry_z))
        self.assertIsNone(reason)
        self.assertTrue(grasps)
        reason, grasps = screen(scene((0.60, 0.25), 10., self.entry_z))   # far out of reach
        self.assertIsNotNone(reason)
        self.assertEqual(grasps, [])


if __name__ == "__main__":
    unittest.main()
