"""Chest RGB-D camera and joint torque logging (robot scene; needs third_party assets)."""
import unittest

import numpy as np

from data_pipeline.scene_bank import as_reset_options, iter_collection_bank, workspace_for_env
from envs.openarm_insert import OpenArmInsertEnv
from controllers.place_insert import plug_geometry
from controllers.pick_insert_expert import GraspPlanner, plan_table_pick, flat_plug_pose

CABLE30 = "configs/workspace_v2_cable30.json"


class ChestCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = OpenArmInsertEnv(images=True, workspace=CABLE30)
        ws = workspace_for_env(cls.env)
        cls.row = next(iter(iter_collection_bank(n=1, validation_fraction=0.1, ws=ws, start=6100)))
        cls.obs, cls.info = cls.env.reset(seed=cls.row["seed"], options=as_reset_options(cls.row["options"]))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_rgb_and_aligned_depth(self):
        c = self.env.workspace["cameras"]["chest_rgbd"]
        rgb, depth = self.obs["images"]["chest_rgb"], self.obs["depth"]["chest"]
        self.assertEqual(rgb.shape, (c["height"], c["width"], 3))
        self.assertEqual(depth.shape, (c["height"], c["width"]))
        self.assertEqual(depth.dtype, np.uint16)
        valid = depth[depth > 0]
        self.assertGreater(valid.size, 0.3 * depth.size)
        self.assertTrue(((valid >= 280) & (valid <= 3000)).all())

    def test_depth_matches_geometry(self):
        """The pixel the plug projects to reads the camera's z-distance to the plug (+/- a few mm)."""
        m, d = self.env.model, self.env.data
        cam = m.camera("chest_rgb").id
        pos, mat = d.cam_xpos[cam], d.cam_xmat[cam].reshape(3, 3)
        local = mat.T @ (d.xpos[self.env.plug] - pos)                 # camera frame, looks along -z
        k = self.env.manifest()["chest_rgbd"]["intrinsics"]
        u = int(round(k["cx"] + k["fx"] * local[0] / -local[2]))
        v = int(round(k["cy"] - k["fy"] * local[1] / -local[2]))
        patch = self.obs["depth"]["chest"][v - 2:v + 3, u - 2:u + 3].astype(float)
        patch = patch[patch > 0]
        self.assertGreater(patch.size, 0)
        # plug body origin is inside the housing; its visible top is a few mm nearer the camera
        self.assertAlmostEqual(np.median(patch) / 1e3, -local[2], delta=0.03)

    def test_depth_reproducible_per_seed_and_step(self):
        again = self.env.reset(seed=self.row["seed"], options=as_reset_options(self.row["options"]))[0]
        np.testing.assert_array_equal(again["depth"]["chest"], self.obs["depth"]["chest"])

    def test_manifest_and_torque(self):
        man = self.env.manifest()
        self.assertEqual(man["chest_rgbd"]["camera"], "chest_rgb")
        self.assertIn("envs/d435.py", man["source_hashes"])
        self.assertEqual(len(man["joint_torque_order"]), 14)
        _, _, _, _, info = self.env.step(np.r_[self.env.target[:7], self.env.target[7]])
        applied = np.asarray(info["joint_torque_applied_nm"])
        peak = np.asarray(info["joint_torque_applied_peak_nm"])
        gravity = np.asarray(info["joint_torque_gravity_nm"])
        self.assertEqual(applied.shape, (14,))
        self.assertTrue(np.isfinite(applied).all() and (peak >= np.abs(applied) - 1e-12).all())
        limits = self.env.model.actuator_forcerange[self.env.torque_act, 1]
        self.assertTrue((peak <= limits + 1e-9).all())
        # holding still: the motors carry gravity
        np.testing.assert_allclose(applied, gravity, atol=0.05)


class OtherWorkspaceTests(unittest.TestCase):
    def test_v1_has_no_chest_camera(self):
        env = OpenArmInsertEnv(images=True)
        obs, _ = env.reset(seed=0)
        names = [env.model.camera(i).name for i in range(env.model.ncam)]
        self.assertNotIn("chest_rgb", names)
        self.assertNotIn("depth", obs)
        self.assertNotIn("chest_rgb", obs["images"])
        self.assertIsNone(env.manifest()["chest_rgbd"])
        env.close()


class MissPlanTests(unittest.TestCase):
    def test_grasp_offset_moves_only_the_grasp_point(self):
        env = OpenArmInsertEnv(images=False, workspace=CABLE30)
        ws = workspace_for_env(env)
        row = next(iter(iter_collection_bank(n=1, validation_fraction=0.1, ws=ws, start=6100)))
        env.reset(seed=row["seed"], options=as_reset_options(row["options"]))
        geom = plug_geometry(env.model, env.workspace)
        planner = GraspPlanner(env.model, row["options"]["table_height_m"], geom=geom)
        plug_pos, plug_mat = flat_plug_pose(env.data.xpos[env.plug].copy(),
                                            env.data.xmat[env.plug].reshape(3, 3).copy(),
                                            row["options"]["table_height_m"], geom.rest_z_m)
        q0 = env.target[:7].copy()
        qpos = env.data.qpos.copy()
        base, _, _ = plan_table_pick(planner, qpos, plug_pos, plug_mat, q0)
        offset = plug_mat @ np.array([0.08, 0., 0.])
        offset[2] = 0.
        miss, _, _ = plan_table_pick(planner, qpos, plug_pos, plug_mat, q0, grasp_offset=offset)
        self.assertIsNotNone(base)
        self.assertIsNotNone(miss)
        np.testing.assert_allclose(miss["grasp_pos"] - base["grasp_pos"], offset, atol=1e-9)
        np.testing.assert_array_equal(env.data.qpos, qpos)          # planner never writes live state
        env.close()


if __name__ == "__main__":
    unittest.main()
