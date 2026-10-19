"""Depth cameras (overhead D435, wrist D405, chest D435) and joint torque logging (robot scene;
needs third_party assets)."""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from data_pipeline.episodes import save_episode, validate_episode
from data_pipeline.scene_bank import as_reset_options, iter_collection_bank, workspace_for_env
from envs import d435
from envs.openarm_insert import OpenArmInsertEnv
from controllers.place_insert import plug_geometry
from controllers.pick_insert_expert import GraspPlanner, plan_table_pick, flat_plug_pose

CABLE30 = "configs/workspace_v2_cable30.json"
SENSORS = {"scene": ("scene_rgb", "D435"), "wrist": ("wrist_rgb", "D405"), "chest": ("chest_rgb", "D435")}
# Share of valid (> 0) depth pixels at the reset of bank row 6100, measured before choosing the
# thresholds: scene 0.908, wrist 0.429 (half of the wrist view is empty background), chest 0.510.
MIN_VALID_FRACTION = {"scene": 0.6, "wrist": 0.2, "chest": 0.3}


def ray_z(env, cam, k, us, vs):
    """Ground-truth z-depth (m) of pixel centres (us, vs): cast a ray from the camera (it looks along
    its -z, image y down) with geom group 3 hidden like the render; nan where the ray hits nothing."""
    m, d = env.model, env.data
    cid = m.camera(cam).id
    pos, mat = d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    out = np.full(len(us), np.nan)
    for i, (u, v) in enumerate(zip(us, vs)):
        local = np.array([(u - k["cx"]) / k["fx"], -(v - k["cy"]) / k["fy"], -1.0])
        dist = mujoco.mj_ray(m, d, pos, mat @ (local / np.linalg.norm(local)), opt.geomgroup, 1, -1,
                             np.zeros(1, dtype=np.int32))
        if dist >= 0:
            out[i] = dist / np.linalg.norm(local)
    return out


def patch_errors(env, cam, depth, k, step=24):
    """Median (measured - true) z in metres of every 5x5 patch on a coarse grid that sits on one
    smooth surface (true z varies < 1.5 cm across it, so no edge) and keeps >= 12 valid pixels."""
    errors = []
    for v in range(12, k["height"] - 12, step):
        for u in range(12, k["width"] - 12, step):
            uu, vv = np.meshgrid(np.arange(u - 2, u + 3), np.arange(v - 2, v + 3))
            truth = ray_z(env, cam, k, uu.ravel(), vv.ravel())
            measured = depth[vv.ravel(), uu.ravel()] / 1e3
            ok = np.isfinite(truth) & (measured > 0)
            if np.isfinite(truth).all() and np.ptp(truth) < 0.015 and ok.sum() >= 12:
                errors.append(np.median(measured[ok] - truth[ok]))
    return np.array(errors)


class DepthCameraTests(unittest.TestCase):
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
        self.assertEqual(set(again["depth"]), {"scene", "wrist", "chest"})
        for short in SENSORS:
            np.testing.assert_array_equal(again["depth"][short], self.obs["depth"][short], err_msg=short)
        # a later observation of the same episode draws a different noise realisation
        self.assertFalse(np.array_equal(self.env.observe()["depth"]["scene"], self.obs["depth"]["scene"]))

    def reset_obs(self):
        """Fresh reset of the bank row (other tests step the shared env), so env.data matches obs."""
        return self.env.reset(seed=self.row["seed"], options=as_reset_options(self.row["options"]))[0]

    def test_each_depth_stream_is_aligned_uint16_and_in_range(self):
        obs, cams = self.reset_obs(), self.env.manifest()["depth_cameras"]
        for short, (rgb_name, sensor) in SENSORS.items():
            with self.subTest(camera=short):
                rgb, depth, p = obs["images"][rgb_name], obs["depth"][short], cams[short]["depth_model"]
                self.assertEqual(depth.shape, rgb.shape[:2])
                self.assertEqual(depth.dtype, np.uint16)
                valid = depth[depth > 0].astype(float)
                self.assertGreater(valid.size, MIN_VALID_FRACTION[short] * depth.size)
                # noise can push a reading a little beyond the cut-offs the true depth respected
                lo = p["min_range_m"] * 1e3 - 3 * 1e3 * d435.noise_sigma_m(p["min_range_m"], p) - 1
                hi = p["max_range_m"] * 1e3 + 3 * 1e3 * d435.noise_sigma_m(p["max_range_m"], p) + 1
                self.assertTrue(((valid >= lo) & (valid <= hi)).all(), (valid.min(), valid.max()))

    def test_scene_and_wrist_depth_match_ray_cast_geometry(self):
        """Depth on smooth surfaces (table, robot links, fingers) equals the true z-distance from mj_ray.

        Measured at row 6100: scene 102 patches, worst median error 2.8 mm; wrist 31 patches, 2.2 mm
        (a half-pixel offset on a sloped surface, not noise: the wrist noise is ~0.2 mm at 0.1 m)."""
        obs, cams = self.reset_obs(), self.env.manifest()["depth_cameras"]
        for short in ("scene", "wrist"):
            with self.subTest(camera=short):
                errors = patch_errors(self.env, cams[short]["camera"], obs["depth"][short],
                                      cams[short]["intrinsics"])
                self.assertGreaterEqual(errors.size, 3, errors.size)
                self.assertLess(np.abs(errors).max(), 0.005, np.abs(errors).max())

    def test_depth_cameras_manifest(self):
        man = self.env.manifest()
        cams = man["depth_cameras"]
        self.assertEqual(set(cams), set(SENSORS))
        for short, (rgb_name, sensor) in SENSORS.items():
            self.assertEqual(cams[short]["camera"], rgb_name)
            self.assertEqual(cams[short]["sensor"], sensor)
            self.assertEqual(cams[short]["depth_aligned_to"], rgb_name)
            k = cams[short]["intrinsics"]
            cam = self.env.model.camera(rgb_name).id
            self.assertAlmostEqual(k["fy"], 0.5 * k["height"] / np.tan(np.radians(self.env.model.cam_fovy[cam]) / 2.))
            self.assertEqual((k["height"], k["width"]), self.obs["images"][rgb_name].shape[:2])
        self.assertEqual(cams["wrist"]["depth_model"]["min_range_m"], 0.07)       # D405: 7 cm - 1 m
        self.assertEqual(cams["wrist"]["depth_model"]["baseline_m"], 0.018)
        numbers = lambda p: {k: v for k, v in p.items() if k != "description"}
        self.assertEqual(numbers(cams["scene"]["depth_model"]), numbers(cams["chest"]["depth_model"]))   # same D435
        self.assertEqual(man["chest_rgbd"]["intrinsics"], cams["chest"]["intrinsics"])

    def test_saved_episode_has_all_depth_arrays_and_pairing_check(self):
        env = self.env
        obs, info = env.reset(seed=self.row["seed"], options=as_reset_options(self.row["options"]))
        hold = np.r_[env.target[:7], env.target[7]]
        steps = 3
        observations, infos, states, velocities = [obs], [info], [env.data.qpos.copy()], [env.data.qvel.copy()]
        requested, applied, rewards, terminated, truncated, durations = [], [], [], [], [], []
        for i in range(steps):
            obs, reward, term, trunc, info = env.step(hold)
            observations.append(obs)
            infos.append(info)
            states.append(env.data.qpos.copy())
            velocities.append(env.data.qvel.copy())
            requested.append(hold.copy())
            applied.append(info["applied_action"].copy())
            rewards.append(reward)
            durations.append(info["executed_dt_s"])
            terminated.append(i == steps - 1)       # validate_episode wants a finished, successful episode
            truncated.append(False)
        infos[-1] = dict(infos[-1], outcome="success")
        with tempfile.TemporaryDirectory() as tmp:
            ep = Path(tmp) / "ep"
            meta = save_episode(ep, observations, requested, applied, states, velocities, rewards, terminated,
                                truncated, durations, ["hold"] * steps, infos, env.manifest())
            self.assertEqual(set(meta["camera_frames"]["depth_folders"]),
                             {"head_depth", "wrist_depth", "chest_depth"})
            self.assertEqual(meta["camera_frames"]["depth_folders"]["head_depth"], "depth_scene_mm")
            self.assertEqual(set(meta["depth_cameras"]), set(SENSORS))
            with np.load(ep / "episode.npz", allow_pickle=False) as a:
                shapes = {k: a[k].shape for k in ("depth_scene_mm", "depth_wrist_mm", "depth_chest_mm")}
                self.assertEqual(shapes, {"depth_scene_mm": (steps + 1, 240, 320),
                                          "depth_wrist_mm": (steps + 1, 240, 320),
                                          "depth_chest_mm": (steps + 1, 240, 424)})
                self.assertTrue(all(a[k].dtype == np.uint16 for k in shapes))
                np.testing.assert_array_equal(a["depth_wrist_mm"][-1], observations[-1]["depth"]["wrist"])
            for folder in ("head_depth", "wrist_depth", "chest_depth"):
                self.assertEqual(len(list((ep / "cameras" / folder).glob("*.png"))), steps + 1, folder)
            validate_episode(ep)
            # depth_<x>_mm must pair with an image_<x>_rgb of the same size
            for name, array in (("depth_orphan_mm", np.zeros((steps + 1, 240, 320), dtype=np.uint16)),
                                ("depth_scene_mm", np.zeros((steps + 1, 10, 10), dtype=np.uint16))):
                bad = Path(tmp) / f"bad_{name}"
                shutil.copytree(ep, bad)
                with np.load(ep / "episode.npz", allow_pickle=False) as a:
                    arrays = {k: a[k] for k in a.files}
                arrays[name] = array
                np.savez_compressed(bad / "episode.npz", **arrays)
                metadata = json.loads((bad / "metadata.json").read_text())
                metadata["data_sha256"] = hashlib.sha256((bad / "episode.npz").read_bytes()).hexdigest()
                (bad / "metadata.json").write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "Depth schema mismatch"):
                    validate_episode(bad)

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
        self.assertIsNone(env.manifest()["depth_cameras"])
        self.assertEqual(set(obs["images"]), {"scene_rgb", "wrist_rgb"})
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
