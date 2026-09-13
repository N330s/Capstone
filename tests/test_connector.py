"""Behavioral checks for geometry-aware success and held-plug dynamics."""
import unittest

import mujoco
import numpy as np

from connector.simulation import ConnectorSimulation, Pose, run_trial


class GeometryTests(unittest.TestCase):
    def test_mass_and_free_joint(self):
        sim = ConnectorSimulation()
        self.assertEqual(sim.model.nv, 6)
        self.assertAlmostEqual(sim.model.body_mass[sim.plug], 0.03)
        self.assertTrue(np.all(sim.model.body_inertia[sim.plug] > 0))
        self.assertIsNone(sim.outcome)

    def test_seating_and_half_turn_symmetry(self):
        for roll in (0, 180):
            with self.subTest(roll=roll):
                sim = ConnectorSimulation()
                sim.reset(Pose(roll_deg=roll), mating_x=-0.0001)
                self.assertTrue(sim.info["valid_pose"])

    def test_false_success_poses(self):
        cases = [
            ("side bypass", Pose(offset_y_mm=50), -0.0001),
            ("one blade in opposite slot", Pose(offset_y_mm=13), -0.0001),
            ("roll", Pose(roll_deg=90), -0.0001),
            ("excess depth", Pose(), 0.025),
            ("shallow", Pose(), -0.005),
        ]
        for name, pose, x in cases:
            with self.subTest(name=name):
                sim = ConnectorSimulation()
                sim.reset(pose, mating_x=x)
                self.assertFalse(sim.info["valid_pose"])

    def test_initial_interpenetration_rejected(self):
        sim = ConnectorSimulation()
        sim.reset(Pose(offset_y_mm=1), mating_x=-0.010)
        self.assertEqual(sim.outcome, "initial_state_rejection")

    def test_world_frame_invariance(self):
        sim = ConnectorSimulation()
        sim.reset(Pose(offset_y_mm=0.1), mating_x=-0.0001)
        before = sim.info
        sim.model.body_pos[sim.socket] = [0.02, 0.03, 0.15]
        sim.model.body_quat[sim.socket] = [np.cos(0.3), 0, 0, np.sin(0.3)]
        sim.reset(Pose(offset_y_mm=0.1), mating_x=-0.0001)
        self.assertTrue(sim.info["valid_pose"])
        for key in ("left_depth_m", "right_depth_m", "offset_y_m", "orientation_error_deg"):
            self.assertAlmostEqual(sim.info[key], before[key], places=8)

    def test_other_contact_is_not_socket_force(self):
        sim = ConnectorSimulation()
        # At reset only: put plug housing on the table, far from the fixture.
        sim.data.qpos[2] = 0.0079
        mujoco.mj_forward(sim.model, sim.data)
        info = sim.diagnostics()
        self.assertGreater(info["other_contact_force_n"], 0)
        self.assertEqual(info["contact_force_n"], 0)
        self.assertFalse(info["valid_pose"])

    def test_hold_is_not_pose_only(self):
        sim = ConnectorSimulation()
        sim.reset(mating_x=-0.0001)
        self.assertTrue(sim.info["valid_pose"])
        self.assertNotEqual(sim.outcome, "success")
        sim.step()
        self.assertNotEqual(sim.outcome, "success")

    def test_paused_holder_does_not_consume_trial_duration(self):
        sim = ConnectorSimulation(config={"duration_s": 0.01})
        for _ in range(40):
            sim.step(advancing=False)
        self.assertIsNone(sim.outcome)
        self.assertEqual(sim.active_time, 0)


class DynamicsTests(unittest.TestCase):
    def test_aligned_repeatability(self):
        results = [run_trial() for _ in range(3)]
        self.assertTrue(all(r["success"] for r in results))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_vertical_errors_are_preserved_and_blocked(self):
        for offset in (-1, 1):
            result = run_trial(Pose(offset_z_mm=offset))
            self.assertEqual(result["outcome"], "jam")
            self.assertAlmostEqual(result["first_contact"]["offset_z_m"], offset / 1000, places=5)

    def test_half_timestep(self):
        a = run_trial()
        b = run_trial(timestep=0.00025)
        self.assertEqual(a["outcome"], b["outcome"])
        self.assertLess(abs(a["final"]["insertion_depth_m"] - b["final"]["insertion_depth_m"]), 1e-5)
        self.assertLess(abs(a["max_contact_force_n"] - b["max_contact_force_n"]), 0.1)


if __name__ == "__main__":
    unittest.main()

