"""Workspace spec tests; no robot assets needed."""
import json
import sys
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from connector import catalog
from connector.geometry import connector_tree
from envs import workspace as wsp

V2_PATH = wsp.ROOT / "configs/workspace_v2.json"


class WorkspaceSpecTests(unittest.TestCase):
    def setUp(self):
        self.ws = wsp.load_workspace()

    def test_table_geometry_matches_frames(self):
        ws = self.ws
        boxes = wsp.table_boxes(ws)
        top_c, top_h = boxes["work_table"]
        self.assertAlmostEqual(top_c[2] + top_h[2], ws["frames"]["table_top_z_m"])
        self.assertAlmostEqual(2 * top_h[0], ws["table"]["width_x_m"])
        self.assertAlmostEqual(2 * top_h[1], ws["table"]["length_y_m"])
        for name, (c, h) in boxes.items():
            if name.startswith("table_leg"):
                self.assertAlmostEqual(c[2] - h[2], ws["frames"]["floor_z_m"])
                self.assertAlmostEqual(c[2] + h[2], top_c[2] - top_h[2])
        ped_c, ped_h = boxes["robot_pedestal"]
        self.assertAlmostEqual(ped_c[2] - ped_h[2], ws["frames"]["floor_z_m"])
        self.assertAlmostEqual(ped_c[2] + ped_h[2], ws["pedestal"]["top_z_m"])
        # Spawn and socket stay on the table top.
        for point in (ws["plug_spawn"]["mating_position_m"], ws["socket"]["position_m"]):
            self.assertLess(abs(point[0] - top_c[0]), top_h[0])
            self.assertLess(abs(point[1] - top_c[1]), top_h[1])

    def test_validation_rejects_inconsistent_floor(self):
        ws = json.loads(json.dumps({k: v for k, v in self.ws.items() if not k.startswith("_")}))
        ws["frames"]["floor_z_m"] += .01
        with self.assertRaises(ValueError):
            wsp.validate(ws)

    def test_cable_rest_arc(self):
        ws = self.ws
        _, positions, rotations = wsp.cable_rest(ws)
        n, length = ws["cable"]["segments"], ws["cable"]["length_m"]
        self.assertEqual(positions.shape, (n + 1, 3))
        self.assertAlmostEqual(float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))), length)
        np.testing.assert_allclose(positions[-1], wsp.anchor_world(ws), atol=1e-9)
        for k in range(n):
            direction = positions[k + 1] - positions[k]
            np.testing.assert_allclose(rotations[k][:, 0], direction / np.linalg.norm(direction), atol=1e-9)
            np.testing.assert_allclose(rotations[k].T @ rotations[k], np.eye(3), atol=1e-9)
        # Rest layout stays on the table top (bulges toward the robot, not off the near edge).
        top_c, top_h = wsp.table_boxes(ws)["work_table"]
        self.assertGreater(positions[:, 0].min(), top_c[0] - top_h[0] + ws["cable"]["radius_m"])
        with self.assertRaises(ValueError):
            wsp.arc_frames([0, 0, 0], [1, 0, 0], .5, 4, [0, 1, 0])

    def test_quaternion_helpers_round_trip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            q *= np.sign(q[0]) or 1
            np.testing.assert_allclose(wsp.mat_to_quat(wsp.quat_to_mat(q)), q, atol=1e-9)

    def test_leaf_numbers(self):
        slot = wsp.slot_geometry(connector_tree(True))
        self.assertAlmostEqual(slot["wall_face_m"], .00765)
        self.assertAlmostEqual(slot["blade_face_m"], .00725)
        openings = wsp.leaf_openings(catalog.legacy(), self.ws)
        self.assertEqual([o["name"] for o in openings], ["left", "right"])
        # The catalog legacy entry reproduces the numbers read from the hand-written XML.
        self.assertAlmostEqual(abs(openings[0]["base_yz"][0]), slot["wall_face_m"])
        self.assertAlmostEqual(openings[0]["clearance_m"], slot["wall_face_m"] - slot["blade_face_m"])
        nums = wsp.leaf_numbers(self.ws, openings)["left"]
        lv = self.ws["socket_leaves"]
        self.assertAlmostEqual(nums["normal_n"], lv["insertion_force_target_n"] / (2 * lv["friction"][0]))
        self.assertAlmostEqual(nums["preload_n"] + lv["stiffness_n_m"] * nums["interference_m"], nums["normal_n"])
        self.assertAlmostEqual(nums["springref_m"] * lv["stiffness_n_m"], nums["preload_n"])

    def test_leaf_openings_universal_socket(self):
        ws = wsp.load_workspace(V2_PATH)
        spec = catalog.from_workspace(ws)
        self.assertEqual((spec.plug.name, spec.socket.name), ("type_o", "universal_th"))
        openings = wsp.leaf_openings(spec, ws)
        self.assertEqual([o["name"] for o in openings], ["left", "right", "earth"])
        by_name = {o["name"]: o for o in openings}
        # The earth leaf presses down and the line/neutral leaves 30 deg above inward-horizontal,
        # so the three equal leaf forces and their torques about X cancel on the held plug.
        self.assertEqual(by_name["earth"]["axis_yz"], (0.0, -1.0))
        np.testing.assert_allclose(by_name["left"]["axis_yz"], (np.cos(np.pi / 6), 0.5))
        np.testing.assert_allclose(by_name["right"]["axis_yz"], (-np.cos(np.pi / 6), 0.5))
        np.testing.assert_allclose(np.sum([o["axis_yz"] for o in openings], axis=0), 0, atol=1e-12)
        torque = sum(o["base_yz"][0] * o["axis_yz"][1] - o["base_yz"][1] * o["axis_yz"][0] for o in openings)
        self.assertAlmostEqual(torque, 0.0)
        np.testing.assert_allclose(by_name["left"]["base_yz"],
                                   (-0.0095 - 0.00255 * np.cos(np.pi / 6), -0.00255 * 0.5))
        self.assertAlmostEqual(by_name["earth"]["base_yz"][1], 0.01189 + 0.00255)
        for o in openings:
            self.assertAlmostEqual(o["clearance_m"], 0.00015)
        # A plug without an earth pin cannot use the tilted set; its leaves fall back to +/-Y.
        c = {o["name"]: o for o in wsp.leaf_openings(catalog.get("type_c", "universal_th"))}
        self.assertEqual((c["left"]["axis_yz"], c["right"]["axis_yz"]), ((1.0, 0.0), (-1.0, 0.0)))
        nums = wsp.leaf_numbers(ws, openings)
        self.assertEqual(set(nums), {"left", "right", "earth"})
        self.assertAlmostEqual(nums["earth"]["normal_n"],
                               ws["socket_leaves"]["insertion_force_target_n"] / (3 * ws["socket_leaves"]["friction"][0]))
        # Blades in a keyhole are pressed outward from the inner web (their outer side is open).
        blade = {o["name"]: o for o in wsp.leaf_openings(catalog.get("type_a", "universal_th"))}
        self.assertEqual(blade["left"]["axis_yz"], (-1.0, 0.0))
        self.assertAlmostEqual(blade["left"]["base_yz"][0], -(0.00635 - 0.00095))
        self.assertAlmostEqual(blade["left"]["clearance_m"], 0.0002)
        # Openings the plug does not use carry no leaf.
        self.assertEqual([o["name"] for o in wsp.leaf_openings(catalog.get("type_c", "universal_th"))],
                         ["left", "right"])

    def test_cable_path_without_waypoints_is_the_single_arc(self):
        cable, spawn = self.ws["cable"], self.ws["plug_spawn"]
        start = np.asarray(spawn["mating_position_m"]) + np.asarray(cable["attach_local_m"])
        expected, _ = wsp.arc_frames(start, wsp.anchor_world(self.ws), cable["length_m"],
                                     cable["segments"], cable["rest_bulge_world"])
        positions, rotations = wsp.cable_path(self.ws, start)
        np.testing.assert_allclose(positions, expected)
        self.assertEqual(len(rotations), cable["segments"])

    def test_v2_cable_runs_over_the_edge_to_the_floor(self):
        ws = wsp.load_workspace(V2_PATH)
        cable = ws["cable"]
        pieces = wsp.cable_pieces(ws, wsp.cable_rest(ws)[1][0])
        self.assertEqual(len(pieces), 3)
        self.assertEqual(sum(k for _, _, k in pieces), cable["segments"])
        ls = cable["length_m"] / cable["segments"]
        for a, b, k in pieces:
            self.assertGreater(k * ls, np.linalg.norm(b - a))
        _, positions, _ = wsp.cable_rest(ws)
        self.assertEqual(len(positions), cable["segments"] + 1)
        floor = ws["frames"]["floor_z_m"]
        self.assertGreaterEqual(positions[:, 2].min(), floor - cable["radius_m"])
        table = wsp.table_boxes(ws)["work_table"]
        (cx, cy, cz), (hx, hy, hz) = table
        inside = [(x, z) for x, y, z in positions
                  if cx - hx < x < cx + hx and cy - hy < y < cy + hy and cz - hz < z < cz + hz]
        self.assertEqual(inside, [], "rest cable must not pass through the table top")
        # Explicit allocation: the drop keeps its 31 segments, the table piece takes only what its
        # chord needs, and the slack coils on the floor.
        counts = [k for _, _, k in pieces]
        self.assertEqual(counts[1], 31)
        self.assertLessEqual(counts[0] * ls - np.linalg.norm(pieces[0][1] - pieces[0][0]), 2 * ls)
        self.assertGreaterEqual(counts[2], 20)
        # The route follows a randomised spawn (waypoint y is null -> plug y) and stays off the table.
        far, _ = wsp.cable_path(ws, (0.50, -0.45, 0.332))
        self.assertEqual(len(far), cable["segments"] + 1)
        self.assertGreaterEqual(far[:, 2].min(), floor - cable["radius_m"])
        self.assertEqual([1 for x, y, z in far if cx - hx < x < cx + hx and cy - hy < y < cy + hy
                          and cz - hz < z < cz + hz], [])
        self.assertAlmostEqual(wsp.cable_pieces(ws, (0.50, -0.45, 0.332))[0][1][1], -0.45)
        # Held near the socket the far slack is used up: the floor waypoint is dropped, not an error.
        lifted, _ = wsp.cable_path(ws, (0.40, -0.155, 0.478), first_bulge=(0, 0, -1))
        self.assertEqual(len(lifted), cable["segments"] + 1)
        self.assertIn("cable_pieces", wsp.derived(ws))

    def test_finger_travel_for_width(self):
        cfg = json.loads((wsp.ROOT / "configs/openarm_v1.json").read_text())
        travel = wsp.finger_travel_for_width(self.ws, 0.024, cfg["finger_base_gap_m"])
        self.assertAlmostEqual(travel, 0.0138)                      # former initial_finger_travel_m
        self.assertAlmostEqual(travel - cfg["grip_squeeze_m"], 0.008)   # former grip_target_travel_m
        self.assertAlmostEqual(travel - cfg["pickup_squeeze_m"], 0.006)  # former pickup closed travel

    def test_v2_validation_ties_numbers_to_the_spec(self):
        ws = wsp.load_workspace(V2_PATH)
        bad = json.loads(json.dumps({k: v for k, v in ws.items() if not k.startswith("_")}))
        bad["cable"]["attach_local_m"][0] += 0.001
        with self.assertRaises(ValueError):
            wsp.validate(bad)
        bad = json.loads(json.dumps({k: v for k, v in ws.items() if not k.startswith("_")}))
        bad["plug_spawn"]["mating_position_m"][2] = 0.330
        with self.assertRaises(ValueError):
            wsp.validate(bad)

    def test_generated_xml_is_deterministic(self):
        def build():
            root = ET.Element("mujoco")
            world = ET.SubElement(root, "worldbody")
            asset = ET.SubElement(root, "asset")
            wsp.add_floor_table_pedestal(world, self.ws)
            wsp.add_appliance(world, self.ws)
            plug = ET.SubElement(world, "body", name="plug")
            wsp.add_cable(plug, self.ws)
            wsp.add_cable_closure(root, self.ws)
            socket = ET.SubElement(world, "body", name="socket")
            wsp.add_retention_leaves(socket, asset, self.ws, wsp.leaf_openings(catalog.legacy(), self.ws), root)
            return ET.tostring(root, encoding="unicode")
        a, b = build(), build()
        self.assertEqual(a, b)
        self.assertIn('name="cable_seg_00"', a)
        self.assertIn('name="socket_leaf_right_slide"', a)
        self.assertIn("<connect", a)

    def test_universal_socket_leaves_xml(self):
        ws = wsp.load_workspace(V2_PATH)
        root = ET.Element("mujoco")
        asset = ET.SubElement(root, "asset")
        socket = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name="socket")
        names = wsp.add_retention_leaves(socket, asset, ws, wsp.leaf_openings(catalog.from_workspace(ws), ws), root)
        self.assertEqual(names, ["socket_leaf_left", "socket_leaf_right", "socket_leaf_earth"])
        earth = socket.find("./body[@name='socket_leaf_earth']")
        self.assertEqual(earth.find("joint").get("axis"), "0 0 -1")
        self.assertEqual(len(root.findall("./contact/exclude")), 3)
        self.assertEqual(len(asset.findall("mesh")), 3)


class CableTautLiftSamplerTests(unittest.TestCase):
    """Sampler rule for plugs whose cable has no slack to lift (no robot assets needed)."""

    @staticmethod
    def options(plug_x, yaw=0.):
        return {"plug_pos_m": [plug_x, -0.2, 0.324], "plug_yaw_deg": yaw, "socket_pos_m": [0.45, -0.3, 0.472],
                "socket_yaw_deg": 0., "socket_tilt_deg": 0., "table_height_m": 0.32}

    def setUp(self):
        from data_pipeline.scene_bank import Workspace
        self.ws = Workspace(cable_anchor_m=(0.30, -0.22, -0.415), cable_max_reach_m=1.4,
                            cable_attach_local_m=(-0.042, 0., 0.006), cable_edge_waypoint_m=(0.235, None, 0.33))

    def test_attachment_near_the_table_edge_is_rejected(self):
        from data_pipeline.scene_bank import _cable_check
        # attachment x = plug x - 0.042 cos(yaw); the edge waypoint is at x = 0.235
        self.assertEqual(_cable_check(self.options(0.320), self.ws), "cable_taut_on_lift")   # 0.278
        self.assertEqual(_cable_check(self.options(0.310, yaw=60.), self.ws), "cable_taut_on_lift")   # 0.289
        self.assertIsNone(_cable_check(self.options(0.320, yaw=60.), self.ws))               # 0.299 (yaw matters)
        self.assertIsNone(_cable_check(self.options(0.340), self.ws))                        # 0.298

    def test_rule_is_off_without_a_table_edge_waypoint(self):
        from dataclasses import replace
        from data_pipeline.scene_bank import _cable_check
        ws = replace(self.ws, cable_edge_waypoint_m=None)
        self.assertIsNone(_cable_check(self.options(0.320), ws))

    def test_workspace_for_env_reads_the_edge_from_the_spec(self):
        from types import SimpleNamespace
        from data_pipeline.scene_bank import Workspace, workspace_for_env
        spec = wsp.load_workspace(wsp.ROOT / "configs/workspace_v2_cable30.json")
        env = SimpleNamespace(
            reach_anchor_m=lambda: (0., -0.031, 0.698), workspace=spec, socket_site=0,
            data=SimpleNamespace(site_xpos=np.array([[0.4, -0.3, 0.472]])),
            derived={"table_rest_z_offset_m": 0.004}, initial_grasp_position=np.array([0.1, 0.2, 0.3]))
        ws = workspace_for_env(env, Workspace())
        self.assertEqual(ws.cable_edge_waypoint_m, (0.235, None, 0.33))
        v1 = wsp.load_workspace()
        env.workspace = v1
        self.assertIsNone(workspace_for_env(env, Workspace()).cable_edge_waypoint_m)


class V2Stage1SamplerTests(unittest.TestCase):
    """v2 reach-map ranges, settled-pose certification bookkeeping, home-collision check."""

    RANGE_FIELDS = ("plug_x_range", "plug_y_range", "plug_yaw_range_deg", "socket_x_range", "socket_y_range",
                    "socket_yaw_range_deg", "reach_range_m", "hand_keepout_m", "home_collision_check",
                    "certify_settled_pose")

    @staticmethod
    def stub_env(spec):
        from types import SimpleNamespace
        return SimpleNamespace(
            reach_anchor_m=lambda: (0., -0.031, 0.698), workspace=spec, socket_site=0,
            data=SimpleNamespace(site_xpos=np.array([[0.4, -0.3, 0.472]])),
            derived={"table_rest_z_offset_m": 0.004}, initial_grasp_position=np.array([0.4008, -0.1551, 0.4761]))

    def test_v1_ranges_unchanged_and_v2_gets_stage1(self):
        from dataclasses import replace
        from data_pipeline.scene_bank import Workspace, workspace_for_env, V2_STAGE1
        base = Workspace()
        v1 = workspace_for_env(self.stub_env(wsp.load_workspace()), base)
        # the pre-existing workspace-spec overrides, nothing more
        self.assertEqual((v1.plug_x_range, v1.socket_x_range, v1.socket_y_range),
                         ((0.28, 0.38), (0.38, 0.46), (-0.36, -0.20)))
        for field in ("plug_y_range", "plug_yaw_range_deg", "socket_yaw_range_deg", "reach_range_m",
                      "hand_keepout_m", "home_collision_check", "certify_settled_pose"):
            self.assertEqual(getattr(v1, field), getattr(base, field), field)
        for name in ("workspace_v2.json", "workspace_v2_cable30.json"):
            v2 = workspace_for_env(self.stub_env(wsp.load_workspace(wsp.ROOT / "configs" / name)), base)
            for field in self.RANGE_FIELDS:
                self.assertEqual(getattr(v2, field), V2_STAGE1[field], field)
        self.assertEqual(V2_STAGE1["plug_x_range"], (0.32, 0.43))
        self.assertEqual(V2_STAGE1["socket_yaw_range_deg"], (-15.0, 35.0))
        self.assertEqual(V2_STAGE1["hand_keepout_m"], 0.06)
        # an explicit override still wins
        custom = workspace_for_env(self.stub_env(wsp.load_workspace(wsp.ROOT / "configs/workspace_v2_cable30.json")),
                                   base, plug_x_range=(0.35, 0.36))
        self.assertEqual(custom.plug_x_range, (0.35, 0.36))

    def test_pick_insert_margin_field(self):
        from data_pipeline.scene_bank import Workspace, workspace_for_env, V2_STAGE1
        self.assertIsNone(Workspace().pick_insert_margin_rad)
        self.assertEqual(V2_STAGE1["pick_insert_margin_rad"], 0.09)
        base = Workspace()
        v1 = workspace_for_env(self.stub_env(wsp.load_workspace()), base)
        self.assertIsNone(v1.pick_insert_margin_rad)
        v2 = workspace_for_env(self.stub_env(wsp.load_workspace(wsp.ROOT / "configs/workspace_v2_cable30.json")), base)
        self.assertEqual(v2.pick_insert_margin_rad, 0.09)

    def test_pick_check_applies_margin_only_when_robust(self):
        from unittest import mock
        from data_pipeline import scene_bank as sb
        calls = []

        class Fake:
            def __call__(self, options, min_insert_margin=None, arm_offset=None):
                calls.append(min_insert_margin)

        ws = sb.Workspace(pick_insert_margin_rad=0.09)
        with mock.patch.object(sb, "_pick_feasibility", return_value=Fake()):
            sb._pick_check({}, ws)
            sb._pick_check({}, ws, robust=True)
            sb._pick_check({}, sb.Workspace(), robust=True)
        self.assertEqual(calls, [None, 0.09, None])

    def test_plan_grasp_prefers_robust_insert_margin(self):
        from unittest import mock
        from controllers import pick_insert_expert as pie
        planner = object.__new__(pie.GraspPlanner)
        margins = {-15.0: 0.056, -30.0: 0.13}

        def cand(plug_pos, plug_mat, flip, tilt, q_now):
            if flip != 0 or tilt not in margins:
                return None, "grasp_ik"
            return {"flip": flip, "tilt_deg": tilt, "margin": 0.2, "score": 0.3 - 0.002 * abs(tilt),
                    "q_grasp": None, "lift": [None]}, "ok"

        with mock.patch.object(pie, "GRASP_TILTS_DEG", (0.0, -15.0, 15.0, -30.0, 30.0)),                 mock.patch.object(planner, "_candidate", side_effect=cand, create=True),                 mock.patch.object(planner, "insert_feasible", side_effect=lambda f, t, *a, **k: margins[t], create=True):
            plan, log = planner.plan_grasp(np.zeros(3), np.eye(3), insert=(None, None), require_insert=True)
            self.assertEqual(plan["tilt_deg"], -30.0)   # no early stop at the marginal -15 level
            self.assertAlmostEqual(plan["insert_margin"], 0.13)
            self.assertEqual({e["tilt_deg"]: e["insert_margin"] for e in log if e["result"] == "ok"},
                             {-15.0: 0.056, -30.0: 0.13})
            margins[-30.0] = None                        # only the marginal grasp is feasible: fall back to it
            plan, _ = planner.plan_grasp(np.zeros(3), np.eye(3), insert=(None, None), require_insert=True)
            self.assertEqual(plan["tilt_deg"], -15.0)
            margins[-30.0] = 0.08                        # best available margin wins when none reach 0.10
            margins[-15.0] = 0.06
            plan, _ = planner.plan_grasp(np.zeros(3), np.eye(3), insert=(None, None), require_insert=True)
            self.assertEqual(plan["tilt_deg"], -30.0)

    def test_settled_pose_rejections_are_distinct_and_counted(self):
        from unittest import mock
        from dataclasses import replace
        from data_pipeline import scene_bank as sb
        ws = replace(sb.Workspace(), socket_x_range=(0.5, 0.5), socket_y_range=(-0.3, -0.3),
                     plug_x_range=(0.3, 0.3), plug_y_range=(-0.2, -0.2), plug_yaw_range_deg=(0., 0.),
                     socket_yaw_range_deg=(0., 0.), grasp_feasibility=False, certify_settled_pose=True,
                     home_collision_check=True,
                     arm_offset_max_rad=0.0)   # posture variation has its own tests (test_reset_variation)
        settled = {"plug_xy_m": [0.302, -0.2], "plug_yaw_deg": 10.0}
        # draws: 1 home collision, 2 settled pick fails, 3 settled noise fails, 4 accepted
        noises = iter(["plug_yaw+:grasp_ik", None])
        homes = iter(["socket_home_collision", None, None, None])
        # _pick_check runs pre-settle then post-settle on draws 2-4 (draw 1 dies on the home check)
        picks = iter([None, "grasp_ik", None, None, None, None])
        with mock.patch.object(sb, "_home_collision_check", lambda o, w, offset=None: next(homes)), \
                mock.patch.object(sb, "_pick_check", lambda o, w, robust=False, arm_offset=None: next(picks)), \
                mock.patch.object(sb, "_insert_check_full", lambda o, w: (None, settled)), \
                mock.patch.object(sb, "_pick_noise_margin_check", lambda o, w, arm_offset=None: next(noises)), \
                mock.patch.object(sb, "_cable_check", lambda o, w: None):
            options, diag = sb.sample_scene(1, ws)
        self.assertEqual(diag["draws"], 4)
        self.assertEqual(diag["rejected"], ["socket_home_collision", "settled_pick_infeasible:grasp_ik",
                                            "settled_noise_margin:plug_yaw+:grasp_ik"])
        self.assertAlmostEqual(diag["settle_drift_mm"], 2.0, places=2)
        self.assertAlmostEqual(diag["settle_drift_yaw_deg"], 10.0, places=2)

    def test_settled_options_wrap_yaw(self):
        from data_pipeline.scene_bank import _settled_options
        o = {"plug_pos_m": [0.3, -0.2, 0.324], "plug_yaw_deg": 175.0}
        s, mm, dyaw = _settled_options(o, {"plug_xy_m": [0.303, -0.196], "plug_yaw_deg": -175.0})
        self.assertAlmostEqual(dyaw, 10.0)
        self.assertAlmostEqual(s["plug_yaw_deg"], 185.0)
        self.assertAlmostEqual(mm, 5.0)

    @unittest.skipUnless((wsp.ROOT / "third_party/openarm_mujoco").exists(), "needs robot assets")
    def test_home_collision_check_on_scratch_model(self):
        from dataclasses import replace
        from data_pipeline import scene_bank as sb
        ws = replace(sb.Workspace(), table_height_m=0.32, workspace_path="configs/workspace_v2_cable30.json",
                     home_collision_check=True)
        pf = sb._pick_feasibility(ws)
        before = pf.qpos0.copy()
        pose = lambda x, y: {"socket_pos_m": [x, y, 0.472], "socket_yaw_deg": 0., "socket_tilt_deg": 0.}
        # a socket wall pressing on the open fingers (0.046 m from the hand: also inside any keep-out)
        self.assertEqual(sb._home_collision_check(pose(0.39, -0.20), ws), "socket_home_collision")
        # 0.063 m from the hand (outside the 0.06 m keep-out) but the fingers still hit the wall
        self.assertEqual(sb._home_collision_check(pose(0.37, -0.10), ws), "socket_home_collision")
        # beyond the hand: clear
        self.assertIsNone(sb._home_collision_check(pose(0.42, -0.20), ws))
        self.assertIsNone(sb._home_collision_check(pose(0.46, -0.155), ws))
        np.testing.assert_array_equal(pf.qpos0, before)


class CollectRandomStartRowTests(unittest.TestCase):
    """--start-row block arithmetic; no env, no rows generated."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "collect_random", wsp.ROOT / "scripts/collect_random.py")
        sys.path.insert(0, str(wsp.ROOT / "scripts"))
        try:
            cls.cr = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.cr)
        finally:
            sys.path.pop(0)

    def test_default_start_reproduces_old_blocks(self):
        for cursor, budget in ((0, 600), (63, 600), (64, 600), (130, 600), (599, 600)):
            block = cursor - cursor % 64
            self.assertEqual(self.cr.row_block(0, cursor, budget),
                             (block, min(64, budget - block), cursor - block))

    def test_start_row_offsets_but_keeps_relative_alignment(self):
        self.assertEqual(self.cr.row_block(5000, 0, 600), (5000, 64, 0))
        self.assertEqual(self.cr.row_block(5000, 70, 600), (5064, 64, 6))
        self.assertEqual(self.cr.row_block(5001, 64, 600), (5065, 64, 0))

    def test_last_partial_block(self):
        self.assertEqual(self.cr.row_block(5000, 590, 600), (5000 + 576, 24, 14))
        self.assertEqual(self.cr.row_block(0, 10, 20), (0, 20, 10))

    def test_run_signature_and_namespace(self):
        from types import SimpleNamespace
        a = SimpleNamespace(workspace=None, detector="noisy", validation_fraction=0.1, start_row=0)
        b = SimpleNamespace(workspace=None, detector="noisy", validation_fraction=0.1, start_row=5000)
        self.assertEqual(self.cr.run_signature(a)["start_row"], 0)
        self.assertNotEqual(self.cr.run_signature(a), self.cr.run_signature(b))
        self.assertGreater(self.cr.SEED_NAMESPACE_WIDTH, 5000)


if __name__ == "__main__":
    unittest.main()
