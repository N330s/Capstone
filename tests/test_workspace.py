"""Workspace spec tests; no robot assets needed."""
import json
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


if __name__ == "__main__":
    unittest.main()
