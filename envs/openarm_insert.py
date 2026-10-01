"""OpenArm v1 bimanual scene; right joint targets, physically grasped free plug."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from connector.simulation import ConnectorMetrics, CONFIG_PATH, ROOT, rotation_vector
from connector.spec import ConnectorSpec
from envs.scene import build_model, source_manifest, scene_xml
from envs import workspace as wsp

ROBOT_CONFIG = ROOT / "configs/openarm_v1.json"

# Grasped mode: plug starts pinched, options perturb the grasp pose (held-plug expert).
LEGACY_KEYS = {"offset_x_m", "offset_y_m", "offset_z_m", "roll_deg", "pitch_deg", "yaw_deg"}
# Tabletop mode: plug lies free on the table, socket (with its fixture) is placed, gripper open.
TABLETOP_KEYS = {"plug_pos_m", "plug_yaw_deg", "socket_pos_m", "socket_yaw_deg",
                 "socket_tilt_deg", "table_height_m"}
WORKSPACE_BOX = np.array([[-0.9, -0.9, -0.05], [0.9, 0.9, 0.6]])
# Tabletop grasp latch. A table pick does not use the held-plug plug_grasp_frame pinch, so the
# hand-to-plug pose is latched once a physical grasp exists and drift is measured against that:
# both right pads touch the plug, the pads are still, the finger command sits inside the pads
# (squeezing) and has not changed for TABLETOP_GRASP_SETTLE_S. Grasped mode never uses these.
TABLETOP_GRASP_SETTLE_S = 0.1
TABLETOP_FINGER_STILL_M_S = 0.005
TABLETOP_SQUEEZE_M = 0.001


def socket_mount_home(model):
    """Home poses of the static socket body and its fixture pedestal geom, read from a fresh model."""
    socket = int(model.site_bodyid[model.site("socket_entry").id])
    fixture = model.geom("fixture").id
    return (socket, fixture, model.body_pos[socket].copy(), model.body_quat[socket].copy(),
            model.geom_pos[fixture].copy(), model.geom_quat[fixture].copy())


def carry_fixture(model, home):
    """Move the fixture geom by the same rigid transform the socket body made from its home pose,
    so the socket stays mounted on its pedestal. ``home`` is socket_mount_home() of this model."""
    socket, fixture, socket_pos, socket_quat, fixture_pos, fixture_quat = home
    mats = [np.zeros(9) for _ in range(3)]
    for mat, quat in zip(mats, (socket_quat, model.body_quat[socket], fixture_quat)):
        mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    home_rot, new_rot, fixture_rot = (m.reshape(3, 3) for m in mats)
    relative = new_rot @ home_rot.T
    model.geom_pos[fixture] = model.body_pos[socket] + relative @ (fixture_pos - socket_pos)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (relative @ fixture_rot).ravel())
    model.geom_quat[fixture] = quat


def _repo_relative(path):
    """Repo-relative POSIX path when inside the repo, else the absolute path (external specs)."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


class OpenArmInsertEnv:
    def __init__(self, *, images=True, timestep=None, workspace=None):
        self.config = json.loads(ROBOT_CONFIG.read_text())
        self.workspace = wsp.load_workspace(workspace)
        self.model = build_model(self.workspace)
        if timestep is not None:
            if not np.isfinite(timestep) or timestep <= 0:
                raise ValueError("Invalid physics timestep")
            self.model.opt.timestep = timestep
        # Right finger position servos: upstream kp (100 N/m) pinches with well under 1 N, which
        # cannot react cable loads or socket retention. The gain lives in the robot config and is
        # part of the manifest; see PHYSICS_CHANGELOG.md (workspace_v1).
        for i in (1, 2):
            a = self.model.actuator(f"right_finger{i}_ctrl").id
            self.model.actuator_gainprm[a, 0] = self.config["finger_servo_kp_n_m"]
            self.model.actuator_biasprm[a, 1] = -self.config["finger_servo_kp_n_m"]
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.images = images
        self.renderer = None
        self.joints = {side: [self.model.joint(f"openarm_{side}_joint{i}").id for i in range(1, 8)]
                       for side in ("left", "right")}
        self.qa = {side: np.array([self.model.jnt_qposadr[j] for j in ids]) for side, ids in self.joints.items()}
        self.va = {side: np.array([self.model.jnt_dofadr[j] for j in ids]) for side, ids in self.joints.items()}
        self.act = {side: np.array([self.model.actuator(f"{side}_joint{i}_ctrl").id for i in range(1,8)])
                    for side in ("left","right")}
        self.fqa = {side: np.array([self.model.jnt_qposadr[self.model.joint(f"openarm_{side}_finger_joint{i}").id]
                                   for i in (1,2)]) for side in ("left","right")}
        self.fva = {side: np.array([self.model.jnt_dofadr[self.model.joint(f"openarm_{side}_finger_joint{i}").id]
                                   for i in (1,2)]) for side in ("left","right")}
        self.fingers = tuple(self.model.body(f"openarm_right_{s}_finger").id for s in ("right","left"))
        self.plug = self.model.body("plug").id
        self.plug_qadr = self.model.jnt_qposadr[self.model.joint("plug_free").id]
        self.grasp_site = self.model.site("robot_grasp").id
        self.plug_vadr = int(self.model.jnt_dofadr[self.model.joint("plug_free").id])
        # Tabletop resets move the static socket in the model, together with the fixture pedestal
        # it is mounted on (one rigid transform). Every reset first restores these home poses, so
        # grasped-mode resets and replays always see the workspace scene.
        self.socket_site = self.model.site("socket_entry").id
        self.socket = int(self.model.site_bodyid[self.socket_site])
        self.socket_qadr = self._free_joint_qadr(self.socket)
        self.mount_home = socket_mount_home(self.model)
        (_, self.fixture_geom, self.socket_home_pos, self.socket_home_quat,
         self.fixture_home_pos, self.fixture_home_quat) = self.mount_home
        self.camera_home_pos = {n: self.model.cam_pos[self.model.camera(n).id].copy()
                                for n in ("scene_rgb", "wrist_rgb")}
        self.camera_home_quat = {n: self.model.cam_quat[self.model.camera(n).id].copy()
                                 for n in ("scene_rgb", "wrist_rgb")}
        self.light_home_dir = self.model.light_dir.copy()
        self.table_geom = self.model.geom("work_table").id
        self.table_home_rgba = self.model.geom_rgba[self.table_geom].copy()
        self.mode = "grasped"
        self.grasp_established = False
        self.drops = 0
        self._clear_tabletop_grasp()
        self.robot_bodies = {i for i in range(self.model.nbody) if self.model.body(i).name.startswith("openarm")}
        # Geoms the robot may brush while picking off the table (table top/legs, floor). In
        # tabletop mode light contact here is scored separately, not as an unwanted collision.
        self.support_geoms = {g for g in range(self.model.ngeom)
                              if any(tag in (self.model.geom(g).name or "").lower()
                                     for tag in ("table", "floor", "ground", "worksurface"))
                              or self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE}
        self.cable_bodies = {self.model.body(n).id for n in wsp.cable_body_names(self.workspace)}
        self.cable_joints = [self.model.joint(f"cable_j_{k:02d}").id
                             for k in range(self.workspace["cable"]["segments"])] if self.cable_bodies else []
        # Connector geometry (pins, openings, clearances, every derived distance) comes from the
        # spec embedded in the model; the legacy two-blade assets map onto the catalog legacy entry.
        self.connector = ConnectorSpec.from_model(self.model)
        self.derived = self.connector.derived()
        self.spawn_position = np.array(self.workspace["plug_spawn"]["mating_position_m"], float)
        leaves = self.workspace["socket_leaves"]
        self.leaf_geoms = wsp.leaf_names_in_model(self.model) if leaves["enabled"] else ()
        self.leaf_joints = [self.model.joint(f"{n}_slide").id for n in self.leaf_geoms]
        self.socket_bodies = {self.model.body("socket").id} | {
            int(self.model.geom_bodyid[self.model.geom(n).id]) for n in self.leaf_geoms}
        self.metrics = ConnectorMetrics()
        self.metric_config = json.loads(CONFIG_PATH.read_text())
        self.metric_config["success_depth_m"] = self.derived["success_depth_m"]
        if leaves["enabled"]:
            self.metric_config["leaf_penetration_limit_m"] = leaves["leaf_penetration_limit_m"]
            self.metric_config["leaf_force_abort_n"] = leaves["leaf_force_abort_n"]
            self.metric_config["contact_abort_n"] = leaves["wall_force_abort_n"]
        self.metrics.bind(self.model, self.data, self.metric_config, self.fingers, self.leaf_geoms,
                          ("cable_root_force", "cable_root_torque") if self.cable_bodies else None,
                          spec=self.connector)
        # Finger travel follows the housing width: pads touch at ``finger_contact_travel`` and the
        # insertion grip squeezes a further ``grip_squeeze_m`` (legacy 13.8 mm / 8 mm for 24 mm).
        self.finger_contact_travel = wsp.finger_travel_for_width(
            self.workspace, self.derived["housing_width_m"], self.config["finger_base_gap_m"])
        self.grip_travel = self.finger_contact_travel - self.config["grip_squeeze_m"]
        self.pickup_travel = self.finger_contact_travel - self.config["pickup_squeeze_m"]
        self.substeps = round(1 / self.config["command_hz"] / self.model.opt.timestep)
        self.dt = self.substeps * self.model.opt.timestep
        if abs(self.dt - 1/self.config["command_hz"]) > 1e-12:
            raise ValueError("Command interval must be an integer number of physics steps")
        self.lower = self.model.jnt_range[self.joints["right"], 0]
        self.upper = self.model.jnt_range[self.joints["right"], 1]
        self.reset()
        # Fixed mount of the right arm in world coordinates (parent of joint1's body);
        # scene_bank.workspace_for_env() centres the reach annulus on it.
        joint1_body = int(self.model.jnt_bodyid[self.joints["right"][0]])
        self.base_pos_right = self.data.xpos[int(self.model.body_parentid[joint1_body])].copy()

    def reach_anchor_m(self):
        """World position to center a reach annulus on for the right arm."""
        return self.base_pos_right.copy()

    def _free_joint_qadr(self, body):
        for j in range(self.model.njnt):
            if int(self.model.jnt_bodyid[j]) == body and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_qposadr[j])
        return None

    @staticmethod
    def _yaw_tilt_matrix(yaw_deg, tilt_deg):
        """Rz(yaw) @ Ry(-tilt): maps world +x to [cy*ct, sy*ct, st], matching the sampler."""
        y, t = math.radians(yaw_deg), math.radians(-tilt_deg)
        rz = np.array([[math.cos(y), -math.sin(y), 0.], [math.sin(y), math.cos(y), 0.], [0., 0., 1.]])
        ry = np.array([[math.cos(t), 0., math.sin(t)], [0., 1., 0.], [-math.sin(t), 0., math.cos(t)]])
        return rz @ ry

    @staticmethod
    def _quat(mat):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(mat, dtype=float).ravel())
        return quat

    @staticmethod
    def _mat(quat):
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
        return mat.reshape(3, 3)

    def _restore_scene(self):
        """Undo every model-level edit a tabletop reset or visual randomization makes."""
        m = self.model
        m.body_pos[self.socket], m.body_quat[self.socket] = self.socket_home_pos, self.socket_home_quat
        m.geom_pos[self.fixture_geom] = self.fixture_home_pos
        m.geom_quat[self.fixture_geom] = self.fixture_home_quat
        for name in ("scene_rgb", "wrist_rgb"):
            cam = m.camera(name).id
            m.cam_pos[cam], m.cam_quat[cam] = self.camera_home_pos[name], self.camera_home_quat[name]
        m.light_dir[:] = self.light_home_dir
        m.geom_rgba[self.table_geom] = self.table_home_rgba

    def _place_socket(self, entry_pos, yaw_deg, tilt_deg):
        """Put the socket_entry site at entry_pos with the requested mating axis, and move the
        fixture pedestal with it (same rigid transform), so the socket stays mounted."""
        mat = self._yaw_tilt_matrix(yaw_deg, tilt_deg)
        entry_pos = np.asarray(entry_pos, dtype=float)
        if self.socket_qadr is not None:
            raise ValueError("Tabletop placement expects the workspace's static socket")
        if int(self.model.body_parentid[self.socket]) != 0:
            raise ValueError("Static socket must be a child of worldbody to be repositioned")
        # Static body: move it in the model. manifest() hashes the unmodified XML; the applied
        # pose is in reset_options and manifest()["socket_pose"].
        self.model.body_quat[self.socket] = self._quat(mat)
        self.model.body_pos[self.socket] = entry_pos
        mujoco.mj_forward(self.model, self.data)
        self.model.body_pos[self.socket] += entry_pos - self.data.site_xpos[self.socket_site]
        carry_fixture(self.model, self.mount_home)
        mujoco.mj_forward(self.model, self.data)
        achieved = self.data.site_xmat[self.socket_site].reshape(3, 3)[:, 0]
        angle = math.degrees(math.acos(float(np.clip(np.dot(achieved, mat[:, 0]), -1., 1.))))
        if angle > 5.:
            raise ValueError(
                f"socket_entry local +x is {angle:.1f} deg from the requested mating axis; "
                "the socket's home orientation does not match the sampler convention "
                "(home mating axis assumed to be world +x).")

    def solve_ik(self, position, orientation, initial=None):
        """Privileged expert IK; modifies scratch data only, never live robot qpos."""
        m, d = self.model, self.ik_data
        d.qpos[:] = self.data.qpos
        if initial is not None:
            d.qpos[self.qa["right"]] = initial
        jp, jr = np.zeros((3,m.nv)), np.zeros((3,m.nv))
        for _ in range(100):
            mujoco.mj_forward(m,d)
            error = np.r_[position - d.site_xpos[self.grasp_site],
                          rotation_vector(orientation @ d.site_xmat[self.grasp_site].reshape(3,3).T)]
            if np.linalg.norm(error[:3]) < 2e-6 and np.linalg.norm(error[3:]) < 2e-5:
                return d.qpos[self.qa["right"]].copy()
            mujoco.mj_jacSite(m,d,jp,jr,self.grasp_site)
            jac = np.vstack((jp[:,self.va["right"]],jr[:,self.va["right"]]))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6)*1e-6, error)
            d.qpos[self.qa["right"]] = np.clip(d.qpos[self.qa["right"]] +
                                              np.clip(delta,-.05,.05), self.lower+.001,self.upper-.001)
        raise ValueError(f"IK target unreachable: translation residual {np.linalg.norm(error[:3]):.6g} m")

    def place_plug(self, position, quat_wxyz, cable_bulge=None):
        """Write the plug free-joint pose and lay the cable consistently from it to the anchor.

        The only sanctioned way to position the plug outside physics (reset/probe reset).
        ``cable_bulge`` is the world direction the slack arc bows toward (default: the spec's
        table layout direction); pass ``(0, 0, -1)`` for a hanging loop when the plug is held.
        """
        self.data.qpos[self.plug_qadr:self.plug_qadr+3] = position
        self.data.qpos[self.plug_qadr+3:self.plug_qadr+7] = quat_wxyz
        if self.cable_bodies:
            wsp.write_cable_qpos(self.model, self.data, self.workspace, cable_bulge)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, *, seed=0, options=None):
        options = dict(options or {})
        allowed = LEGACY_KEYS | TABLETOP_KEYS | {"visual"}
        if options.keys() - allowed:
            raise ValueError(f"Unknown reset options: {options.keys() - allowed}")
        scene_keys = options.keys() & (TABLETOP_KEYS | {"visual"})
        legacy_keys = options.keys() & LEGACY_KEYS
        if scene_keys and legacy_keys:
            raise ValueError(f"Mixed reset modes: tabletop {sorted(scene_keys)} with grasped {sorted(legacy_keys)}")
        mode = "tabletop" if scene_keys else "grasped"
        for key, value in options.items():
            if key == "visual":
                if not isinstance(value, dict):
                    raise ValueError("visual must be a dict")
            elif key.endswith("_pos_m"):
                array = np.asarray(value, dtype=float)
                if array.shape != (3,) or not np.isfinite(array).all():
                    raise ValueError(f"{key} must be three finite metres")
                if not (WORKSPACE_BOX[0] <= array).all() or not (array <= WORKSPACE_BOX[1]).all():
                    raise ValueError(f"{key} {array.tolist()} outside the workspace box {WORKSPACE_BOX.tolist()}")
            elif key in LEGACY_KEYS:
                if not np.isfinite(value) or abs(value) > (2. if key.endswith("_deg") else .003):
                    raise ValueError("Reset offsets must be within +/-3 mm and angles within +/-2 degrees")
            elif not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
        self.mode = mode
        self.seed, self.reset_options = int(seed), options
        self._restore_scene()
        mujoco.mj_resetData(self.model,self.data)
        self.grasp_established, self.drops = False, 0
        self._clear_tabletop_grasp()
        for side in ("left","right"):
            # Tabletop: both arms at the official v1 zero pose with the right hand open (as the
            # table pickup); grasped: the validated held-plug home pose.
            self.data.qpos[self.qa[side]] = (self.config["home_arm_rad"] if side == "right" or mode == "grasped"
                                             else np.zeros(7))
            self.data.qpos[self.fqa[side]] = (.025 if side == "left" else
                                              self._open_travel() if mode == "tabletop" else self.finger_contact_travel)
        mujoco.mj_forward(self.model,self.data)
        if mode == "tabletop":
            return self._reset_tabletop(options)
        if options:
            pos = self.data.site_xpos[self.grasp_site].copy()
            pos += np.array([options.get("offset_x_m",0),options.get("offset_y_m",0),options.get("offset_z_m",0)])
            rot = self.data.site_xmat[self.grasp_site].reshape(3,3).copy()
            # World-axis Rz(yaw) Ry(pitch) Rx(roll), about the grasp site.
            r,p,y = np.radians([options.get(k,0) for k in ("roll_deg","pitch_deg","yaw_deg")])
            rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
            ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
            rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
            rot = rz @ ry @ rx @ rot
            self.data.qpos[self.qa["right"]] = self.solve_ik(pos,rot)
            mujoco.mj_forward(self.model,self.data)
        # Only reset sets the plug pose. It is free and physically pinched thereafter. The mating
        # frame sits at minus the plug's grasp offset from the robot grasp site (16 mm ahead).
        rot = self.data.site_xmat[self.grasp_site].reshape(3,3)
        pos = self.data.site_xpos[self.grasp_site] - rot @ np.array(self.connector.plug.grasp_offset_m)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat,rot.ravel())
        self.place_plug(pos, quat, cable_bulge=(0, 0, -1))
        return self._finalize_reset(self.grip_travel)

    def _open_travel(self):
        if "grip_open_travel_m" not in self.config:
            raise ValueError("configs/openarm_v1.json must define grip_open_travel_m "
                             "(the finger travel that clears the plug) for tabletop resets")
        return float(self.config["grip_open_travel_m"])

    def _reset_tabletop(self, options):
        """Plug lying free on the table, socket placed, gripper open and empty."""
        table_z = float(options.get("table_height_m", self.workspace["frames"]["table_top_z_m"]))
        if "socket_pos_m" in options:
            self._place_socket(options["socket_pos_m"],
                               float(options.get("socket_yaw_deg", 0.)),
                               float(options.get("socket_tilt_deg", 0.)))
        if "plug_pos_m" in options:
            plug_pos = np.asarray(options["plug_pos_m"], dtype=float)
            if plug_pos[2] < table_z:
                raise ValueError("plug_pos_m is below the table surface")
            # place_plug also lays the cable from the new pose to the appliance anchor.
            self.place_plug(plug_pos, self._quat(self._yaw_tilt_matrix(float(options.get("plug_yaw_deg", 0.)), 0.)))
        if options.get("visual"):
            self.apply_visual(options["visual"])
        # The gripper starts empty, so no pinch is imposed and no grasp is asserted.
        return self._finalize_reset(self._open_travel(), require_grasp=False, table_height_m=table_z)

    def apply_visual(self, visual):
        """Optional domain randomization of light, table colour and cameras (model-level; undone by
        the next reset). Silently ignores keys the scene lacks."""
        m = self.model
        if "light_azimuth_deg" in visual and m.nlight:
            az, el = math.radians(visual["light_azimuth_deg"]), math.radians(visual.get("light_elevation_deg", 60.))
            m.light_dir[0] = [-math.cos(el)*math.cos(az), -math.cos(el)*math.sin(az), -math.sin(el)]
        if "table_hue_shift" in visual:
            m.geom_rgba[self.table_geom, :3] = np.clip(self.table_home_rgba[:3] + visual["table_hue_shift"], 0., 1.)
        for name in ("scene_rgb", "wrist_rgb"):
            cam = m.camera(name).id
            if "camera_offset_m" in visual:
                m.cam_pos[cam] = self.camera_home_pos[name] + np.asarray(visual["camera_offset_m"], dtype=float)
            if "camera_rotation_deg" in visual:
                delta = self._yaw_tilt_matrix(visual["camera_rotation_deg"][2], visual["camera_rotation_deg"][1])
                m.cam_quat[cam] = self._quat(delta @ self._mat(self.camera_home_quat[name]))
        mujoco.mj_forward(m, self.data)

    def _finalize_reset(self, grip_travel, *, require_grasp=True, table_height_m=0.):
        self.target = np.r_[self.data.qpos[self.qa["right"]],grip_travel]
        self.park_target = self.data.qpos[self.qa["left"]].copy()
        self.outcome, self.hold, self.elapsed = None, 0., 0.
        self.last_action = self.target.copy()
        self.peak_force = self.peak_penetration = self.peak_robot_contact = 0.
        self.peak_support_contact = 0.
        # Capture initial state before closure for reproducible startup playback.
        self.reset_qpos = self.data.qpos.copy()
        for _ in range(round(self.config["settle_s"]/self.model.opt.timestep)):
            self._physics_step()
        mujoco.mj_forward(self.model,self.data)
        self.episode_start_time = float(self.data.time)
        self.initial_grasp_position, self.initial_grasp_rotation = self.grasp_pose()
        self.info = self._info()
        if require_grasp:
            if (self.info["grasp_slip_m"] > self.config["grasp_slip_limit_m"]
                    or self.info["robot_unwanted_contact_n"] > self.config["robot_contact_abort_n"]
                    or any(w.number for w in self.data.warning)):
                raise RuntimeError(f"Reset grasp did not hold: {self.info}")
            self.grasp_established = True
        else:
            if (self.info["robot_unwanted_contact_n"] > self.config["robot_contact_abort_n"]
                    or any(w.number for w in self.data.warning)):
                raise RuntimeError(f"Reset scene is in collision: {self.info}")
            # Tabletop: assert the plug settled where it was put instead of asserting a grasp.
            plug_z = float(self.data.qpos[self.plug_qadr+2])
            if plug_z < table_height_m - .01:
                raise RuntimeError(f"Plug fell off the table during settle (z={plug_z:.4f})")
            if np.linalg.norm(self.data.qvel[self.plug_vadr:self.plug_vadr+6]) > .05:
                raise RuntimeError("Scene had not settled at the end of reset")
        self.info["grasp_established"] = bool(self.grasp_established)
        self.reset_state = self.data.qpos.copy()
        self.reset_velocity = self.data.qvel.copy()
        return self.observe(), self.info.copy()

    def _physics_step(self):
        m,d,c = self.model,self.data,self.config
        for side in ("left","right"):
            desired = self.target[:7] if side=="right" else self.park_target
            torque = (np.array(c["kp_nm_rad"])*(desired-d.qpos[self.qa[side]]) -
                      np.array(c["kd_nms_rad"])*d.qvel[self.va[side]] + d.qfrc_bias[self.va[side]])
            ids = self.act[side]
            d.ctrl[ids] = np.clip(torque, m.actuator_forcerange[ids,0], m.actuator_forcerange[ids,1])
        for i in range(2):
            # Upstream v1 has force actuators on left fingers and position servos on right.
            d.ctrl[m.actuator(f"right_finger{i+1}_ctrl").id] = self.target[7]
            d.ctrl[m.actuator(f"left_finger{i+1}_ctrl").id] = (
                100*(.025-d.qpos[self.fqa["left"][i]])-2*d.qvel[self.fva["left"][i]])
        d.qfrc_applied[:] = 0
        d.xfrc_applied[:] = 0
        mujoco.mj_step(m,d)

    def grasp_pose(self):
        return (self.data.site_xpos[self.grasp_site].copy(),
                self.data.site_xmat[self.grasp_site].reshape(3,3).copy())

    # ---------------------------------------------------- tabletop grasp latch
    def _clear_tabletop_grasp(self):
        self._grasp_reference = None       # (plug position, rotation) in the robot_grasp frame
        self._grasp_candidate_s = 0.
        self._latched_finger_command = None

    def plug_in_hand(self):
        """Current plug body pose in the robot_grasp site frame: (position (3,), rotation (3,3)).
        p_plug_world = p_site + R_site @ position; R_plug_world = R_site @ rotation."""
        rot = self.data.site_xmat[self.grasp_site].reshape(3,3)
        return (rot.T @ (self.data.xpos[self.plug] - self.data.site_xpos[self.grasp_site]),
                rot.T @ self.data.xmat[self.plug].reshape(3,3))

    def latched_grasp(self):
        """Tabletop mode: plug_in_hand() as latched when the current physical grasp was
        established (copies), or None while no grasp is established. Tabletop
        info["grasp_slip_m"] / ["grasp_angle_deg"] are measured against it. None in grasped mode."""
        if self._grasp_reference is None:
            return None
        return self._grasp_reference[0].copy(), self._grasp_reference[1].copy()

    def _update_tabletop_grasp(self, info):
        """Latch, drop and release bookkeeping for a physical table pick."""
        limit = self.config["grasp_slip_limit_m"]
        command = float(self.target[7])
        if self.grasp_established:
            if command > self._latched_finger_command + 1e-4:
                # The command opened: a deliberate release, not a drop.
                self.grasp_established = False
                self._grasp_reference = None
            elif info["grasp_slip_m"] > limit:
                self.grasp_established = False
                self._grasp_reference = None
                self.drops += 1
            self._grasp_candidate_s = 0.
            return
        travel = self.data.qpos[self.fqa["right"]]
        candidate = (info["plug_finger_contacts"] == 2
                     and float(np.max(np.abs(self.data.qvel[self.fva["right"]]))) < TABLETOP_FINGER_STILL_M_S
                     and command < float(np.min(travel)) - TABLETOP_SQUEEZE_M)
        self._grasp_candidate_s = self._grasp_candidate_s + self.model.opt.timestep if candidate else 0.
        if self._grasp_candidate_s >= TABLETOP_GRASP_SETTLE_S:
            self._grasp_reference = self.plug_in_hand()
            self._latched_finger_command = command
            self.grasp_established = True
            self._grasp_candidate_s = 0.
            info["grasp_slip_m"], info["grasp_angle_deg"] = 0., 0.

    def _info(self):
        info = self.metrics.diagnostics()
        pos,rot = self.grasp_pose()
        plug_grasp = self.data.site_xpos[self.model.site("plug_grasp_frame").id]
        info["grasp_slip_m"] = float(np.linalg.norm(plug_grasp-pos))
        info["grasp_angle_deg"] = float(np.degrees(np.linalg.norm(rotation_vector(
            rot.T @ self.data.xmat[self.plug].reshape(3,3)))))
        info["left_park_error_rad"] = float(np.max(np.abs(
            self.data.qpos[self.qa["left"]]-self.park_target)))
        info["grip_travel_m"] = float(self.data.qpos[self.fqa["right"]].mean())
        info["grip_force_n"] = float(np.mean([abs(self.data.actuator_force[
            self.model.actuator(f"right_finger{i+1}_ctrl").id]) for i in range(2)]))
        external = support = 0.
        cable_robot = 0.
        wrench = np.zeros(6)
        pads = set()
        for i,contact in enumerate(self.data.contact):
            bodies = [int(self.model.geom_bodyid[g]) for g in contact.geom]
            if not any(b in self.robot_bodies for b in bodies):
                continue
            if self.plug in bodies and any(b in self.fingers for b in bodies):
                pads.update(b for b in bodies if b in self.fingers)
                continue
            mujoco.mj_contactForce(self.model,self.data,i,wrench)
            if any(b in self.cable_bodies for b in bodies):
                # Brushing the cable is not a collision failure; it is reported, not aborted.
                cable_robot += float(np.linalg.norm(wrench[:3]))
                continue
            if any(int(g) in self.support_geoms for g in contact.geom):
                support += float(np.linalg.norm(wrench[:3]))
            else:
                external += float(np.linalg.norm(wrench[:3]))
        # Grasped mode keeps the original accounting: any robot contact but the pinch and the
        # cable is unwanted. Tabletop mode reports table/floor contact separately.
        info["robot_unwanted_contact_n"] = external if self.mode == "tabletop" else external + support
        info["robot_support_contact_n"] = support
        info["cable_robot_contact_n"] = cable_robot
        info["leaf_travel_m"] = [float(self.data.qpos[self.model.jnt_qposadr[j]]) for j in self.leaf_joints]
        if self.mode == "tabletop":
            # Tabletop grasps are scored against the latched hand-to-plug pose, not the
            # held-plug plug_grasp_frame.
            info["plug_finger_contacts"] = len(pads)
            if self._grasp_reference is not None:
                relative_position, relative_rotation = self.plug_in_hand()
                info["grasp_slip_m"] = float(np.linalg.norm(relative_position - self._grasp_reference[0]))
                info["grasp_angle_deg"] = float(np.degrees(np.linalg.norm(rotation_vector(
                    relative_rotation @ self._grasp_reference[1].T))))
        info["grasp_established"] = bool(self.grasp_established)
        info["drops"] = int(self.drops)
        info["reset_mode"] = self.mode
        info["outcome"] = self.outcome or "running"
        return info

    def observe(self):
        obs = {"state": np.r_[self.data.qpos[self.qa["left"]],
                              self.data.qpos[self.fqa["left"]].mean(),
                              self.data.qpos[self.qa["right"]],
                              self.data.qpos[self.fqa["right"]].mean()].astype(np.float32),
               "timestamp_s": float(self.data.time), "instruction": self.config["instruction"]}
        if self.images:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=self.config["image_height"],
                                               width=self.config["image_width"])
            opt = mujoco.MjvOption()
            opt.geomgroup[3] = 0
            opt.sitegroup[:] = 0
            obs["images"] = {}
            for name in ("scene_rgb","wrist_rgb"):
                self.renderer.update_scene(self.data,camera=name,scene_option=opt)
                obs["images"][name] = self.renderer.render().copy()
        return obs

    def step(self, action):
        if self.outcome:
            raise RuntimeError("Episode ended; call reset before stepping")
        a = np.asarray(action,dtype=float)
        if a.shape != (8,) or not np.isfinite(a).all():
            raise ValueError("Action must be 7 finite right joint targets (rad) and finger travel (m)")
        bounded = np.r_[np.clip(a[:7],self.lower,self.upper),np.clip(a[7],0,.044)]
        max_delta = np.r_[np.full(7,self.config["joint_target_rate_rad_s"]*self.dt),
                          self.config["finger_target_rate_m_s"]*self.dt]
        finger_command = float(self.target[7])
        self.target = self.target + np.clip(bounded-self.target,-max_delta,max_delta)
        self.last_action = self.target.copy()
        tabletop = self.mode == "tabletop"
        if tabletop and self.target[7] != finger_command:
            self._grasp_candidate_s = 0.   # the latch needs a steady finger command
        limit_s = self.config["tabletop_episode_limit_s"] if tabletop else self.config["episode_limit_s"]
        max_drops = self.config.get("max_drops", 2)
        count = 0
        interval_peak_force = 0.
        for _ in range(self.substeps):
            previous_time = float(self.data.time)
            self._physics_step()
            mujoco.mj_forward(self.model,self.data)
            count += 1
            info = self._info()
            interval_peak_force = max(interval_peak_force, info["contact_force_n"])
            self.elapsed += self.model.opt.timestep
            self.peak_force = max(self.peak_force,info["contact_force_n"])
            self.peak_penetration = max(self.peak_penetration,info["max_penetration_m"])
            self.peak_robot_contact = max(self.peak_robot_contact,info["robot_unwanted_contact_n"])
            self.peak_support_contact = max(self.peak_support_contact,info["robot_support_contact_n"])
            if tabletop:
                # The plug starts on the table, so slip/rotation only mean something once the
                # latched grasp exists; a drop counts and aborts after max_drops.
                self._update_tabletop_grasp(info)
                slipped = self.drops > max_drops
                rotated = False
                valid = (info["valid_pose"] and self.grasp_established
                         and info["grasp_slip_m"] < self.config["grasp_slip_limit_m"]
                         and info["grasp_angle_deg"] < self.config["grasp_angle_limit_deg"])
            else:
                slipped = info["grasp_slip_m"] > self.config["grasp_slip_limit_m"]
                rotated = info["grasp_angle_deg"] > self.config["grasp_angle_limit_deg"]
                valid = (info["valid_pose"] and info["grasp_slip_m"] < self.config["grasp_slip_limit_m"]
                         and info["grasp_angle_deg"] < self.config["grasp_angle_limit_deg"])
            self.hold = self.hold+self.model.opt.timestep if valid else 0.
            if (any(w.number for w in self.data.warning) or not np.isfinite(self.data.qpos).all()
                    or not np.isfinite(self.data.qvel).all() or self.data.time <= previous_time
                    or info["speed_m_s"] > 1 or info["angular_speed_rad_s"] > 50):
                self.outcome = "numerical_failure"
            elif info["robot_unwanted_contact_n"] > self.config["robot_contact_abort_n"]:
                self.outcome = "robot_collision"
            elif tabletop and info["robot_support_contact_n"] > self.config.get("table_contact_abort_n", 30.):
                self.outcome = "table_collision"
            elif info["max_penetration_m"] > self.metric_config["penetration_limit_m"]:
                self.outcome = "penetration_abort"
            elif info["contact_force_n"] > self.metric_config["contact_abort_n"]:
                self.outcome = "force_limit_abort"
            elif info["leaf_contact_force_n"] > self.metric_config.get("leaf_force_abort_n", math.inf):
                self.outcome = "leaf_force_abort"
            elif info["leaf_penetration_m"] > self.metric_config.get("leaf_penetration_limit_m", math.inf):
                self.outcome = "penetration_abort"
            elif slipped:
                self.outcome = "grasp_slip"
            elif rotated:
                self.outcome = "grasp_rotation"
            elif self.hold >= self.metric_config["hold_s"]:
                self.outcome = "success"
            elif self.elapsed >= limit_s:
                self.outcome = "timeout"
            if self.outcome:
                break
        self.info = self._info()
        self.info.update({"executed_dt_s": count*self.model.opt.timestep,
                          "interval_peak_contact_force_n": interval_peak_force,
                          "action_clipped": bool(np.any(a != self.last_action)),
                          "applied_action": self.last_action.copy(),
                          "peak_contact_force_n": self.peak_force,
                "peak_penetration_m": self.peak_penetration})
        self.info["robot_unwanted_contact_peak_n"] = self.peak_robot_contact
        self.info["robot_support_contact_peak_n"] = self.peak_support_contact
        terminated = self.outcome not in (None,"timeout","numerical_failure")
        truncated = self.outcome in ("timeout","numerical_failure")
        return self.observe(), float(self.outcome=="success"), terminated, truncated, self.info.copy()

    def manifest(self):
        canonical_xml = scene_xml(self.workspace).replace(str((ROOT/"third_party/openarm_mujoco/v1/meshes").resolve()), "UPSTREAM_V1_MESHES")
        source_paths = ("envs/scene.py", "envs/workspace.py", "envs/openarm_insert.py", "connector/geometry.py",
                        "connector/simulation.py", "connector/spec.py", "connector/catalog.py", "connector/slab.py",
                        "connector/builder.py", "controllers/expert.py")
        workspace = {k: v for k, v in self.workspace.items() if not k.startswith("_")}
        return {"schema": self.config["schema_version"], "robot": source_manifest(),
                "scene_sha256": hashlib.sha256(canonical_xml.encode()).hexdigest(),
                "connector": {"plug": self.connector.plug.name, "socket": self.connector.socket.name,
                              "spec": self.connector.to_dict(), "derived": self.derived,
                              "finger_contact_travel_m": self.finger_contact_travel,
                              "grip_travel_m": self.grip_travel, "leaf_geoms": list(self.leaf_geoms)},
                "source_hashes": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
                "workspace": {"path": _repo_relative(self.workspace["_path"]),
                              "sha256": self.workspace["_sha256"], "spec": workspace},
                "robot_config": self.config, "contact_config": self.metric_config,
                "mujoco_version": mujoco.__version__, "physics_timestep_s": self.model.opt.timestep,
                "command_interval_s": self.dt, "seed": self.seed, "reset_options": self.reset_options,
                "reset_mode": self.mode,
                "socket_placement": "model_body_pos_with_fixture",
                "socket_pose": {"pos_m": self.data.site_xpos[self.socket_site].tolist(),
                                "mat": self.data.site_xmat[self.socket_site].tolist()},
                "action_names": [f"openarm_right_joint{i}" for i in range(1,8)]+["right_finger_travel"],
                "state_order": "left joint1..7, left mean finger travel, right joint1..7, right mean finger travel",
                "grasp": "physical_finger_contact; no weld or plug force",
                "controller": "joint PD plus model bias feedforward through capped upstream torque actuators",
                "camera_calibration": {name: {"position_body_m": self.model.cam_pos[self.model.camera(name).id].tolist(),
                    "quaternion_wxyz": self.model.cam_quat[self.model.camera(name).id].tolist(),
                    "fovy_deg": float(self.model.cam_fovy[self.model.camera(name).id]),
                    "parent_body": self.model.body(self.model.cam_bodyid[self.model.camera(name).id]).name}
                    for name in ("scene_rgb","wrist_rgb")}}

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None
