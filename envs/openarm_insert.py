"""OpenArm v1 bimanual scene; right joint targets, physically grasped free plug."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from connector.simulation import ConnectorMetrics, CONFIG_PATH, ROOT, rotation_vector
from envs.scene import build_model, source_manifest, scene_xml

ROBOT_CONFIG = ROOT / "configs/openarm_v1.json"


class OpenArmInsertEnv:
    def __init__(self, *, images=True, timestep=None):
        self.config = json.loads(ROBOT_CONFIG.read_text())
        self.model = build_model()
        if timestep is not None:
            if not np.isfinite(timestep) or timestep <= 0:
                raise ValueError("Invalid physics timestep")
            self.model.opt.timestep = timestep
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
        self.robot_bodies = {i for i in range(self.model.nbody) if self.model.body(i).name.startswith("openarm")}
        self.metrics = ConnectorMetrics()
        self.metric_config = json.loads(CONFIG_PATH.read_text())
        self.metrics.bind(self.model, self.data, self.metric_config, self.fingers)
        self.substeps = round(1 / self.config["command_hz"] / self.model.opt.timestep)
        self.dt = self.substeps * self.model.opt.timestep
        if abs(self.dt - 1/self.config["command_hz"]) > 1e-12:
            raise ValueError("Command interval must be an integer number of physics steps")
        self.lower = self.model.jnt_range[self.joints["right"], 0]
        self.upper = self.model.jnt_range[self.joints["right"], 1]
        self.reset()

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

    def reset(self, *, seed=0, options=None):
        options = dict(options or {})
        allowed = {"offset_x_m", "offset_y_m", "offset_z_m", "roll_deg", "pitch_deg", "yaw_deg"}
        if options.keys() - allowed:
            raise ValueError(f"Unknown reset options: {options.keys() - allowed}")
        if any(not np.isfinite(v) or abs(v) > (2. if k.endswith("_deg") else .003)
               for k,v in options.items()):
            raise ValueError("Reset offsets must be within +/-3 mm and angles within +/-2 degrees")
        self.seed, self.reset_options = int(seed), options
        mujoco.mj_resetData(self.model,self.data)
        for side in ("left","right"):
            self.data.qpos[self.qa[side]] = self.config["home_arm_rad"]
            self.data.qpos[self.fqa[side]] = self.config["initial_finger_travel_m"] if side == "right" else .025
        mujoco.mj_forward(self.model,self.data)
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
        # Only reset sets the plug pose. It is free and physically pinched thereafter.
        rot = self.data.site_xmat[self.grasp_site].reshape(3,3)
        pos = self.data.site_xpos[self.grasp_site] + rot @ np.array([.016,0,0])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat,rot.ravel())
        self.data.qpos[self.plug_qadr:self.plug_qadr+3] = pos
        self.data.qpos[self.plug_qadr+3:self.plug_qadr+7] = quat
        mujoco.mj_forward(self.model,self.data)
        self.target = np.r_[self.data.qpos[self.qa["right"]],self.config["grip_target_travel_m"]]
        self.park_target = self.data.qpos[self.qa["left"]].copy()
        self.outcome, self.hold, self.elapsed = None, 0., 0.
        self.last_action = self.target.copy()
        self.peak_force = self.peak_penetration = self.peak_robot_contact = 0.
        # Capture initial state before closure for reproducible startup playback.
        self.reset_qpos = self.data.qpos.copy()
        for _ in range(round(self.config["settle_s"]/self.model.opt.timestep)):
            self._physics_step()
        mujoco.mj_forward(self.model,self.data)
        self.episode_start_time = float(self.data.time)
        self.initial_grasp_position, self.initial_grasp_rotation = self.grasp_pose()
        self.info = self._info()
        if (self.info["grasp_slip_m"] > self.config["grasp_slip_limit_m"]
                or self.info["robot_unwanted_contact_n"] > self.config["robot_contact_abort_n"]
                or any(w.number for w in self.data.warning)):
            raise RuntimeError(f"Reset grasp did not hold: {self.info}")
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

    def _info(self):
        info = self.metrics.diagnostics()
        pos,rot = self.grasp_pose()
        plug_grasp = self.data.site_xpos[self.model.site("plug_grasp_frame").id]
        info["grasp_slip_m"] = float(np.linalg.norm(plug_grasp-pos))
        info["grasp_angle_deg"] = float(np.degrees(np.linalg.norm(rotation_vector(
            rot.T @ self.data.xmat[self.plug].reshape(3,3)))))
        info["left_park_error_rad"] = float(np.max(np.abs(
            self.data.qpos[self.qa["left"]]-self.park_target)))
        external = 0.
        wrench = np.zeros(6)
        for i,contact in enumerate(self.data.contact):
            bodies = [int(self.model.geom_bodyid[g]) for g in contact.geom]
            if not any(b in self.robot_bodies for b in bodies):
                continue
            if self.plug in bodies and any(b in self.fingers for b in bodies):
                continue
            mujoco.mj_contactForce(self.model,self.data,i,wrench)
            external += float(np.linalg.norm(wrench[:3]))
        info["robot_unwanted_contact_n"] = external
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
        self.target = self.target + np.clip(bounded-self.target,-max_delta,max_delta)
        self.last_action = self.target.copy()
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
            valid = (info["valid_pose"] and info["grasp_slip_m"] < self.config["grasp_slip_limit_m"]
                     and info["grasp_angle_deg"] < self.config["grasp_angle_limit_deg"])
            self.hold = self.hold+self.model.opt.timestep if valid else 0.
            if (any(w.number for w in self.data.warning) or not np.isfinite(self.data.qpos).all()
                    or not np.isfinite(self.data.qvel).all() or self.data.time <= previous_time
                    or info["speed_m_s"] > 1 or info["angular_speed_rad_s"] > 50):
                self.outcome = "numerical_failure"
            elif info["robot_unwanted_contact_n"] > self.config["robot_contact_abort_n"]:
                self.outcome = "robot_collision"
            elif info["max_penetration_m"] > self.metric_config["penetration_limit_m"]:
                self.outcome = "penetration_abort"
            elif info["contact_force_n"] > self.metric_config["contact_abort_n"]:
                self.outcome = "force_limit_abort"
            elif info["grasp_slip_m"] > self.config["grasp_slip_limit_m"]:
                self.outcome = "grasp_slip"
            elif info["grasp_angle_deg"] > self.config["grasp_angle_limit_deg"]:
                self.outcome = "grasp_rotation"
            elif self.hold >= self.metric_config["hold_s"]:
                self.outcome = "success"
            elif self.elapsed >= self.config["episode_limit_s"]:
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
        terminated = self.outcome not in (None,"timeout","numerical_failure")
        truncated = self.outcome in ("timeout","numerical_failure")
        return self.observe(), float(self.outcome=="success"), terminated, truncated, self.info.copy()

    def manifest(self):
        canonical_xml = scene_xml().replace(str((ROOT/"third_party/openarm_mujoco/v1/meshes").resolve()), "UPSTREAM_V1_MESHES")
        source_paths = ("envs/scene.py", "envs/openarm_insert.py", "connector/geometry.py", "connector/simulation.py", "controllers/expert.py")
        return {"schema": self.config["schema_version"], "robot": source_manifest(),
                "scene_sha256": hashlib.sha256(canonical_xml.encode()).hexdigest(),
                "source_hashes": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
                "robot_config": self.config, "contact_config": self.metric_config,
                "mujoco_version": mujoco.__version__, "physics_timestep_s": self.model.opt.timestep,
                "command_interval_s": self.dt, "seed": self.seed, "reset_options": self.reset_options,
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
