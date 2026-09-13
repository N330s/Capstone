"""Privileged scripted expert; outputs the SAME joint targets exposed to a policy."""
import numpy as np
import mujoco
from connector.simulation import rotation_vector


class InsertionExpert:
    def __init__(self, env, *, probe_offset_y_m=0.0):
        self.env = env
        self.phase = "align"
        self.retries = 0
        self.phase_time = 0.0
        self.forward = -.022
        self.last_depth = -1.
        self.stall_time = 0.
        self.probe_offset_y_m = probe_offset_y_m
        self.jacp = np.zeros((3, env.model.nv))
        self.jacr = np.zeros((3, env.model.nv))

    def action(self):
        e,m,d = self.env,self.env.model,self.env.data
        info = e.info
        entry = d.site_xpos[m.site("socket_entry").id]
        basis = d.site_xmat[m.site("socket_entry").id].reshape(3,3)
        self.phase_time += e.dt
        # Closed-loop privileged pose feedback compensates calibrated grasp offset.
        if self.phase == "align":
            self.forward = -.022
            if self.phase_time > .5 and abs(info["offset_y_m"]) < .0001 and abs(info["offset_z_m"]) < .0001 and info["orientation_error_deg"] < .2:
                self.phase,self.phase_time = "insert",0.
        elif self.phase == "insert":
            self.forward = min(.00015,self.forward+.008*e.dt)
            depth = info["insertion_depth_m"]
            self.stall_time = self.stall_time+e.dt if abs(depth-self.last_depth) < 1e-5 and info["contact_force_n"]>.8 else 0.
            self.last_depth = depth
            if (self.stall_time > .4 or info.get("interval_peak_contact_force_n",0) > .3) and not info["valid_pose"] and self.retries < 2:
                self.phase,self.phase_time = "withdraw",0.
                self.retries += 1
                self.forward = min(self.forward, -.018)
            if info["valid_pose"]:
                self.phase = "hold"
        elif self.phase == "withdraw":
            self.forward = max(-.022,self.forward-.015*e.dt)
            if self.forward <= -.022:
                self.phase,self.phase_time,self.stall_time = "align",0.,0.
        elif self.phase == "hold":
            self.forward = .00003
        desired_plug = entry + basis @ np.array([self.forward,0,0])
        # Evaluation-only fault injection: intentionally probe off-axis, then
        # withdraw/re-align. Do not label the initial faulty probe as expert BC.
        if self.phase == "insert" and self.retries == 0:
            desired_plug += basis[:,1]*self.probe_offset_y_m
        delta_pos = desired_plug-d.xpos[e.plug]
        delta_rot = rotation_vector(basis @ d.xmat[e.plug].reshape(3,3).T)
        # Increment target based on live pose error. Never mutate live qpos.
        mujoco.mj_jacSite(m,d,self.jacp,self.jacr,e.grasp_site)
        jac = np.vstack([self.jacp[:,e.va["right"]],self.jacr[:,e.va["right"]]])
        error = np.r_[np.clip(delta_pos,-.001,.001),np.clip(delta_rot,-.01,.01)]
        change = jac.T @ np.linalg.solve(jac@jac.T+np.eye(6)*1e-5,error)
        base = d.qpos[e.qa["right"]] if self.phase == "withdraw" else e.target[:7]
        return np.r_[base+.3*change,e.config["grip_target_travel_m"]]
