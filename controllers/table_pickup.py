"""Physical pickup diagnostic with downward rest and substep contact guards."""
import mujoco
import numpy as np
from scripts.preview_table_task import inspect_ik
from connector.simulation import rotation_vector


class PickupProbe:
    def __init__(self, env):
        self.env=env
        self.trace=[]
        self.peak_robot_force=0.
        self.peak_penetration=0.
        self.reference=None
        self.max_slip=0.
        self.max_angle=0.

    def reset(self, position=(.381,-.22,.330)):
        e=self.env; m,d=e.model,e.data
        mujoco.mj_resetData(m,d)
        # Official v1 zero-joint posture: both hands hang down, behind the table.
        for side in ('left','right'):
            d.qpos[e.qa[side]]=0
            d.qpos[e.fqa[side]]=.035 if side=='right' else .025
        d.qpos[e.plug_qadr:e.plug_qadr+3]=position
        d.qpos[e.plug_qadr+3:e.plug_qadr+7]=[1,0,0,0]
        e.target=np.r_[np.zeros(7),.035]
        e.park_target=np.zeros(7)
        mujoco.mj_forward(m,d)
        for _ in range(round(1/m.opt.timestep)):
            self.physics_step()
        mujoco.mj_forward(m,d)
        self.start_height=float(d.xpos[e.plug,2])
        self.rest_axis=d.site_xmat[e.grasp_site].reshape(3,3)[:,0].copy()

    def physics_step(self):
        e=self.env; m,d=e.model,e.data
        e._physics_step()
        force=0.; wrench=np.zeros(6)
        for index,c in enumerate(d.contact):
            bodies={int(m.geom_bodyid[g]) for g in (c.geom1,c.geom2)}
            if not bodies & e.robot_bodies:
                continue
            if e.plug in bodies and bodies & set(e.fingers):
                continue
            mujoco.mj_contactForce(m,d,index,wrench)
            force+=np.linalg.norm(wrench[:3])
            self.peak_penetration=max(self.peak_penetration,float(-c.dist))
        self.peak_robot_force=max(self.peak_robot_force,float(force))
        if force>5 or any(w.number for w in d.warning):
            raise RuntimeError(f'contact/numerical abort: {force:.4f} N')

    def relative_plug(self):
        e=self.env; d=e.data
        return d.site_xmat[e.grasp_site].reshape(3,3).T @ (d.xpos[e.plug]-d.site_xpos[e.grasp_site])

    def finger_contacts(self):
        e=self.env
        return sorted({int(e.model.geom_bodyid[g]) for c in e.data.contact
                       if e.plug in {int(e.model.geom_bodyid[h]) for h in (c.geom1,c.geom2)}
                       for g in (c.geom1,c.geom2) if int(e.model.geom_bodyid[g]) in e.fingers})

    def move(self, phase, goal, finger, seconds, frame_callback=None):
        e=self.env
        for i in range(round(seconds/e.dt)):
            e.target[:7]+=np.clip(goal-e.target[:7],-.4*e.dt,.4*e.dt)
            e.target[7]+=np.clip(finger-e.target[7],-.01*e.dt,.01*e.dt)
            for _ in range(e.substeps): self.physics_step()
            mujoco.mj_forward(e.model,e.data)
            slip=float(np.linalg.norm(self.relative_plug()-self.reference)) if self.reference is not None else 0.
            if phase in ('lift','hold'): self.max_slip=max(self.max_slip,slip)
            if phase in ('lift','hold'):
                relative_rotation=e.data.site_xmat[e.grasp_site].reshape(3,3).T @ e.data.xmat[e.plug].reshape(3,3)
                self.max_angle=max(self.max_angle,float(np.degrees(np.linalg.norm(rotation_vector(
                    relative_rotation @ self.reference_rotation.T)))))
            self.trace.append({'phase':phase,'time_s':float(e.data.time),
                'plug_height_m':float(e.data.xpos[e.plug,2]),'finger_contacts':len(self.finger_contacts()),
                'grasp_drift_m':slip,'joint_error_rad':float(np.max(np.abs(goal-e.data.qpos[e.qa['right']]))),
                'left_park_error_rad':float(np.max(np.abs(e.data.qpos[e.qa['left']]))),
                'finger_travel_m':e.data.qpos[e.fqa['right']].tolist()})
            if frame_callback and i%5==0: frame_callback(phase)

    def ik(self, position, rotation, initial):
        q,check=inspect_ik(self.env,np.array(position),rotation,initial)
        if not check['ik_converged'] or check['unexpected_contacts']:
            raise RuntimeError(f'Invalid IK endpoint: {check}')
        return q

    def run(self, *, pitch=90., height=.003, x_shift=0., position=(.381,-.22,.330), frame_callback=None, closed_travel=.006):
        self.reset(position)
        e=self.env
        if frame_callback: frame_callback('rest')
        angle=np.radians(pitch)
        rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        grasp=e.data.site_xpos[e.model.site('plug_grasp_frame').id].copy()+[x_shift,0,height]
        q=np.array([0,0,0,1.57,0,0,0.])  # scratch IK seed, not a robot reset
        # Raise behind the table before extending over it. Each is a physical move.
        for name,pos,seconds in [('raise',[.15,-.22,.30],7.),('approach',[.22,-.22,.39],5.),
                                 ('pregrasp',grasp+[0,0,.07],5.)]:
            q=self.ik(pos,rotation,q)
            self.move(name,q,.035,seconds,frame_callback)
        above=q.copy()
        down=self.ik(grasp,rotation,q)
        self.move('descend',down,.035,3.,frame_callback)
        self.move('close',down,closed_travel,4.,frame_callback)
        self.reference=self.relative_plug().copy()
        self.reference_rotation=e.data.site_xmat[e.grasp_site].reshape(3,3).T @ e.data.xmat[e.plug].reshape(3,3)
        self.move('lift',above,closed_travel,3.,frame_callback)
        self.move('hold',above,closed_travel,1.,frame_callback)
        gain=float(e.data.xpos[e.plug,2]-self.start_height)
        contacts=len(self.finger_contacts())
        return {'passed':gain>.04 and self.max_slip<.001 and self.max_angle<5 and contacts==2,
            'height_gain_m':gain,'max_grasp_drift_m':self.max_slip,'final_finger_contact_bodies':contacts,
            'max_grasp_angle_drift_deg':self.max_angle,
            'grasp_position_in_tool_m':self.reference.tolist(),
            'grasp_rotation_in_tool':self.reference_rotation.tolist(),
            'peak_unwanted_robot_force_n':self.peak_robot_force,'peak_unwanted_penetration_m':self.peak_penetration,
            'rest_tool_forward_axis_world':self.rest_axis.tolist(),'physical_end_to_end_insertion':False}
