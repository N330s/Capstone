"""Continuous physical pickup and insertion in an explicitly rotated socket variant."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image
from envs.openarm_insert import OpenArmInsertEnv
from controllers.table_pickup import PickupProbe
from controllers.carry_path import plan_carry
from connector.simulation import rotation_vector
from scripts.preview_table_task import aim_camera


class TransportProbe(PickupProbe):
    def __init__(self,env):
        super().__init__(env)
        self.carrying=False
        self.inserting=False
        self.peak_socket_force=0.
        self.peak_socket_penetration=0.
        self.phase='settle'
        self.command_trace=[]
        self.physics_ticks=0

    def physics_step(self):
        super().physics_step()
        self.physics_ticks+=1
        if self.physics_ticks%self.env.substeps==0:
            self.command_trace.append({'phase':self.phase,'time_s':float(self.env.data.time),
                'command':self.env.target.copy().tolist(),'qpos':self.env.data.qpos.copy().tolist(),
                'qvel':self.env.data.qvel.copy().tolist()})
        if not self.carrying:return
        e=self.env; m,d=e.model,e.data
        total=0.; wrench=np.zeros(6)
        for i,c in enumerate(d.contact):
            bodies={int(m.geom_bodyid[g]) for g in (c.geom1,c.geom2)}
            if e.plug not in bodies or bodies & set(e.fingers):continue
            mujoco.mj_contactForce(m,d,i,wrench)
            force=float(np.linalg.norm(wrench[:3]))
            if m.body('socket').id not in bodies:
                if force>.1:raise RuntimeError('Carried plug contacted table/fixture/robot')
            elif not self.inserting and force>.1:
                raise RuntimeError('Socket collision during transport')
            else:
                total+=force
                self.peak_socket_penetration=max(self.peak_socket_penetration,float(-c.dist))
        self.peak_socket_force=max(self.peak_socket_force,total)
        if total>5 or self.peak_socket_penetration>.0002:raise RuntimeError('Socket force/penetration abort')

    def check_grasp(self):
        e=self.env
        slip=float(np.linalg.norm(self.relative_plug()-self.reference))
        relative=e.data.site_xmat[e.grasp_site].reshape(3,3).T@e.data.xmat[e.plug].reshape(3,3)
        angle=float(np.degrees(np.linalg.norm(rotation_vector(relative@self.reference_rotation.T))))
        self.max_slip=max(self.max_slip,slip);self.max_angle=max(self.max_angle,angle)
        if slip>.001 or angle>5 or len(self.finger_contacts())!=2:raise RuntimeError('Grasp lost or slipped in transport/insertion')

    def execute_waypoint(self,goal,callback=None):
        self.phase='transport'
        e=self.env
        start=e.target[:7].copy()
        duration=max(2.,1.5*float(np.max(np.abs(goal-start)))/.2)
        steps=int(np.ceil(duration/e.dt))
        for i in range(steps+25):
            fraction=min(1.,(i+1)/steps)
            blend=fraction*fraction*(3-2*fraction)
            # Synchronize all joints along the collision-checked segment. The old
            # per-joint saturating ramp traces a different curve in joint space.
            e.target[:7]=start+blend*(goal-start)
            for _ in range(e.substeps):self.physics_step()
            mujoco.mj_forward(e.model,e.data)
            self.check_grasp()
            self.trace.append({'phase':'transport','time_s':float(e.data.time),
                'joint_error_rad':float(np.max(np.abs(e.target[:7]-e.data.qpos[e.qa['right']]))),
                'grasp_drift_m':float(np.linalg.norm(self.relative_plug()-self.reference))})
            if callback and i%5==0:callback('transport')

    def move(self,phase,goal,finger,seconds,frame_callback=None):
        self.phase=phase
        return super().move(phase,goal,finger,seconds,frame_callback)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/table_insert_v1'))
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--lower-fixture',action='store_true')
    parser.add_argument('--offset-x-mm',type=float,default=0)
    parser.add_argument('--offset-y-mm',type=float,default=0)
    parser.add_argument('--timestep',type=float,default=.0005)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    env=OpenArmInsertEnv(images=False,timestep=args.timestep)
    m,d=env.model,env.data
    # Scene construction before reset only. Do not move a live fixture or plug.
    m.body_quat[m.body('socket').id]=[np.sqrt(.5),np.sqrt(.5),0,0]
    if args.lower_fixture:
        m.body_pos[m.body('socket').id]=[.4191,-.22,.44]
        m.geom_pos[m.geom('fixture').id]=[.4431,-.22,.38]
        m.geom_size[m.geom('fixture').id,2]=.06
    probe=TransportProbe(env)
    frames=[];renderer=None;transport_trace=[];result={'passed':False}
    cfg=json.loads(Path('configs/table_task_v3.json').read_text())
    try:
        callback=None
        if args.render:
            for c in cfg['cameras'].values():aim_camera(m,c['model_camera'],c['position_m'],c['target_m'],c['fovy_deg'])
            renderer=mujoco.Renderer(m,height=480,width=640)
            def callback(phase):
                renderer.update_scene(d,camera='scene_rgb')
                frame=Image.fromarray(renderer.render())
                frames.append(frame.resize((480,360)))
                frame.save(args.output/f'{phase}.png')
        pickup=probe.run(position=(.381+args.offset_x_mm/1000,-.22+args.offset_y_mm/1000,.330),frame_callback=callback)
        if not pickup['passed']:raise RuntimeError('Pickup failed')
        probe.carrying=True
        entry=d.site_xpos[m.site('socket_entry').id].copy()
        basis=d.site_xmat[m.site('socket_entry').id].reshape(3,3).copy()
        tool_rot=basis@probe.reference_rotation.T
        preplug=entry+basis@[-.025,0,0]
        target=preplug-tool_rot@probe.reference
        q=probe.ik(target,tool_rot,d.qpos[env.qa['right']])
        # Plan with the carried payload, then execute only through motor commands.
        path,planning=plan_carry(env,d.qpos[env.qa['right']].copy(),q,probe.reference,probe.reference_rotation)
        for waypoint in path:
            probe.execute_waypoint(waypoint,callback)
            probe.check_grasp()
        probe.check_grasp()
        probe.inserting=True
        probe.phase='insert'
        jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv))
        forward=-.025;hold=0.
        for step in range(500):
            info=env.metrics.diagnostics()
            valid=bool(info['valid_pose'])
            # Align at approach before advancing; subsequent correction remains closed-loop.
            if step>50:forward=min(.00003,forward+.005*env.dt)
            desired=entry+basis@np.array([forward,0,0])
            error=np.r_[np.clip(desired-d.xpos[env.plug],-.0005,.0005),
                        np.clip(rotation_vector(basis@d.xmat[env.plug].reshape(3,3).T),-.005,.005)]
            mujoco.mj_jacSite(m,d,jp,jr,env.grasp_site)
            jac=np.vstack([jp[:,env.va['right']],jr[:,env.va['right']]])
            change=jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*1e-5,error)
            env.target[:7]=np.clip(env.target[:7]+np.clip(.2*change,-.008,.008),env.lower,env.upper)
            for _ in range(env.substeps):probe.physics_step()
            mujoco.mj_forward(m,d)
            probe.check_grasp()
            info=env.metrics.diagnostics()
            hold=hold+env.dt if info['valid_pose'] else 0.
            transport_trace.append({'time_s':float(d.time),'depth_m':info['insertion_depth_m'],
                                    'valid_pose':bool(info['valid_pose']),'hold_s':hold})
            if callback and step%5==0:callback('insert')
            if hold>=.25:
                result={'passed':True,'hold_s':hold,'final_depth_m':info['insertion_depth_m']}
                break
        result.update(pickup=pickup,planning=planning,peak_socket_force_n=probe.peak_socket_force,
                      peak_socket_penetration_m=probe.peak_socket_penetration,
                      max_grasp_drift_m=probe.max_slip,max_grasp_angle_deg=probe.max_angle)
        if not result['passed']:result['error']='Insertion timeout'
    except RuntimeError as error:result['error']=str(error)
    finally:
        result.update(socket_roll_deg=90,lower_fixture=args.lower_fixture,timestep_s=args.timestep,
            physical_end_to_end_insertion=bool(result['passed']),
            command_hz=env.config['command_hz'],mujoco_version=mujoco.__version__,
            duration_including_settle_s=float(d.time),
            socket_position_m=m.body_pos[m.body('socket').id].tolist(),offset_x_mm=args.offset_x_mm,offset_y_mm=args.offset_y_mm,
            peak_unwanted_robot_force_n=probe.peak_robot_force,
            source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__),Path('controllers/table_pickup.py'),Path('controllers/carry_path.py'),
                 Path('scripts/preview_table_task.py'),Path('envs/openarm_insert.py'),Path('envs/scene.py')]})
        (args.output/'report.json').write_text(json.dumps(result,indent=2))
        (args.output/'trace.json').write_text(json.dumps({'pickup_and_transport':probe.trace,'insertion':transport_trace},indent=2))
        (args.output/'command_trace.json').write_text(json.dumps(probe.command_trace))
        if frames:frames[0].save(args.output/'rollout.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
        if renderer:renderer.close()
        env.close()
    print(result,flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
