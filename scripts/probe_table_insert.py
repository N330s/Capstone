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
from connector import catalog
from envs import workspace as wsp


class TransportProbe(PickupProbe):
    def __init__(self,env):
        super().__init__(env)
        self.carrying=False
        self.inserting=False
        self.peak_socket_force=0.
        self.peak_socket_penetration=0.
        self.peak_leaf_force=0.
        self.peak_leaf_penetration=0.
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
            if not bodies & e.socket_bodies:
                if force>.1:raise RuntimeError('Carried plug contacted table/fixture/robot')
            elif not self.inserting and force>.1:
                raise RuntimeError('Socket collision during transport')
            elif not {int(g) for g in (c.geom1,c.geom2)} & e.metrics.leaf_geoms:
                total+=force
                self.peak_socket_penetration=max(self.peak_socket_penetration,float(-c.dist))
        self.peak_socket_force=max(self.peak_socket_force,total)
        # Spring-leaf retention is intentional load; only rigid-wall force/penetration abort here.
        info=e.metrics.diagnostics()
        wall=info['contact_force_n']
        self.peak_leaf_force=max(self.peak_leaf_force,info['leaf_contact_force_n'])
        self.peak_leaf_penetration=max(self.peak_leaf_penetration,info['leaf_penetration_m'])
        detail=(f"wall {wall:.2f} N, wall penetration {info['max_penetration_m']*1000:.3f} mm, leaf {info['leaf_contact_force_n']:.1f} N, "
                f"leaf penetration {info['leaf_penetration_m']*1000:.3f} mm, depth {info['insertion_depth_m']*1000:.2f} mm, "
                f"offset y/z {info['offset_y_m']*1e6:.0f}/{info['offset_z_m']*1e6:.0f} um, axial {info['socket_force_x_n']:.1f} N, t={d.time:.3f}s")
        wall_limit=e.metric_config['contact_abort_n'] if e.leaf_geoms else 5.
        if wall>wall_limit or info['max_penetration_m']>.0002:
            w=np.zeros(6)
            for i,c in enumerate(d.contact):
                b={int(m.geom_bodyid[g]) for g in (c.geom1,c.geom2)}
                if e.plug in b and b&e.socket_bodies:
                    mujoco.mj_contactForce(m,d,i,w)
                    detail+=(f"; {m.geom(c.geom1).name}-{m.geom(c.geom2).name} pos {np.round(c.pos,4).tolist()} "
                             f"n {np.round(c.frame[:3],3).tolist()} f {np.round(w[:3],2).tolist()} dist {c.dist*1000:.3f} mm")
        if wall>wall_limit or info['max_penetration_m']>.0002:raise RuntimeError('Socket force/penetration abort: '+detail)
        if (info['leaf_contact_force_n']>e.metric_config.get('leaf_force_abort_n',np.inf)
                or info['leaf_penetration_m']>e.metric_config.get('leaf_penetration_limit_m',np.inf)):
            raise RuntimeError('Leaf force/penetration abort: '+detail)

    def check_grasp(self):
        e=self.env
        slip=float(np.linalg.norm(self.relative_plug()-self.reference))
        relative=e.data.site_xmat[e.grasp_site].reshape(3,3).T@e.data.xmat[e.plug].reshape(3,3)
        angle=float(np.degrees(np.linalg.norm(rotation_vector(relative@self.reference_rotation.T))))
        self.max_slip=max(self.max_slip,slip);self.max_angle=max(self.max_angle,angle)
        contacts=len(self.finger_contacts())
        if slip>.001 or angle>5 or contacts!=2:
            raise RuntimeError(f'Grasp lost or slipped in transport/insertion: slip {slip*1000:.3f} mm, angle {angle:.2f} deg, '
                               f'finger contacts {contacts}, phase {self.phase}, t={self.env.data.time:.2f}s')

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
    parser.add_argument('--socket-roll-deg',type=float,default=0.,
        help='extra roll about the socket X axis applied at model construction (0 = workspace orientation)')
    parser.add_argument('--offset-x-mm',type=float,default=0)
    parser.add_argument('--offset-y-mm',type=float,default=0)
    parser.add_argument('--timestep',type=float,default=.0005)
    parser.add_argument('--workspace',type=Path,default=None,help='workspace spec (default configs/workspace_v1.json)')
    parser.add_argument('--plug-type',choices=catalog.plug_names(),default=None,
        help='override the workspace connector.plug (needs a workspace with a connector section)')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    workspace=wsp.load_workspace(args.workspace)
    if args.plug_type:wsp.apply_connector(workspace,plug=args.plug_type)
    env=OpenArmInsertEnv(images=False,timestep=args.timestep,workspace=workspace)
    derived=env.derived
    m,d=env.model,env.data
    # Scene construction before reset only. Do not move a live fixture or plug.
    if args.socket_roll_deg:
        half=np.radians(args.socket_roll_deg)/2
        roll=np.array([np.cos(half),np.sin(half),0,0]);out=np.zeros(4)
        mujoco.mju_mulQuat(out,roll,m.body_quat[m.body('socket').id]);m.body_quat[m.body('socket').id]=out
    if args.lower_fixture:
        m.body_pos[m.body('socket').id]=[.4191,-.22,.44]
        m.geom_pos[m.geom('fixture').id]=[.4431,-.22,.38]
        m.geom_size[m.geom('fixture').id,2]=.06
    probe=TransportProbe(env)
    e_socket_bodies=env.socket_bodies
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
        pickup=probe.run(position=env.spawn_position+[args.offset_x_mm/1000,args.offset_y_mm/1000,0],frame_callback=callback)
        if not pickup['passed']:raise RuntimeError('Pickup failed')
        probe.carrying=True
        entry=d.site_xpos[m.site('socket_entry').id].copy()
        basis=d.site_xmat[m.site('socket_entry').id].reshape(3,3).copy()
        tool_rot=basis@probe.reference_rotation.T
        preplug=entry+basis@[derived['probe_preplug_x_m'],0,0]
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
        forward=derived['probe_preplug_x_m'];hold=0.
        # Accumulated lateral (socket y/z) correction. Without a bound the loop integrates while
        # friction holds the plug and releases the wind-up as a slam into a slot wall.
        # The bound only counts from first socket contact: free-space alignment of the ~1 mm
        # carry error is real motion, and counting it saturated the bound before contact and
        # left the plug ~150 um low (the workspace_v2 1.5 m cable abort).
        lateral_bias=np.zeros(2);lateral_cap=.001;engaged=False
        for step in range(500):
            info=env.metrics.diagnostics()
            valid=bool(info['valid_pose'])
            # Align at approach before advancing; subsequent correction remains closed-loop.
            # Force-limited push: keep advancing at 5 mm/s only while the axial socket load is
            # below the cap, so the compliant arm cannot wind up far beyond the retention force.
            push_cap=env.workspace['socket_leaves'].get('push_force_cap_n',np.inf) if env.leaf_geoms else np.inf
            if step>50 and abs(info['socket_force_x_n'])<push_cap:forward=min(.00003,forward+.005*env.dt)
            desired=entry+basis@np.array([forward,0,0])
            # Lateral (socket y/z) corrections run 2x faster than the axial advance: under the
            # retention load the compliant arm droops faster than a uniform 0.1 mm/step loop
            # can re-centre the elements inside the sub-millimetre opening clearance.
            local=np.clip(basis.T@(desired-d.xpos[env.plug]),-.0005,.0005)*np.array([1.,2.,2.])
            if not engaged and max(abs(info['socket_force_x_n']),info['contact_force_n'],info['leaf_contact_force_n'])>.1:
                engaged=True;lateral_bias[:]=0.
            if engaged:
                for axis in (1,2):
                    proposed=lateral_bias[axis-1]+.2*local[axis]
                    if abs(proposed)>lateral_cap and np.sign(proposed)==np.sign(local[axis]):local[axis]=0.
                lateral_bias+=.2*local[1:]
            error=np.r_[basis@local,
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
                                    'valid_pose':bool(info['valid_pose']),'hold_s':hold,
                                    **{k:info[k] for k in ('socket_force_x_n','socket_lateral_force_n','socket_torque_nm',
                                        'leaf_normal_n','leaf_friction_n','contact_force_n','grasp_contact_force_n',
                                        'finger_force_x_n','cable_tension_n','cable_force_n')},
                                    'grasp_slip_m':float(np.linalg.norm(probe.relative_plug()-probe.reference)),
                                    'grasp_rotvec_deg':np.degrees(rotation_vector(
                                        (d.site_xmat[env.grasp_site].reshape(3,3).T@d.xmat[env.plug].reshape(3,3))@probe.reference_rotation.T)).tolist(),
                                    'plug_offset_yz_m':[info['offset_y_m'],info['offset_z_m']],
                                    'lateral_bias_m':lateral_bias.tolist(),'engaged':engaged,
                                    'orientation_error_deg':info['orientation_error_deg'],
                                    'finger_travel_m':d.qpos[env.fqa['right']].tolist()})
            if callback and step%5==0:callback('insert')
            # One sample of the plug/socket contacts once the retention load exceeds 8 N (diagnostic).
            if info['socket_force_x_n']<-8 and not result.get('retention_contact_sample'):
                w=np.zeros(6);dump=[]
                for i,c in enumerate(d.contact):
                    b={int(m.geom_bodyid[g]) for g in (c.geom1,c.geom2)}
                    if env.plug in b and b&e_socket_bodies:
                        mujoco.mj_contactForce(m,d,i,w)
                        dump.append({'geoms':[m.geom(c.geom1).name,m.geom(c.geom2).name],'pos':c.pos.tolist(),
                                     'normal_world':c.frame[:3].tolist(),'force_contact_frame':w[:3].tolist(),'dist_mm':c.dist*1000})
                result['retention_contact_sample']=dump
            if hold>=.25:
                result={'passed':True,'hold_s':hold,'final_depth_m':info['insertion_depth_m']}
                break
        result.update(pickup=pickup,planning=planning,peak_socket_force_n=probe.peak_socket_force,
                      peak_socket_penetration_m=probe.peak_socket_penetration,
                      peak_leaf_force_n=probe.peak_leaf_force,peak_leaf_penetration_m=probe.peak_leaf_penetration,
                      final_force_breakdown={k:v for k,v in env.metrics.diagnostics().items()
                                             if k.endswith(('_n','_nm'))},
                      max_grasp_drift_m=probe.max_slip,max_grasp_angle_deg=probe.max_angle)
        if not result['passed']:result['error']='Insertion timeout'
    except RuntimeError as error:result['error']=str(error)
    finally:
        result.update(socket_roll_deg=args.socket_roll_deg,lower_fixture=args.lower_fixture,timestep_s=args.timestep,
            plug_type=args.plug_type,connector=env.manifest()['connector'],
            physical_end_to_end_insertion=bool(result['passed']),
            command_hz=env.config['command_hz'],mujoco_version=mujoco.__version__,
            duration_including_settle_s=float(d.time),
            socket_position_m=m.body_pos[m.body('socket').id].tolist(),offset_x_mm=args.offset_x_mm,offset_y_mm=args.offset_y_mm,
            peak_unwanted_robot_force_n=probe.peak_robot_force,peak_cable_robot_force_n=probe.peak_cable_robot_force,
            workspace=env.manifest()['workspace'],
            source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__),Path('controllers/table_pickup.py'),Path('controllers/carry_path.py'),
                 Path('scripts/preview_table_task.py'),Path('envs/openarm_insert.py'),Path('envs/scene.py'),
                 Path('envs/workspace.py'),Path(env.workspace['_path']),Path('configs/openarm_v1.json'),
                 Path('connector/spec.py'),Path('connector/catalog.py'),Path('connector/slab.py'),Path('connector/builder.py')]})
        (args.output/'report.json').write_text(json.dumps(result,indent=2))
        (args.output/'trace.json').write_text(json.dumps({'pickup_and_transport':probe.trace,'insertion':transport_trace},indent=2))
        (args.output/'command_trace.json').write_text(json.dumps(probe.command_trace))
        if frames:frames[0].save(args.output/'rollout.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
        if renderer:renderer.close()
        env.close()
    print(result,flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
