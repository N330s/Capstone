"""Pickup-only privileged recovery teacher; never changes live simulator state.

The snapshot planner uses a PRIVATE MjData. Live motion is exclusively through
env.step in the caller. Prefix commands are diagnostic context, not teacher labels.
"""
import numpy as np


def check_prefix_state(qpos, qvel, expected_qpos, expected_qvel, tolerance=1e-8):
    errors = {}
    for key, actual, expected in (("qpos", qpos, expected_qpos), ("qvel", qvel, expected_qvel)):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError("Invalid prefix state")
        errors[key] = float(np.max(np.abs(actual - expected)))
    if max(errors.values()) > tolerance:
        raise ValueError(f"Physical prefix did not reproduce failed trajectory: {errors}")
    return errors


def command_owner(command_index, anchor):
    if command_index < 0 or anchor < 1:
        raise ValueError("Nonnegative command and positive anchor required")
    return "prefix_replay" if command_index < anchor else "teacher_recovery"


def separation_direction(hand_to_object_normals):
    """Horizontal motion away from all remaining one-finger contact normals."""
    normals=np.asarray(hand_to_object_normals,dtype=float)
    if normals.ndim!=2 or normals.shape[1]!=3 or not len(normals) or not np.isfinite(normals).all():
        raise ValueError('Finite hand-to-object contact normals required')
    away=-normals.mean(axis=0); away[2]=0.
    length=float(np.linalg.norm(away))
    if length<.5: raise ValueError('No unambiguous horizontal separation direction')
    away/=length
    if np.any(normals@away>-.25): raise ValueError('Separation would approach another contact')
    return away


def recovery_teacher(env):
    """Construct only after importing the hash-verified scene snapshot."""
    from controllers.pick_insert_expert import (PickInsertExpert, JointPath,
        flat_plug_pose, plan_table_pick)

    class PickupOnlyExpert(PickInsertExpert):
        def _plan_pick(self):
            # Fresh actual object pose, not a timed continuation of the old demo.
            self.detection = self.detector.truth()
            pos, mat = flat_plug_pose(self.detection['plug_pos'], self.detection['plug_mat'],
                                     self.table_z, self.geom.rest_z_m)
            height_error = float(abs(pos[2] - self.detection['plug_pos'][2]))
            tilt_error = float(np.degrees(np.arccos(np.clip((np.trace(mat.T @ self.detection['plug_mat'])-1)/2, -1, 1))))
            if height_error > .003 or tilt_error > 10:
                self.plan_summaries.append({'result': 'plug_not_flat_after_settle',
                    'height_error_m': height_error, 'tilt_error_deg': tilt_error})
                return None, 'plug_not_flat_after_settle'
            plan, reason, log = plan_table_pick(self.planner, self.env.data.qpos,
                pos, mat, self._q_now(), require_insert=False, travel=self.geom.preshape_travel_m)
            self.plan_summaries.append({'result': reason, 'candidates': log,
                'actual_plug_pos_m': self.detection['plug_pos'].tolist(),
                'flat_pose_height_error_m': height_error, 'flat_pose_tilt_error_deg': tilt_error,
                'require_insert': False})
            self.plan = plan
            return plan, reason

        def _begin_place(self):
            # Explicitly prohibit entering the snapshot's insertion controller.
            self._goto('pickup_hold')

        def action(self):
            if self.phase == 'pickup_hold':
                return np.r_[self.env.target[:7], self.grip]
            return super().action()

    class RecoveryTeacher:
        def __init__(self):
            self.env = env
            self.expert = PickupOnlyExpert(env, detector_mode='privileged', max_regrasps=0)
            self.phase, self.failure, self.time = 'takeover_open', None, 0.
            self.hold = env.data.qpos[env.qa['right']].copy()
            self.open_travel = float(env.config['grip_open_travel_m'])
            self.path = None
            self.release_active=False
            self.release_plan=None
            self.transitions = [{'phase': self.phase, 'sim_time_s': float(env.data.time)}]

        def goto(self, phase):
            self.phase, self.time = phase, 0.
            self.transitions.append({'phase': phase, 'sim_time_s': float(env.data.time)})

        def action(self):
            self.time += env.dt
            if self.phase == 'takeover_open':
                if self.time >= 1.:
                    contacts=int(env.info['plug_finger_contacts'])
                    ready=(np.min(env.data.qpos[env.fqa['right']]) >= self.open_travel-.001
                           and not env.info['grasp_established']
                           and (contacts==0 or (contacts==1 and env.info['grasp_contact_force_n']<=.5)))
                    if not ready:
                        if self.time >= 2.:
                            self.failure = 'takeover_open_timeout'
                    else:
                        planner = self.expert.planner
                        planner.set_state(env.data.qpos)
                        q = env.data.qpos[env.qa['right']].copy()
                        p, rotation = env.grasp_pose()
                        path=[]; offset=np.zeros(3)
                        if contacts:
                            normals=[]
                            for c in env.data.contact:
                                b1,b2=[int(env.model.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
                                if b1 in env.fingers and b2==env.plug: normals.append(c.frame[:3].copy())
                                elif b2 in env.fingers and b1==env.plug: normals.append(-c.frame[:3].copy())
                            try: offset=.006*separation_direction(normals)
                            except ValueError:
                                self.failure='takeover_ambiguous_separation'
                                return np.r_[self.hold,self.open_travel]
                            path=planner.line(q,p,p+offset,rotation,step=.001)
                            self.release_active=True
                            self.release_plan={'offset_m':offset.tolist(),'initial_force_n':float(env.info['grasp_contact_force_n']),
                                'force_abort_n':.5,'maximum_still_touching_horizontal_motion_m':.003,
                                'initial_hand_pos_m':p.tolist()}
                        up=None if path is None else planner.line(path[-1] if path else q,
                            p+offset,p+offset+np.array([0.,0.,.08]),rotation,step=.01)
                        path=None if up is None else path+up
                        if path is None:
                            self.failure = 'takeover_retreat_ik'
                        elif not all(planner.segment_clear(a, b, self.open_travel, min_clearance=0.)
                                     for a, b in zip([q]+path[:-1], path)):
                            self.failure = 'takeover_retreat_blocked'
                        else:
                            self.path = JointPath(q, path, speed=.2, min_duration=2., hold=.5)
                            self.goto('takeover_retreat')
                return np.r_[self.hold, self.open_travel]
            if self.phase == 'takeover_retreat':
                if self.release_active:
                    if env.info['plug_finger_contacts']==0:
                        self.release_active=False
                    else:
                        p,_=env.grasp_pose()
                        moved=np.linalg.norm((p-np.asarray(self.release_plan['initial_hand_pos_m']))[:2])
                        if env.info['grasp_contact_force_n']>.5 or moved>.003 or env.info['grasp_established']:
                            self.failure='takeover_separation_not_safe'
                q = self.path.step(env.dt)
                if self.path.done:
                    if env.info['plug_finger_contacts']:
                        self.failure='takeover_retreat_contact_not_cleared'
                    elif np.max(np.abs(self.path.goal-env.data.qpos[env.qa['right']])) > .004:
                        self.failure = 'takeover_retreat_tracking'
                    else:
                        self.hold = self.path.goal.copy()
                        self.goto('takeover_settle')
                return np.r_[q, self.open_travel]
            if self.phase == 'takeover_settle':
                if self.time >= .5:
                    self.expert.start_plug_z = float(env.data.xpos[env.plug][2])
                    self.expert._begin_pick()
                    self.goto('teacher_pick')
                    self.failure = self.expert.failure
                return np.r_[self.hold, self.open_travel]
            command = self.expert.action()
            self.failure = self.expert.failure
            return command

        def diagnostics(self):
            return {'phase': self.expert.phase if self.phase == 'teacher_pick' else self.phase,
                'failure': self.failure, 'transitions': self.transitions,
                'release_plan':self.release_plan,
                'expert': self.expert.diagnostics(), 'teacher_privileged_pose': True,
                'live_state_writes': False, 'insertion_enabled': False}

    return RecoveryTeacher()
