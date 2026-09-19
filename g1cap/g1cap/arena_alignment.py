"""Bounded measured-state box alignment for placement; native arm dependencies."""
import math
import struct
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from .toolkit.box_alignment import AlignmentPath
from g1cap.arena_observation import assess_hold
from g1cap.arena_retreat import preparation_contact_fault


class HeldAlignment:
    def __init__(self,obs,action,wrist_motion):
        b=obs['box_quat']
        self.box_rotation=Rotation.from_quat([*b[1:],b[0]]).as_matrix()
        self.path=AlignmentPath(obs['box_pos'],self.box_rotation)
        self.wrists=wrist_motion(obs,[0.,0.,0.])
        self.initial={n:p.copy() for n,p in self.wrists.targets.items()}
        self.started=obs['time'];self.last_action=list(action)
        self.phase='place_align';self.result=None
        self.history=deque(maxlen=51);self.measurements=[]
        if not obs['bilateral'] or not obs['stance_clear']:raise ValueError('alignment_requires_held_standing')
        if min(s['clearance_m'] for s in obs['surfaces'].values())<.03:
            raise ValueError('alignment_requires_box_clear_of_tables')

    def command(self,obs):
        if self.result or self.phase=='place_align_hold':
            self.last_action[43:46]=[0.,0.,0.]
            return list(self.last_action)
        elapsed=max(0.,obs['time']-self.started+.02)
        fraction=min(1.,math.radians(5)*elapsed/max(self.path.angle,1e-9))
        D=self.path.delta(fraction);center=self.path.center
        for n,original in self.initial.items():
            target=original.copy();target.rotation=D@original.rotation
            target.translation=center+D@(original.translation-center)
            self.wrists.targets[n]=target
        action=list(self.last_action);action[43:46]=[0.,0.,0.];q=obs['root_quat']
        action[:43]=self.wrists.command(obs['joint_pos'],obs['root_pos'],[*q[1:],q[0]],action[:43],0.)
        self.last_action=list(struct.unpack('50f',struct.pack('50f',*action)))
        return self.last_action

    def update(self,obs):
        if self.result:return self.result
        elapsed=obs['time']-self.started;self.history.append(obs)
        b=obs['box_quat'];R=Rotation.from_quat([*b[1:],b[0]]).as_matrix()
        fraction=self.hold_fraction if self.phase=='place_align_hold' else min(1.,math.radians(5)*elapsed/max(self.path.angle,1e-9))
        desired=self.path.delta(fraction)@self.box_rotation
        error=Rotation.from_matrix(desired.T@R).magnitude()
        drift=float(np.linalg.norm(np.array(obs['box_pos'])-self.path.center))
        wrist_error=max(float(np.linalg.norm(np.array(obs['wrists_world'][n]['pos'])-p.translation))
                        for n,p in self.wrists.targets.items())
        self.measurements.append(dict(step=obs['step'],time=obs['time'],box_position_error_m=drift,
            box_rotation_error_rad=error,wrist_position_error_m=wrist_error,
            requested_angle_rad=fraction*self.path.angle,phase=self.phase))
        reason=preparation_contact_fault(obs)
        if any(s['robot_contact_peak_N']>5 for s in obs['surfaces'].values()):reason='robot_table_contact'
        if not obs['stance_clear'] or obs['tilt']>math.radians(15):reason='stance_lost'
        if min(s['clearance_m'] for s in obs['surfaces'].values())<.03:reason='box_table_clearance_lost'
        if drift>.03:reason='box_center_tracking_lost'
        if error>math.radians(10):reason='box_orientation_tracking_lost'
        if wrist_error>.02:reason='wrist_tracking_lost'
        if reason:self.result=('failed',reason)
        elif self.phase=='place_align' and max(abs(R[2]))>=math.cos(math.radians(3)):
            # Stop correcting loaded wrist error once the object is aligned.
            # Retain the achieved servo references; verification does not run IK.
            self.phase='place_align_hold';self.hold_started=obs['time'];self.hold_fraction=fraction
            self.history.clear()
        elif self.phase=='place_align' and elapsed>=12:self.result=('failed','alignment_timeout')
        elif self.phase=='place_align_hold' and obs['time']-self.hold_started>=3.-1e-9:
            flat=max(abs(R[2]))>=math.cos(math.radians(5))
            self.result=('completed','aligned_hold') if flat and assess_hold(list(self.history))['ready'] else ('failed','aligned_hold_not_verified')
        return self.result
