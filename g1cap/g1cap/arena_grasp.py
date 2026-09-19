"""Measured-contact stop for the existing bounded inward wrist preparation.

At most4 mm per hand at2 mm/s. Stop changing arm references when either hand's
substep contact-norm sum reaches20 N; retain a25 N hard ceiling. These are
uncalibrated development bounds, not estimates of normal gripping force.
"""
from collections import deque
from copy import deepcopy
import math,struct
from .arena_observation import assess_hold
from .arena_retreat import preparation_contact_fault

class GripPreparation:
    def __init__(self,obs,action,wrist_motion):
        fault=preparation_contact_fault(obs)
        if fault:raise ValueError(fault)
        self.last_action=list(action);self.last_action[43:46]=[0.,0.,0.]
        self.motion=wrist_motion(obs,[0.,0.,0.],inward_m=.004)
        self.started=obs['time'];self.stopped_at=obs['time'] if obs['loaded_contacts']['maximum_hand_N']>=20 else None
        self.stop_reason='contact_margin' if self.stopped_at is not None else None
        self.phase='prepare_carry';self.method='prepare_grasp';self.result=None;self.terminal_reason=None
        self.history=deque(maxlen=51)

    def command(self,obs):
        action=list(self.last_action);action[43:46]=[0.,0.,0.]
        if self.stopped_at is None and self.result is None:
            q=obs['root_quat'];elapsed=obs['time']-self.started+.02
            action[:43]=self.motion.command(obs['joint_pos'],obs['root_pos'],[*q[1:],q[0]],action[:43],elapsed)
        self.last_action=list(struct.unpack('50f',struct.pack('50f',*action)))
        return list(self.last_action)

    def _finish(self,status,reason):
        self.result=(status,reason);self.phase='idle';self.method=None
        return self.result

    def update(self,obs):
        if self.result:return self.result
        fault=preparation_contact_fault(obs)
        if not obs['bilateral'] or not obs['stance_clear'] or obs['clearance']<.03:fault=fault or 'preparation_contact_stance_or_height'
        if obs['robot_source_peak_N']>5 or obs['tilt']>math.radians(15):fault=fault or 'preparation_physical_fault'
        if fault:return self._finish('failed',fault)
        elapsed=obs['time']-self.started
        if self.stopped_at is None:
            if obs['loaded_contacts']['maximum_hand_N']>=20:self.stop_reason='contact_margin'
            elif elapsed>=2:self.stop_reason='displacement_limit'
            if self.stop_reason:self.stopped_at=obs['time']
        self.history.append(deepcopy(obs))
        if elapsed>=3 and self.stopped_at is not None and obs['time']-self.stopped_at>=.999 and assess_hold(list(self.history))['ready']:
            return self._finish('completed','grasp_preparation_verified')
        if elapsed>=5:return self._finish('failed','preparation_did_not_settle')
        return None
