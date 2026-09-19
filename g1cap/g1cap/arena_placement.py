"""Placement on a selected, already-reached static table.

World metres and WXYZ poses. Retain fingers and arm references while lowering
pelvis at2 cm/s, at most14 cm and never below0.60 m. Withdraw wrists only after
measured support. This uses the source-return envelope without expanding it.
"""
from collections import deque
from copy import deepcopy
import math,struct
from .arena_observation import angular_distance


def selected_support(obs,surface_id):
    """Reject missing/malformed selected measurements, rather than using source data."""
    try:
        if surface_id not in ('source','destination'):raise ValueError('unsupported_surface')
        d=obs['surfaces'][surface_id]
        numbers=[d[k] for k in ('clearance_m','robot_contact_peak_N','box_upward_N')]
        numbers+=list(d['bounds']['min'])+list(d['bounds']['max'])
        if len(d['bounds']['min'])!=3 or len(d['bounds']['max'])!=3:raise ValueError('invalid_surface_observation')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in numbers):
            raise ValueError('invalid_surface_observation')
        if any(type(d[k]) is not bool for k in ('contained','supported')):raise ValueError('invalid_surface_observation')
        return d
    except (KeyError,TypeError):raise ValueError('missing_surface_observation') from None


def release_ready(rows,surface_id):
    if len(rows)<51 or rows[-1]['time']-rows[0]['time']<.999:return False
    if any(not r['surfaces'][surface_id]['supported'] or not r['surfaces'][surface_id]['contained']
           or max(r['hand_forces_N'].values())>.5 or not r['stance_clear'] for r in rows):return False
    pairs=list(zip(rows,rows[1:]))
    if any(b['time']<=a['time'] for a,b in pairs):return False
    speed=max(math.dist(a['box_pos'],b['box_pos'])/(b['time']-a['time']) for a,b in pairs)
    angular=max(angular_distance(a['box_quat'],b['box_quat'])/(b['time']-a['time']) for a,b in pairs)
    return speed<=.05 and angular<=.2


class PlaceOnSurface:
    def __init__(self,obs,action,surface_id,wrist_motion):
        self.surface_id=surface_id
        d=selected_support(obs,surface_id)
        if not d['contained']:raise ValueError('box_not_contained_over_selected_surface')
        available=min(.14,action[46]-.60)
        if not 0<d['clearance_m']<=available:raise ValueError('selected_surface_outside_lowering_envelope')
        self.last_action=list(action);self.last_action[43:46]=[0.,0.,0.]
        self.height=action[46];self.wrist_motion=wrist_motion;self.motion=None
        self.stage='lower';self.phase='place_lower';self.started=obs['time']
        self.history=deque(maxlen=51);self.support_count=0;self.result=None
        self.terminal_reason=None;self.method='place_box'

    def _phase(self,stage,obs):
        self.stage=stage;self.phase='place_'+stage;self.started=obs['time'];self.history.clear()

    def _finish(self,status,reason):
        self.last_action[43:46]=[0.,0.,0.];self.method=None;self.phase='idle'
        self.result=(status,reason);return self.result

    def command(self,obs):
        action=list(self.last_action);action[43:46]=[0.,0.,0.]
        elapsed=obs['time']-self.started+.02
        if not self.result and self.stage=='lower':action[46]=max(.60,self.height-min(.14,.02*elapsed))
        elif not self.result and self.stage=='withdraw':
            q=obs['root_quat']
            action[:43]=self.motion.command(obs['joint_pos'],obs['root_pos'],[*q[1:],q[0]],action[:43],elapsed)
        self.last_action=list(struct.unpack('50f',struct.pack('50f',*action)))
        return list(self.last_action)

    def update(self,obs):
        if self.result:return self.result
        try:d=selected_support(obs,self.surface_id)
        except ValueError as error:return self._finish('failed',str(error))
        elapsed=obs['time']-self.started
        if max(obs['robot_source_peak_N'],d['robot_contact_peak_N'])>5:
            return self._finish('failed','forbidden_support_contact')
        if not obs['stance_clear'] or obs['tilt']>math.radians(15):return self._finish('failed','placement_stance_lost')
        if not d['contained']:return self._finish('failed','box_left_selected_footprint')
        self.history.append(deepcopy(obs))
        if self.stage=='lower':
            if not obs['bilateral'] and not d['supported']:
                return self._finish('failed','retention_lost_before_support')
            self.support_count=self.support_count+1 if d['supported'] else 0
            if self.support_count>=15:self._phase('supported_hold',obs)
            elif elapsed>=7:return self._finish('failed','lowering_timeout')
        else:
            # As in the source-return controller: require actual support before
            # opening, allow bounded settling while opening, then independently
            # require sustained actual support and released hands at completion.
            if abs(d['clearance_m'])>.01:
                return self._finish('failed','box_left_selected_surface')
            if self.stage=='supported_hold' and not d['supported']:
                return self._finish('failed','selected_support_lost')
            if self.stage=='supported_hold' and elapsed>=2:
                self.motion=self.wrist_motion(obs,None);self._phase('withdraw',obs)
            elif self.stage=='withdraw' and elapsed>=4:self._phase('settle',obs)
            elif self.stage=='settle' and elapsed>=2:
                return self._finish('completed','supported_release') if release_ready(list(self.history),self.surface_id) else self._finish('failed','release_not_verified')
        return None


class PreparedPlacement:
    """Alignment → geometric clearance → supported release in one live world.

    Factories create bounded controllers from the current observation and the
    last sent actuator references. They never reset, step or score physics.
    """
    def __init__(self,obs,action,surface_id,alignment,retraction,wrist_motion):
        support=selected_support(obs,surface_id)
        if not support['contained']:raise ValueError('box_not_contained_over_selected_surface')
        if not 0<support['clearance_m']<=min(.14,action[46]-.60):
            raise ValueError('selected_surface_outside_lowering_envelope')
        self.surface_id=surface_id;self.retraction=retraction;self.wrist_motion=wrist_motion
        self.current=alignment(obs,action);self.stage='align';self.result=None
        self.last_action=list(action);self.phase=self.current.phase

    def command(self,obs):
        if not self.result:self.last_action=list(self.current.command(obs))
        return list(self.last_action)

    def update(self,obs):
        if self.result:return self.result
        result=self.current.update(obs);self.phase=self.current.phase
        if not result:return None
        if result[0]!='completed' or self.stage=='release':
            self.result=result;return self.result
        try:
            if self.stage=='align':
                self.current=self.retraction(obs,self.last_action,self.surface_id);self.stage='retract'
            else:
                self.current=PlaceOnSurface(obs,self.last_action,self.surface_id,self.wrist_motion);self.stage='release'
            self.phase=self.current.phase
        except ValueError as error:self.result=('failed',str(error))
        return self.result
