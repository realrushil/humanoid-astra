"""Lift from an already stable, source-supported bilateral grasp.

Use the existing6 cm world-up wrist envelope and retained servo references.
This is a separately qualified support-unloading operation, not a passing score
for the earlier failed neural pickup. No new finger closing or grasp squeeze.
"""
from collections import deque
from copy import deepcopy
import math,struct
from .arena_observation import assess_hold,angular_distance
from .arena_retreat import preparation_contact_fault


def supported_grasp_ready(rows):
    if len(rows)<51 or rows[-1]['time']-rows[0]['time']<.999:return False
    if any(not r['supported'] or not r['bilateral'] or not r['stance_clear'] or abs(r['clearance'])>.01
           or preparation_contact_fault(r) or r['robot_source_peak_N']>5 for r in rows):return False
    pairs=list(zip(rows,rows[1:]))
    if any(b['time']<=a['time'] for a,b in pairs):return False
    return (max(math.dist(a['box_pos'],b['box_pos'])/(b['time']-a['time']) for a,b in pairs)<=.05
        and max(angular_distance(a['box_quat'],b['box_quat'])/(b['time']-a['time']) for a,b in pairs)<=.2
        and max(math.dist(a['root_pos'][:2],b['root_pos'][:2])/(b['time']-a['time']) for a,b in pairs)<=.05)


class SupportedLift:
    def __init__(self,history,action,wrist_motion):
        if not supported_grasp_ready(history):raise ValueError('stable_supported_grasp_required')
        obs=history[-1]
        self.last_action=list(action);self.last_action[43:46]=[0.,0.,0.]
        self.motion=wrist_motion(obs,[0.,0.,.06]);self.started=obs['time']
        self.phase='supported_lift';self.method='lift_supported_box';self.result=None
        self.terminal_reason=None;self.history=deque(maxlen=51)

    def command(self,obs):
        action=list(self.last_action);action[43:46]=[0.,0.,0.]
        if self.phase=='supported_lift':
            q=obs['root_quat'];elapsed=obs['time']-self.started+.02
            action[:43]=self.motion.command(obs['joint_pos'],obs['root_pos'],[*q[1:],q[0]],action[:43],elapsed)
        self.last_action=list(struct.unpack('50f',struct.pack('50f',*action)))
        return list(self.last_action)

    def _finish(self,status,reason):
        self.result=(status,reason);self.phase='idle';self.method=None
        self.last_action[43:46]=[0.,0.,0.];return self.result

    def update(self,obs):
        if self.result:return self.result
        fault=preparation_contact_fault(obs)
        if not obs['bilateral'] or not obs['stance_clear'] or obs['clearance']<-.01:fault=fault or 'lift_contact_or_stance_lost'
        if obs['robot_source_peak_N']>5:fault=fault or 'forbidden_source_contact'
        if obs['tilt']>math.radians(15):fault=fault or 'body_tilt_limit'
        if fault:return self._finish('failed',fault)
        self.history.append(deepcopy(obs));elapsed=obs['time']-self.started
        if self.phase=='supported_lift':
            if elapsed>=7:
                self.phase='verify_supported_lift';self.started=obs['time'];self.history.clear()
        else:
            if assess_hold(list(self.history))['ready']:return self._finish('completed','supported_grasp_lifted')
            if elapsed>=5:return self._finish('failed','supported_lift_hold_timeout')
        return None
