"""Limited loaded retreat witnessed in trial28; world metres, WXYZ poses.

HOMIE receives body-frame velocities. Upper targets stay fixed; position
feedback uses simulator localization. No obstacle planner or grasp recovery.
"""
from collections import deque
from copy import deepcopy
import math

from .arena_observation import assess_hold, angular_distance
from .toolkit.arena_approach import rotate_xyzw


def relative_box(obs):
    w,x,y,z=obs['root_quat']
    return rotate_xyzw([-x,-y,-z,w],[b-r for b,r in zip(obs['box_pos'],obs['root_pos'])])


def contact_fault(obs):
    summary=obs.get('loaded_contacts')
    if not isinstance(summary,dict) or summary.get('samples')!=4:return 'missing_loaded_contacts'
    values=[summary.get('minimum_hand_N'),summary.get('minimum_total_foot_N')]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
        return 'invalid_loaded_contacts'
    if values[0]<=.5:return 'substep_hand_contact_lost'
    if values[1]<=5:return 'substep_ground_support_lost'
    return None


def preparation_contact_fault(obs):
    fault=contact_fault(obs)
    if fault:return fault
    peak=obs['loaded_contacts'].get('maximum_hand_N')
    if isinstance(peak,bool) or not isinstance(peak,(int,float)) or not math.isfinite(peak) or peak<0:
        return 'invalid_preparation_force'
    # Sum of contact norms for one hand, not calibrated grip-normal force.
    # 25 N is a development bound witnessed with the 0.1 kg cube.
    if peak>25:return 'preparation_force_limit'
    return None


class LoadedRetreat:
    def __init__(self,obs,action,distance):
        self.anchor=deepcopy(obs)
        self.reference=list(action)
        self.distance=distance
        self.phase='loaded_retreat'
        self.started=obs['time']
        self.history=deque(maxlen=51)
        self.speed=self.lateral_speed=0.

    def command(self,obs):
        action=list(self.reference);action[43:46]=[0.,0.,0.]
        if self.phase=='loaded_retreat':
            remaining=self.anchor['root_pos'][0]-self.distance-obs['root_pos'][0]
            wanted=-min(.12,max(.06,abs(remaining)))
            self.speed=max(wanted,self.speed-.006)
            # Gain1/s; lateral cap4 cm/s, acceleration0.3 m/s² at50 Hz.
            vy=max(-.04,min(.04,self.anchor['root_pos'][1]-obs['root_pos'][1]))
            self.lateral_speed+=max(-.006,min(.006,vy-self.lateral_speed))
            vx,vy=self.speed,self.lateral_speed
            scale=min(1.,.12/max(math.hypot(vx,vy),1e-12));vx*=scale;vy*=scale
            w,x,y,z=obs['root_quat'];yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            action[43:46]=[math.cos(yaw)*vx+math.sin(yaw)*vy,-math.sin(yaw)*vx+math.cos(yaw)*vy,0.]
        return action

    def measurements(self,obs):
        return dict(root_retreat_m=self.anchor['root_pos'][0]-obs['root_pos'][0],
                    box_retreat_m=self.anchor['box_pos'][0]-obs['box_pos'][0],
                    lateral_error_m=abs(obs['root_pos'][1]-self.anchor['root_pos'][1]),
                    box_pelvis_change_m=math.dist(relative_box(self.anchor),relative_box(obs)))

    def update(self,obs):
        self.history.append(obs)
        fault=contact_fault(obs)
        if not fault and (not obs['bilateral'] or obs['clearance']<.03):fault='grasp_or_height_lost'
        # Torso-relative motion includes compliant arm motion; it is diagnostic,
        # not grasp loss. Retention is checked by contact, clearance and final hold.
        if not fault and abs(obs['root_pos'][1]-self.anchor['root_pos'][1])>.05:fault='lateral_drift'
        if not fault and angular_distance(self.anchor['root_quat'],obs['root_quat'])>.20:fault='body_orientation_drift'
        if fault:return 'failed',fault
        distance=self.measurements(obs)['root_retreat_m']
        if self.phase=='loaded_retreat':
            if distance>self.distance+.03:return 'failed','retreat_overshoot'
            # Aim at the requested center; reserve tolerance for stopping error.
            if distance>=self.distance:
                self.phase='settle_loaded';self.started=obs['time'];self.history.clear()
            elif obs['time']-self.started>=6:return 'failed','retreat_timeout'
        else:
            if assess_hold(list(self.history))['ready'] and all(s['stance_clear'] for s in self.history):
                passed=self.distance-.02<=distance<=self.distance+.03 and self.measurements(obs)['box_retreat_m']>=self.distance-.05
                return ('completed','loaded_retreat_and_hold') if passed else ('failed','final_displacement_outside_goal')
            if obs['time']-self.started>=5:return 'failed','loaded_hold_timeout'
        return None
