"""Bounded common wrist retraction to clear a selected table before lowering."""
from collections import deque
from copy import deepcopy
import math
import struct
from g1cap.arena_observation import assess_hold,angular_distance
from g1cap.arena_retreat import preparation_contact_fault
from .arena_placement_geometry import placement_retraction


class PlacementRetraction:
    def __init__(self,obs,action,surface_id,wrist_motion,geometry,supports,*,arm_clearance=None):
        self.anchor=deepcopy(obs);self.geometry=geometry;self.supports=supports;self.surface_id=surface_id
        self.arm_clearance=arm_clearance
        # Require 5 mm; prefer an 8 mm reserve when geometry permits it.
        # Otherwise maximize feasible clearance and verify the actual held pose.
        # The desired reserve is a development assumption, not calibration.
        self.plan=placement_retraction(self.anchor,self.geometry,self.supports,self.surface_id,
            preferred_arm_margin_m=.008,arm_clearance=self.arm_clearance)
        if self.plan is None:raise ValueError('no_bounded_retraction_with_placement_margins')
        if not self.anchor['bilateral'] or not self.anchor['stance_clear']:raise ValueError('held_standing_required')
        self.motion=wrist_motion(self.anchor,self.plan['translation_world'])
        self.last_action=list(action);self.last_action[43:46]=[0.,0.,0.]
        self.started=self.anchor['time'];self.phase='place_retract';self.method='placement_retraction'
        self.result=None;self.history=deque(maxlen=51);self.measurements=[]
        if self.plan['distance_m']==0.:
            self.phase='place_retract_hold';self.hold_started=self.started;self.hold_fraction=0.

    def command(self,obs):
        if self.result or self.phase=='place_retract_hold':return list(self.last_action)
        action=list(self.last_action);action[43:46]=[0.,0.,0.];q=obs['root_quat']
        action[:43]=self.motion.command(obs['joint_pos'],obs['root_pos'],[*q[1:],q[0]],action[:43],obs['time']-self.started+.02)
        self.last_action=list(struct.unpack('50f',struct.pack('50f',*action)))
        return list(self.last_action)

    def update(self,obs):
        if self.result:return self.result
        elapsed=obs['time']-self.started;distance=self.plan['distance_m'];delta=self.plan['translation_world']
        moved=[b-a for a,b in zip(self.anchor['box_pos'],obs['box_pos'])]
        along=sum(a*b for a,b in zip(moved,delta))/max(distance,1e-9)
        transverse=math.sqrt(sum((moved[i]-along*delta[i]/max(distance,1e-9))**2 for i in range(3)))
        fraction=self.hold_fraction if self.phase=='place_retract_hold' else min(1.,.01*elapsed/max(distance,1e-9))
        wrist_error=max(math.dist(obs['wrists_world'][n]['pos'],[p.translation[i]+delta[i]*fraction for i in range(3)]) for n,p in self.motion.targets.items())
        reason=preparation_contact_fault(obs)
        if any(s['robot_contact_peak_N']>5 for s in obs['surfaces'].values()):reason=reason or 'robot_table_contact'
        if not obs['stance_clear'] or obs['tilt']>math.radians(15):reason=reason or 'stance_lost'
        if min(s['clearance_m'] for s in obs['surfaces'].values())<.03:reason=reason or 'box_table_clearance_lost'
        if not obs['surfaces'][self.surface_id]['contained']:reason=reason or 'box_left_selected_footprint'
        if transverse>.01:reason=reason or 'box_cross_axis_or_height_error'
        if angular_distance(self.anchor['box_quat'],obs['box_quat'])>math.radians(5):reason=reason or 'box_rotation_drift'
        if wrist_error>.02:reason=reason or 'wrist_tracking_lost'
        if along>distance+.005:reason=reason or 'retraction_overshoot'
        remaining=None
        if not reason and (along>=distance-.005 or self.phase=='place_retract_hold'):
            candidate=placement_retraction(obs,self.geometry,self.supports,self.surface_id,arm_clearance=self.arm_clearance)
            remaining=candidate['distance_m'] if candidate is not None else None
        self.measurements.append(dict(step=obs['step'],time=obs['time'],phase=self.phase,box_displacement_m=along,
            cross_axis_or_height_error_m=transverse,wrist_error_m=wrist_error,remaining_geometry_request_m=remaining))
        self.history.append(deepcopy(obs))
        if reason:self.result=('failed',reason)
        elif self.phase=='place_retract' and along>=distance-.002 and remaining==0.:
            self.phase='place_retract_hold';self.hold_started=obs['time'];self.hold_fraction=fraction;self.history.clear()
        # Up to 8 cm at 1 cm/s, plus 2 s for measured tracking to settle.
        elif self.phase=='place_retract' and elapsed>=10:self.result=('failed','retraction_timeout')
        elif self.phase=='place_retract_hold' and obs['time']-self.hold_started>=2.-1e-9:
            ready=remaining==0. and assess_hold(list(self.history))['ready']
            self.result=('completed','placement_clearance_and_hold') if ready else ('failed','retracted_hold_not_verified')
        return self.result
