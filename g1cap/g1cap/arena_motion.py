"""Loaded translation and turning with retained arm/body references.

World positions use metres, WXYZ orientation; HOMIE commands use body m/s.
Keep loaded joint/height/torso references; no new grasp preparation. These are
bounded development controllers; there is no scene route or global planner.
"""
from collections import deque
from copy import deepcopy
import math
from .arena_observation import assess_hold,angular_distance
from .arena_retreat import contact_fault


def direction(obs):
    w,x,y,z=obs['root_quat']
    angle=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    return math.cos(angle),math.sin(angle)


def measurements(anchor,obs):
    c,s=direction(anchor)
    delta=[obs['root_pos'][i]-anchor['root_pos'][i] for i in range(2)]
    box=[obs['box_pos'][i]-anchor['box_pos'][i] for i in range(2)]
    return dict(root_forward_m=c*delta[0]+s*delta[1],box_forward_m=c*box[0]+s*box[1],
                lateral_error_m=-s*delta[0]+c*delta[1])


def fault(anchor,obs,distance):
    reason=contact_fault(obs);m=measurements(anchor,obs)
    if not obs['bilateral'] or obs['clearance']<.03:reason=reason or 'grasp_or_height_lost'
    if abs(m['lateral_error_m'])>.05:reason=reason or 'lateral_drift'
    if angular_distance(anchor['root_quat'],obs['root_quat'])>.2:reason=reason or 'body_orientation_drift'
    if m['root_forward_m']>distance+.03:reason=reason or 'translation_overshoot'
    if m['root_forward_m']<-.03:reason=reason or 'wrong_translation_direction'
    return reason


def at_goal(anchor,obs,distance,history):
    m=measurements(anchor,obs)
    return (distance-.02<=m['root_forward_m']<=distance+.03 and m['box_forward_m']>=distance-.05
            and assess_hold(list(history))['ready'] and all(s['stance_clear'] for s in history))


class LoadedTranslation:
    def __init__(self,obs,action,distance):
        if isinstance(distance,bool) or not math.isfinite(distance) or not .2<=distance<=.5:raise ValueError('forward travel must be20–50 cm')
        self.anchor=deepcopy(obs);self.reference=list(action);self.distance=distance
        self.phase='loaded_forward';self.started=obs['time'];self.speed=0.;self.side_speed=0.
        self.history=deque(maxlen=51)

    def measurements(self,obs):return measurements(self.anchor,obs)

    def command(self,obs):
        action=list(self.reference);action[43:46]=[0.,0.,0.]
        if self.phase=='loaded_forward':
            m=self.measurements(obs)
            # Trial50 lost translation response as a distance-proportional request
            # fell toward .06 m/s. Keep the witnessed .12 m/s walking request
            # until the measured stop point; keep the same cap and stop checks.
            wanted=.12
            self.speed+=max(-.006,min(.006,wanted-self.speed))
            side=max(-.04,min(.04,-m['lateral_error_m']))
            self.side_speed+=max(-.006,min(.006,side-self.side_speed))
            c,s=direction(self.anchor)
            vx=c*self.speed-s*self.side_speed;vy=s*self.speed+c*self.side_speed
            scale=min(1.,.12/max(math.hypot(vx,vy),1e-12));vx*=scale;vy*=scale
            c,s=direction(obs)
            action[43:45]=[c*vx+s*vy,-s*vx+c*vy]
        return action

    def update(self,obs):
        self.history.append(deepcopy(obs))
        reason=fault(self.anchor,obs,self.distance)
        if reason:return 'failed',reason
        if self.phase=='loaded_forward':
            if self.measurements(obs)['root_forward_m']>=self.distance:
                self.phase='settle_forward';self.started=obs['time'];self.history.clear()
            elif obs['time']-self.started>=6:return 'failed','translation_timeout'
        else:
            if assess_hold(list(self.history))['ready'] and all(s['stance_clear'] for s in self.history):
                return ('completed','translation_and_hold') if at_goal(self.anchor,obs,self.distance,self.history) else ('failed','final_displacement_outside_goal')
            if obs['time']-self.started>=5:return 'failed','translation_hold_timeout'
        return None


def wrap(angle):return math.atan2(math.sin(angle),math.cos(angle))

def yaw(q):
    w,x,y,z=q
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

def turn_measurements(anchor,obs,delta):
    change=wrap(yaw(obs['root_quat'])-yaw(anchor['root_quat']))
    return dict(yaw_change_rad=change,progress_rad=math.copysign(1,delta)*change,
                heading_error_rad=wrap(delta-change),
                pelvis_translation_m=math.dist(anchor['root_pos'][:2],obs['root_pos'][:2]))

def turn_fault(anchor,obs,delta):
    fault=contact_fault(obs);m=turn_measurements(anchor,obs,delta)
    if not obs['bilateral'] or obs['clearance']<.03:fault=fault or 'grasp_or_height_lost'
    if obs['robot_source_peak_N']>5:fault=fault or 'forbidden_source_contact'
    if obs['tilt']>math.radians(15):fault=fault or 'body_tilt_limit'
    if m['pelvis_translation_m']>.15:fault=fault or 'turn_translation_limit'
    if m['progress_rad']>abs(delta)+math.radians(5):fault=fault or 'turn_overshoot'
    if m['progress_rad']<-math.radians(5):fault=fault or 'wrong_turn_direction'
    return fault

def settled_turn(history):
    rows=list(history);hold=assess_hold(rows)
    rates=[abs(wrap(yaw(b['root_quat'])-yaw(a['root_quat'])))/(b['time']-a['time'])
           for a,b in zip(rows,rows[1:]) if b['time']>a['time']]
    rate=max(rates,default=float('inf'))
    return dict(ready=hold['ready'] and rate<=.1 and all(r['stance_clear'] for r in rows),
                max_root_yaw_rate_rad_s=rate,hold=hold)


class LoadedTurn:
    def __init__(self,obs,action,delta):
        if isinstance(delta,bool) or not math.isfinite(delta) or not 0<abs(delta)<=math.pi/2:
            raise ValueError('turn must be nonzero and at most90 degrees')
        self.anchor=deepcopy(obs);self.reference=list(action);self.delta=delta
        self.phase='loaded_turn';self.started=obs['time'];self.rate=0.
        self.world_velocity=[0.,0.]
        self.history=deque(maxlen=51)

    def measurements(self,obs):return turn_measurements(self.anchor,obs,self.delta)

    def command(self,obs):
        action=list(self.reference);action[43:46]=[0.,0.,0.]
        if self.phase=='turn_hold':return action
        # World-frame position gain1/s, total planar cap0.04 m/s and vector
        # acceleration0.3 m/s². Keep position feedback during final settling.
        wanted_xy=[self.anchor['root_pos'][i]-obs['root_pos'][i] for i in range(2)]
        scale=min(1.,.04/max(math.hypot(*wanted_xy),1e-12))
        wanted_xy=[v*scale for v in wanted_xy]
        delta_xy=[v-old for v,old in zip(wanted_xy,self.world_velocity)]
        slew=min(1.,.006/max(math.hypot(*delta_xy),1e-12))
        self.world_velocity=[old+slew*d for old,d in zip(self.world_velocity,delta_xy)]
        angle=yaw(obs['root_quat']);c,s=math.cos(angle),math.sin(angle)
        vx,vy=self.world_velocity
        action[43:45]=[c*vx+s*vy,-s*vx+c*vy]
        wanted=0.
        if self.phase=='loaded_turn':
            error=self.measurements(obs)['heading_error_rad']
            wanted=math.copysign(min(.25,max(.08,abs(error))),self.delta)
        # The zero request also ramps down: no instantaneous stop at the goal.
        self.rate+=max(-.01,min(.01,wanted-self.rate))
        action[45]=self.rate
        return action

    def update(self,obs):
        self.history.append(deepcopy(obs));fault=turn_fault(self.anchor,obs,self.delta)
        if fault:return 'failed',fault
        m=self.measurements(obs)
        if self.phase=='loaded_turn':
            # Commanded-rate stopping distance w²/(2a), a=0.5 rad/s²,
            # plus 2 degrees reserved before the requested heading. This is
            # a development model, not an identified HOMIE stopping law.
            brake_angle=self.rate*self.rate/(2*.5)+math.radians(2)
            if abs(self.delta)-m['progress_rad']<=brake_angle:
                self.phase='settle_turn';self.started=obs['time'];self.history.clear()
            elif obs['time']-self.started>=12:return 'failed','turn_timeout'
        elif self.phase=='settle_turn':
            if settled_turn(self.history)['ready'] and abs(m['heading_error_rad'])<=math.radians(5):
                self.phase='turn_hold';self.started=obs['time'];self.history.clear()
            elif obs['time']-self.started>=5:return 'failed','turn_hold_timeout'
        elif obs['time']-self.started>=3:
            if settled_turn(self.history)['ready'] and abs(m['heading_error_rad'])<=math.radians(5):
                return 'completed','turn_and_zero_navigation_hold'
            return 'failed','zero_navigation_hold_not_verified'
        return None


def space_fault(obs, *, turning=False, admission=False):
    """Require measured support geometry; the turning region includes15cm drift."""
    surfaces=obs.get('surfaces')
    if not isinstance(surfaces,dict) or not {'source','destination'}<=surfaces.keys():
        return 'missing_support_geometry'
    distances=[s.get('lower_body_clearance_m') for s in surfaces.values()]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in distances):
        return 'invalid_support_geometry'
    if min(distances)<.03:return 'insufficient_support_clearance'
    if turning:
        value=obs.get('turn_clearance_m')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            return 'invalid_turn_geometry'
        if value<(.18 if admission else .03):return 'insufficient_turn_region'
    return None
