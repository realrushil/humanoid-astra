"""Bounded SONIC command sequences, not IK, collision or task planning.

World XY metres/second; facing is world yaw radians. Named right-arm references
are joint radians. All bounds are development assumptions, not calibration.
"""
import math
from ..models import finite_number,wrap_yaw
from .arm_planner import RIGHT_ARM,UPPER_BODY,upper_reference
from .sonic_backend import LeasedPlanner


def prepare_motion(segments, sample, planner):
    """Validate everything without commanding motors; return ordinary segment dicts."""
    if not isinstance(segments,list) or not 1<=len(segments)<=8:
        raise ValueError('provide 1–8 motion segments')
    allowed={'mode','duration','velocity_world','facing_world','height','right_arm'}
    prepared=[];total=0.;facing=sample['pelvis_yaw'];entry_facing=facing
    positions=None;entry_positions=None
    for segment in segments:
        if not isinstance(segment,dict) or set(segment)-allowed:
            raise ValueError('unknown motion segment fields')
        mode={'idle':0,'walk':1,'squat':4}.get(segment.get('mode'))
        duration=segment.get('duration')
        if mode is None or not finite_number(duration) or not .2<=duration<=2.:
            raise ValueError('mode idle/walk/squat and duration [0.2,2] seconds required')
        total+=duration
        if total>8.+1e-9:raise ValueError('sequence exceeds 8 simulation seconds')
        velocity=segment.get('velocity_world',[0.,0.])
        height=segment.get('height',-1.)
        end_facing=segment.get('facing_world',facing)
        if not finite_number(end_facing):raise ValueError('finite world facing required')
        yaw_delta=wrap_yaw(end_facing-facing)
        if abs(wrap_yaw(end_facing-entry_facing))>.4+1e-9 or abs(yaw_delta)/duration>.4+1e-9:
            raise ValueError('facing exceeds entry +/-0.4 rad or 0.4 rad/s reference rate')
        start_positions=positions
        if 'right_arm' in segment:
            targets=segment['right_arm']
            if not isinstance(targets,dict) or not targets or set(targets)-set(RIGHT_ARM):
                raise ValueError('right_arm requires named right-arm joint targets')
            if planner is None:raise ValueError('model joint limits unavailable')
            if entry_positions is None:entry_positions=upper_reference(sample)
            start_positions=list(positions if positions is not None else entry_positions)
            positions=start_positions.copy()
            for name,value in targets.items():
                if not finite_number(value):raise ValueError('finite joint radians required')
                positions[UPPER_BODY.index(name)]=value
            for j,name in enumerate(RIGHT_ARM):
                i=UPPER_BODY.index(name)
                if not planner.lower[j]<=positions[i]<=planner.upper[j]:
                    raise ValueError('right-arm target exceeds model joint limits')
                if abs(positions[i]-entry_positions[i])>.20+1e-9:
                    raise ValueError('right-arm target exceeds entry +/-0.20 rad')
                if abs(positions[i]-start_positions[i])/duration>.35+1e-9:
                    raise ValueError('right-arm reference exceeds 0.35 rad/s')
        # The same transport validator runs again at the supervisor boundary.
        LeasedPlanner(0.,0.).motion(mode,velocity,end_facing,height,positions,now=0.)
        prepared.append(dict(duration=float(duration),mode=mode,velocity_world=list(velocity),height=height,
                             facing_start=facing,facing_delta=yaw_delta,
                             positions_start=start_positions,positions_end=positions))
        facing=wrap_yaw(end_facing)
    return prepared


def motion_fields(segment,elapsed):
    """Reference interpolation only; it does not interpolate the physical robot."""
    fraction=max(0.,min(1.,elapsed/segment['duration']))
    positions=None
    if segment['positions_end'] is not None:
        positions=[a+(b-a)*fraction for a,b in zip(segment['positions_start'],segment['positions_end'])]
    return dict(mode=segment['mode'],velocity_world=segment['velocity_world'],height=segment['height'],
                facing_world=wrap_yaw(segment['facing_start']+fraction*segment['facing_delta']),positions=positions)
