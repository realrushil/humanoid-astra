"""Temporary sensor-only world adapter for destination approach; no motor writes."""
from copy import deepcopy
import math

def destination_observation(world,now_s):
    """Camera-time stance geometry plus current heading, metres/radians.

    StanceMotion integrates gyro orientation in initial local axes. Its latest
    camera pose supplies translation and segment identity, not current heading.
    No source height, world pose, contacts or obstacle metadata are consulted.
    The approach controller separately rejects stale/mismatched camera samples.
    """
    perception=world.box_perception;motion=perception.motion
    if type(now_s) not in (int,float) or not math.isfinite(now_s):
        raise ValueError('destination_clock_invalid')
    if (type(motion.imu_time) not in (int,float) or not math.isfinite(motion.imu_time)
            or abs(motion.imu_time-now_s)>1e-8):
        raise ValueError('destination_imu_time_mismatch')
    if world.sensor_recorder.latest_packet['time_s']!=now_s:
        raise ValueError('destination_packet_time_mismatch')
    frame=perception.latest_motion
    if frame is None or frame.get('status')!='tracked_local_segment':
        raise ValueError('destination_motion_unavailable')
    world.control.carry_admission.require_owner(world.scene_wrist_hold)
    rotation=motion.rotation
    yaw=math.atan2(float(rotation[1,0]),float(rotation[0,0]))
    return dict(destination=world.destination_observation.observe(now_s),
        retention=world.control.carry_feedback(now_s),frame=deepcopy(frame),
        navigation=dict(time_s=now_s,yaw_segment_rad=yaw,segment=frame['segment']))
