"""Short loaded retreat from observed front distance and coupled wrist holding.

The sensor callback supplies fresh floor-frame/grasp/front estimates only.
This controller owns navigation references; the existing paired wrist controller
and native actuator owner supply arm references and whole-body stabilization.
No rear free-space or lateral-position estimate is implied.
"""
import math

from .retreat_stop import RetreatStop


class SensorRetreat:
    def __init__(self, now_s, distance_m, observe, wrist_ready):
        self.observe, self.wrist_ready = observe, wrist_ready
        self.stopper = RetreatStop(distance_m, now_s)
        self.phase = 'sensor_retreat'
        self.braking = self.latest = None
        sample = self.sample(now_s)
        if not sample['grasp']['ready']:
            raise ValueError('carry_initial_grasp_not_ready')
        self.segment = sample['frame']['segment']
        self.rotation = sample['frame']['body_in_control_frame']
        self.offset = sample['front']['offset_body_m']
        self.normal = sample['front']['normal_segment']

    def sample(self, now_s):
        sample = self.observe(now_s)
        grasp, frame, front = (sample[k] for k in ('grasp','frame','front'))
        if grasp['status'] != 'available' or not grasp['raised']:
            raise ValueError('carry_visual_grasp_lost')
        if frame is None or frame['status'] == 'unavailable' or not 0 <= now_s-frame['time_s'] <= .150001:
            raise ValueError('carry_motion_unavailable')
        if front['status'] != 'available' or not 0 <= now_s-front['time_s'] <= .150001:
            raise ValueError('sensor_approach_unavailable')
        if abs(frame['time_s']-front['time_s']) > 1e-8:
            raise ValueError('carry_measurement_time_mismatch')
        if hasattr(self,'segment'):
            if frame['segment'] != self.segment:
                raise ValueError('carry_motion_unavailable')
            rotation = frame['body_in_control_frame']
            trace = sum(self.rotation[i][j]*rotation[i][j] for i in range(3) for j in range(3))
            if math.acos(max(-1.,min(1.,(trace-1.)/2))) > .2:
                raise ValueError('carry_rotation_drift')
            if sum(a*b for a,b in zip(self.normal,front['normal_segment'],strict=True)) < math.cos(math.radians(5)):
                raise ValueError('carry_front_identity_changed')
        return sample

    def command(self, now_s, previous):
        sample = self.sample(now_s)
        action = list(previous)
        action[43:46] = [0.,0.,0.]
        # A tool may begin between images. Keep the loaded references and zero
        # navigation until the shared wrist owner anchors at a camera timestamp.
        if not self.wrist_ready():
            if now_s-self.stopper.started > .150001:
                raise ValueError('carry_wrist_handoff_timeout')
            return action
        front = sample['front']
        progress = self.offset-front['offset_body_m']
        self.latest = dict(time_s=front['time_s'],retreat_m=progress,
                           reference='observed_source_front_normal',horizontal_position_observed=False)
        self.braking = self.stopper.update(now_s,front['time_s'],progress,sample['grasp']['ready'])
        self.phase = 'sensor_retreat' if self.braking['phase']=='drive' else 'sensor_settle'
        if self.braking['phase']=='drive':
            speed = -self.braking['speed_m_s']
            normal = front['normal_navigation_xy']
            action[43:46] = [speed*normal[0],speed*normal[1],0.]
        return action

    def update(self, now_s):
        self.sample(now_s)
        if self.braking and self.braking['outcome']=='completed':
            return 'completed','sensor_retreat_and_hold_completed'
        return None

    def measurements(self, now_s):
        return dict(self.latest or {},braking=dict(self.braking) if self.braking else None)
