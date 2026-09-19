"""Loaded relative turn from current IMU heading and fresh RGB-D readiness.

Only navigation is owned here. The persistent paired wrist owner follows
measured heading independently, including after completion/cancellation.
Angles are radians about floor up; XY position is not controlled or measured.
"""
import math
import numpy as np
from .measured_turn import MeasuredHeadingTurn


class SensorTurn:
    def __init__(self,now_s,angle_rad,observe):
        self.observe=observe
        self.phase='sensor_turn'
        self.latest=None
        sample=self.sample(now_s)
        if not sample['grasp']['ready']:raise ValueError('turn_initial_grasp_not_ready')
        self.segment=sample['frame']['segment']
        # Native HOMIE switches to standing below command norm 0.05.
        # Keep active pure yaw above that switch; zero yaw still means settle.
        self.turn=MeasuredHeadingTurn(now_s,angle_rad,self.rotation(sample),sample['up'],
            minimum_yaw_rate=.08,require_sustained_readiness=True)

    @staticmethod
    def rotation(sample):
        return np.asarray(sample['frame']['body_in_control_frame'],float)[:3,:3]

    def sample(self,now_s):
        sample=self.observe(now_s);frame=sample['frame'];grasp=sample['grasp']
        if (frame is None or frame['status']=='unavailable'
                or not math.isfinite(frame['time_s']) or abs(now_s-frame['time_s'])>1e-8
                or hasattr(self,'segment') and frame['segment']!=self.segment):
            raise ValueError('turn_control_frame_unavailable')
        if (grasp['status']!='available' or not grasp['opposing_near_wrists']
                or not grasp['attitude_ok'] or grasp['gap_m']<=.02):
            raise ValueError('hold_visual_retention_lost')
        observed=grasp.get('time_s')
        if (type(observed) not in (int,float) or not math.isfinite(observed)
                or not 0<=now_s-observed<=.150001):
            raise ValueError('turn_ready_camera_time_invalid')
        return sample

    def command(self,now_s,previous):
        sample=self.sample(now_s)
        self.latest=self.turn.update(now_s,self.rotation(sample),sample['grasp']['ready'],
                                     observed_at_s=sample['grasp']['time_s'])
        self.phase='sensor_turn' if self.latest['phase']=='turn_drive' else 'sensor_turn_settle'
        action=list(previous);action[43:46]=[0.,0.,self.latest['yaw_rate_command_rad_s']]
        return action

    def update(self,now_s):
        sample=self.sample(now_s)
        if self.latest and self.latest['outcome']:
            if self.latest['outcome']=='completed':
                # update follows a physics step. A command-time completion
                # candidate cannot certify a newly out-of-tolerance state.
                error=self.turn.goal-self.turn.reference.angle(self.rotation(sample))
                if not sample['grasp']['ready'] or abs(error)>.05:
                    return 'failed','turn_completion_not_retained'
            return self.latest['outcome'],self.latest['reason']
        return None

    def measurements(self,now_s):
        return dict(self.latest or {})
