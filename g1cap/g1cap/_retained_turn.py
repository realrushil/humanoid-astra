"""Disposable retained-object turn; admission and path screen remain external."""
import math
import numpy as np
from g1cap.measured_turn import MeasuredHeadingTurn

class RetainedTurn:
    def __init__(self,now_s,angle_rad,observe):
        self.observe=observe;self.phase='sensor_turn';self.latest=None;self.failure=None
        sample=self.sample(now_s)
        if sample['retention']['settled'] is not True:raise ValueError('carry_not_settled')
        self.identity=(sample['retention']['track_epoch'],sample['retention']['segment'])
        self.turn=MeasuredHeadingTurn(now_s,angle_rad,self.rotation(sample),sample['up'],
            minimum_yaw_rate=.08,require_sustained_readiness=True)

    @staticmethod
    def rotation(sample):return np.asarray(sample['frame']['body_in_control_frame'],float)[:3,:3]

    def sample(self,now_s):
        if self.failure:raise ValueError(self.failure)
        try:
            sample=self.observe(now_s);f=sample['frame'];r=sample['retention']
            if (f is None or f['status']=='unavailable' or not math.isfinite(f['time_s'])
                    or abs(f['time_s']-now_s)>1e-8):raise ValueError('turn_control_frame_unavailable')
            identity=(r.get('track_epoch'),r.get('segment'))
            if (r.get('status')!='available' or r.get('retained') is not True
                    or any(type(v) is not int for v in identity) or f['segment']!=identity[1]
                    or hasattr(self,'identity') and self.identity!=identity):raise ValueError('carry_retention_unavailable')
            stamp=r.get('time_s')
            if type(stamp) not in (int,float) or not math.isfinite(stamp) or not 0<=now_s-stamp<=.150001:
                raise ValueError('turn_ready_camera_time_invalid')
            return sample
        except (KeyError,TypeError,ValueError) as error:
            self.failure=str(error);raise ValueError(self.failure) from None

    def command(self,now_s,previous):
        s=self.sample(now_s)
        self.latest=self.turn.update(now_s,self.rotation(s),s['retention']['settled'],observed_at_s=s['retention']['time_s'])
        self.phase='sensor_turn' if self.latest['phase']=='turn_drive' else 'sensor_turn_settle'
        action=list(previous);action[43:46]=[0.,0.,self.latest['yaw_rate_command_rad_s']]
        return action

    def update(self,now_s):
        s=self.sample(now_s)
        if self.latest and self.latest['outcome']:
            if self.latest['outcome']=='completed':
                error=self.turn.goal-self.turn.reference.angle(self.rotation(s))
                if not s['retention']['settled'] or abs(error)>.05:return 'failed','turn_completion_not_retained'
            return self.latest['outcome'],self.latest['reason']
        return None

    def measurements(self,now_s):return dict(self.latest or {})
