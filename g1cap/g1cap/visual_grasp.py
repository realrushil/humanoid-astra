"""Visual raised-box stability, not tactile contact or payload-force estimation."""
from collections import deque
from copy import deepcopy
import math
import numpy as np


class VisualGraspWindow:
    """One second of observed table gap and geometry relative to both wrists.

    A 5.5 cm observed gap leaves a 5 mm experiment reserve over the task's 5 cm
    lift criterion. Wrist proximity (35 cm), opposing sides, 5 cm/s translation,
    0.2 rad/s box/body rotation and 15-degree body tilt are uncalibrated G1 limits.
    They characterize a visual hypothesis; they do not prove bilateral contact.
    """
    def __init__(self):
        self.samples=deque()

    def add(self,sample):
        t=sample['time_s']
        if self.samples:
            dt=t-self.samples[-1]['time_s']
            if dt<=0 or dt>.150001:self.samples.clear()
        c=np.array(sample['box_center_pelvis_m']);r=np.array(sample['box_axes_pelvis'])
        wrists=[(np.array(sample['wrist_positions_pelvis_m'][s])-c)@r for s in ['left','right']]
        axis=int(np.argmax(abs(wrists[0]-wrists[1])))
        opposed=wrists[0][axis]*wrists[1][axis]<0 and abs(wrists[0][axis]-wrists[1][axis])>=sample['dimensions_m'][axis]
        near=max(np.linalg.norm(p) for p in wrists)<=.35
        up=np.array(sample['up_body']);up=up/np.linalg.norm(up)
        attitude_ok=up[2]>=math.cos(math.radians(15))
        raised=bool(sample['gap_m']>=.055 and opposed and near and attitude_ok)
        self.samples.append(dict(deepcopy(sample),raised=raised,retained=bool(opposed and near and attitude_ok)))
        # Keep the segment crossing one second, whatever the camera cadence.
        while len(self.samples)>2 and self.samples[1]['time_s']<=t-1.+1e-9:
            self.samples.popleft()
        ready=retention_stable=False;linear=[];angular=[];gap_speeds=[]
        if len(self.samples)>=2 and t-self.samples[0]['time_s']>=.999:
            for a,b in zip(self.samples,list(self.samples)[1:]):
                dt=b['time_s']-a['time_s']
                linear.extend(np.linalg.norm(np.array(b['box_center_wrist_m'][s])-a['box_center_wrist_m'][s])/dt for s in ['left','right'])
                angle=math.acos(float(np.clip((np.trace(np.array(a['box_axes_pelvis']).T@b['box_axes_pelvis'])-1)/2,-1,1)))
                angular.append(angle/dt);gap_speeds.append(abs(b['gap_m']-a['gap_m'])/dt)
            quiet=max(linear)<=.05 and max(angular)<=.2 and max(gap_speeds)<=.05
            retention_stable=all(s['retained'] for s in self.samples) and quiet
            ready=all(s['raised'] for s in self.samples) and quiet
        return dict(status='available',time_s=t,raised=raised,ready=bool(ready),gap_m=sample['gap_m'],
            retention_stable=bool(retention_stable),
            opposing_near_wrists=bool(opposed and near),attitude_ok=bool(attitude_ok),
            max_relative_speed_m_s=float(max(linear)) if linear else None,
            max_relative_rotation_rad_s=float(max(angular)) if angular else None,
            max_gap_speed_m_s=float(max(gap_speeds)) if gap_speeds else None)
