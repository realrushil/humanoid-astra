"""Camera-time visual retention, separate from pickup proof and path clearance."""
from collections import deque
import math
import numpy as np
from g1cap.scene_motion import SceneStability,_rigid,_vector

class VisualRetention:
    def __init__(self):
        self.clear()

    def clear(self):
        self.history=deque();self.scene=SceneStability();self.identity=None

    def update(self,*,time_s,box,relative,motion,camera_transform,up_body):
        """Camera-time metres/radians; original .05m/s/.2rad/s/15deg limits.

        This tests a visually retained object, which can still rest on a table.
        Pickup proof and all source/destination/floor path screens are separate.
        """
        t=time_s
        try:
            if type(t) not in (int,float) or not math.isfinite(t) or t<0:raise ValueError('invalid_time')
            if box.get('status')!='accepted' or relative is None or relative.get('status')!='accepted':raise ValueError('box_unavailable')
            if motion.get('status')!='tracked_local_segment':raise ValueError('motion_unavailable')
            for stamp in (box.get('observed_at_s'),relative.get('observed_at_s'),motion.get('time_s')):
                if type(stamp) not in (int,float) or not math.isfinite(stamp) or abs(stamp-t)>1e-8:raise ValueError('measurement_time_mismatch')
            if type(box.get('track_epoch')) is not int or box['track_epoch']!=relative.get('track_epoch'):raise ValueError('box_identity_mismatch')
            if type(motion.get('segment')) is not int:raise ValueError('motion_identity_missing')
            identity=(box['track_epoch'],motion['segment'])
            if self.identity is not None and identity!=self.identity:self.clear()
            if self.history:
                dt=t-self.history[-1]['time_s']
                if dt<=0:raise ValueError('camera_time_not_increasing')
                if dt>.150001:self.clear()
            self.identity=identity
            center=_vector(relative['box_center_pelvis_m']);axes=np.asarray(relative['box_axes_pelvis'],float)
            T=np.eye(4);T[:3,:3]=axes;_rigid(T)
            dimensions=_vector(box['dimensions_m'])
            if min(dimensions)<=0:raise ValueError('invalid_dimensions')
            up=_vector(up_body)
            if abs(np.linalg.norm(up)-1)>1e-5:raise ValueError('invalid_up')
            wrists=[(_vector(relative['wrist_positions_pelvis_m'][s])-center)@axes for s in ('left','right')]
            relative_centers={s:_vector(relative['box_center_wrist_m'][s]) for s in ('left','right')}
            axis=int(np.argmax(abs(wrists[0]-wrists[1])))
            opposed=wrists[0][axis]*wrists[1][axis]<0 and abs(wrists[0][axis]-wrists[1][axis])>=dimensions[axis]
            near=max(np.linalg.norm(w) for w in wrists)<=.35
            attitude=up[2]>=math.cos(math.radians(15));retained=bool(opposed and near and attitude)
            self.history.append(dict(time_s=t,centers=relative_centers,axes=axes.copy(),retained=retained))
            while len(self.history)>2 and self.history[1]['time_s']<=t-1.+1e-9:self.history.popleft()
            linear=[];angular=[];quiet=False
            if len(self.history)>=2 and t-self.history[0]['time_s']>=.999:
                for a,b in zip(self.history,list(self.history)[1:]):
                    dt=b['time_s']-a['time_s']
                    linear.extend(float(np.linalg.norm(b['centers'][s]-a['centers'][s])/dt) for s in ('left','right'))
                    angular.append(math.acos(float(np.clip((np.trace(a['axes'].T@b['axes'])-1)/2,-1,1)))/dt)
                quiet=all(s['retained'] for s in self.history) and max(linear)<=.05 and max(angular)<=.2
            scene=self.scene.update(t,motion,box,camera_transform,up,grasp_ready=quiet)
            return dict(status='available',time_s=t,track_epoch=identity[0],segment=identity[1],retained=retained,
                settled=bool(scene['ready']),wrist_relative_settled=bool(quiet),
                opposing_near_wrists=bool(opposed and near),attitude_ok=bool(attitude),
                max_relative_speed_m_s=max(linear) if linear else None,
                max_relative_rotation_rad_s=max(angular) if angular else None,
                scene=scene,pickup_proven=False)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            self.clear()
            return dict(status='unavailable',time_s=t,reason=str(error),retained=False,settled=False,pickup_proven=False)
