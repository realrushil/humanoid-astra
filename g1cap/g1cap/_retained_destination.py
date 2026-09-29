"""Temporary retained-carry destination approach; no runtime wiring or motion qualification.

Metres/radians in a continuous local stance segment. The screen must certify
observed travel space, body/box/arm paths and stopping; a plane candidate alone
cannot do that. No live factory or public tool exposes this controller.
"""
import math


def number(value):
    return type(value) in (int,float) and math.isfinite(value)


def rigid(value):
    try:
        if len(value)!=4 or any(len(row)!=4 for row in value):raise ValueError()
        if not all(number(x) for row in value for x in row):raise ValueError()
        if any(abs(a-b)>1e-6 for a,b in zip(value[3],[0,0,0,1])):raise ValueError()
        r=[row[:3] for row in value[:3]]
        for i in range(3):
            for j in range(3):
                if abs(sum(r[k][i]*r[k][j] for k in range(3))-(i==j))>1e-6:raise ValueError()
        det=sum(r[0][i]*(r[1][(i+1)%3]*r[2][(i+2)%3]-r[1][(i+2)%3]*r[2][(i+1)%3]) for i in range(3))
        if abs(det-1)>1e-6:raise ValueError()
    except (TypeError,ValueError,IndexError):raise ValueError('destination_geometry_invalid') from None
    return value


class SensorDestinationApproach:
    def __init__(self,now_s,distance_m,observe,wrist_ready,*,screen,admission,owner):
        if not number(now_s) or now_s<0:raise ValueError('destination_clock_invalid')
        if not number(distance_m) or not .20<=distance_m<=.50:raise ValueError('distance_outside_envelope')
        if not callable(screen):raise ValueError('destination_path_screen_required')
        from g1cap.carry_admission import CarryAdmission
        if not isinstance(admission,CarryAdmission) or not callable(owner):
            raise ValueError('destination_carry_admission_required')
        self.admission,self.owner=admission,owner
        self.observe,self.wrist_ready,self.screen=observe,wrist_ready,screen
        self.distance=float(distance_m);self.started=self.previous=self.command_time=float(now_s)
        self.segment=self.epoch=self.direction=self.anchor=None
        self.failure=None;self.speed=0.;self.phase='destination_approach'
        self.last_camera=None;self.last_progress=None;self.settled_since=None
        self.zero_since=None;self.latest=None
        sample=self.sample(now_s)
        if sample['retention']['settled'] is not True:self.fail('destination_initial_carry_not_settled')
        self.segment=sample['frame']['segment'];self.epoch=sample['destination']['view_epoch']
        self.anchor=[row[3] for row in sample['frame']['body_in_segment'][:3]]
        direction=sample['destination']['direction_segment_xy'];length=math.hypot(*direction)
        self.direction=[x/length for x in direction]

    def fail(self,reason):
        self.failure=reason;self.speed=0.;self.phase='destination_failed'
        raise ValueError(reason)

    def sample(self,now_s):
        if self.failure:self.fail(self.failure)
        if not number(now_s) or not 0<=now_s-self.previous<=.150001:self.fail('destination_motion_time_gap')
        self.previous=now_s
        if now_s-self.started>12.:self.fail('destination_approach_timeout')
        try:
            sample=self.observe(now_s);d,g,f=sample['destination'],sample['retention'],sample['frame']
            if d.get('status')!='observed_destination_candidate':self.fail('destination_observation_unavailable')
            stamp=d.get('observed_at_s')
            if not number(stamp) or not 0<=now_s-stamp<=.150001:self.fail('destination_observation_stale')
            if (f.get('status')!='tracked_local_segment' or not number(f.get('time_s'))
                    or abs(f['time_s']-stamp)>1e-8 or f.get('segment') is None):self.fail('destination_motion_unavailable')
            rigid(f['body_in_segment'])
            nav=sample['navigation']
            if (not number(nav.get('time_s')) or abs(nav['time_s']-now_s)>1e-8
                    or not number(nav.get('yaw_segment_rad')) or nav.get('segment')!=f['segment']):
                self.fail('destination_navigation_frame_unavailable')
            if (d.get('direction_status')!='available' or d.get('segment_id')!=f['segment']
                    or type(d.get('view_epoch')) is not int):self.fail('destination_geometry_invalid')
            if self.segment is not None and (f['segment']!=self.segment or d['view_epoch']!=self.epoch):self.fail('destination_identity_changed')
            direction=d.get('direction_segment_xy')
            if not isinstance(direction,(list,tuple)) or len(direction)!=2 or not all(number(x) for x in direction):self.fail('destination_geometry_invalid')
            length=math.hypot(*direction)
            if not .999<=length<=1.001:self.fail('destination_geometry_invalid')
            if self.direction is not None and sum(a*b/length for a,b in zip(self.direction,direction))<math.cos(.2):self.fail('destination_direction_changed')
            # Proof was recorded on completed pickup and bound to the active
            # paired wrist owner by the caller. This controller never arms,
            # resets or rebinds it. Relative retention grants no path clearance.
            self.admission.require_owner(self.owner())
            if not self.admission.observe(now_s,g):
                self.fail(self.admission.failure or 'carry_pickup_not_verified')
            if g.get('segment')!=f['segment']:self.fail('destination_carry_segment_mismatch')
            if not number(g.get('time_s')) or abs(g['time_s']-stamp)>1e-8:self.fail('destination_measurement_time_mismatch')
            if self.last_camera is not None and stamp<self.last_camera:self.fail('destination_camera_time_reversed')
            return sample
        except (KeyError,TypeError,ValueError) as error:
            self.fail(self.failure or (str(error) if isinstance(error,ValueError) else 'destination_observation_invalid'))

    def progress(self,sample):
        p=[row[3] for row in sample['frame']['body_in_segment'][:3]]
        value=sum((p[i]-self.anchor[i])*self.direction[i] for i in range(2))
        if value<-.03 or value>self.distance+.03:self.fail('destination_approach_overshoot')
        # A straight short segment cannot silently become lateral wandering.
        lateral=abs((p[0]-self.anchor[0])*self.direction[1]-(p[1]-self.anchor[1])*self.direction[0])
        if lateral>.03:self.fail('destination_lateral_drift')
        return value

    def command(self,now_s,previous):
        old_time=self.command_time;sample=self.sample(now_s);progress=self.progress(sample)
        self.command_time=now_s
        if len(previous)!=50 or not all(number(x) for x in previous):self.fail('destination_action_invalid')
        action=list(previous);action[43:46]=[0.,0.,0.]
        ready=self.wrist_ready()
        if not ready and now_s-self.started>.150001:self.fail('destination_wrist_handoff_timeout')
        remaining=self.distance-progress
        if ready and remaining>.02:
            self.speed=min(.08,self.speed+.30*min(now_s-old_time,.02))
            yaw=sample['navigation']['yaw_segment_rad']
            # Native navigation XY follows current IMU pelvis heading. The
            # camera-time stance rotation must not stand in for current heading.
            c,s=math.cos(yaw),math.sin(yaw);x,y=self.direction
            action[43:46]=[self.speed*(c*x+s*y),self.speed*(-s*x+c*y),0.]
            self.phase='destination_approach';self.zero_since=None;self.settled_since=None
        else:
            self.speed=0.;self.phase='destination_settle' if ready else 'destination_wrist_handoff'
            if self.zero_since is None:self.zero_since=now_s
        try:certificate=self.screen(now_s,sample,list(action))
        except Exception:self.fail('destination_path_unavailable')
        if (not isinstance(certificate,dict) or certificate.get('status')!='clear'
                or not number(certificate.get('observed_at_s'))
                or abs(certificate['observed_at_s']-sample['destination']['observed_at_s'])>1e-8):self.fail('destination_path_unavailable')
        self.latest=dict(time_s=now_s,observed_at_s=sample['destination']['observed_at_s'],
            progress_m=progress,target_m=self.distance,phase=self.phase)
        return action

    def update(self,now_s):
        sample=self.sample(now_s);progress=self.progress(sample);stamp=sample['destination']['observed_at_s']
        if self.latest is None or not 0<=now_s-self.latest['time_s']<=.020001:
            self.fail('destination_command_stale')
        if stamp==self.last_camera:return None
        continuous=self.last_camera is not None and 0<stamp-self.last_camera<=.150001
        slow=(continuous and abs(progress-self.last_progress)/(stamp-self.last_camera)<=.02)
        settled=(self.phase=='destination_settle' and self.zero_since is not None
            and stamp>=self.zero_since and abs(progress-self.distance)<=.02
            and sample['retention']['settled'] is True and self.wrist_ready() and slow)
        self.settled_since=(self.settled_since if self.settled_since is not None else stamp) if settled else None
        self.last_camera,self.last_progress=stamp,progress
        if self.settled_since is not None and stamp-self.settled_since>=1.-1e-8:
            return 'completed','destination_approach_and_hold'
        return None

    def measurements(self,now_s):
        return dict(self.latest or {},settled_since_s=self.settled_since,failure=self.failure)
