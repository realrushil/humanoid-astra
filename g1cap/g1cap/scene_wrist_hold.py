"""Paired scene-relative wrist holding from encoders and local body estimates.

This supplies 14 arm references to the existing HOMIE actuator owner. It does
not command legs/fingers or infer grasp force. A fixed initial reference-minus-
measurement offset retains the loaded servo bias without integrating pose error.
The model is kinematic: its zero effort/velocity fields are NOT actuator limits.
"""
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

from .arena_sensors import proprioception_packet
from .scene_motion import _rigid


class SceneWristHold:
    def __init__(self,model,joint_names,packet,motion,reference,*,frame_key='body_in_segment'):
        if frame_key not in ('body_in_segment','body_in_control_frame'):
            raise ValueError('unsupported wrist control frame')
        self.frame_key=frame_key
        self.model=model;self.names=list(joint_names)
        self.arm_names=[n for n in self.names if any(p in n for p in ('shoulder','elbow','wrist'))]
        if len(self.names)!=43 or len(set(self.names))!=43 or len(self.arm_names)!=14:
            raise ValueError('expected G1 joint order and fourteen arm joints')
        self.ids=np.array([model.joints[model.getJointId(n)].idx_q for n in self.arm_names])
        self.action_ids=[self.names.index(n) for n in self.arm_names]
        self.frames=[model.getFrameId(s+'_wrist_yaw_link') for s in ('left','right')]
        self.segment=motion['segment'];self.invalid=False
        q,body=self.measurement(packet,motion)
        self.initial=q[self.ids].copy();self.solution=self.initial.copy()
        self.targets=[body*p for p in self.poses(q)]
        self.anchor_targets=[p.copy() for p in self.targets]
        self.heading_reference=None
        self.heading_anchors=None
        self.vertical_lift_m=0.
        self.reference=self.action(reference);self.goal=self.reference.copy()
        self.preload=self.reference[self.action_ids]-self.initial
        self.lo=np.maximum(model.lowerPositionLimit[self.ids],model.lowerPositionLimit[self.ids]-self.preload)
        self.hi=np.minimum(model.upperPositionLimit[self.ids],model.upperPositionLimit[self.ids]-self.preload)
        if np.any(self.initial<self.lo) or np.any(self.initial>self.hi):
            raise ValueError('initial joint/reference outside model limits')
        self.updated_at=packet['time_s'];self.commanded_at=self.updated_at
        self.last_solve=dict(status='anchored',time_s=self.updated_at,segment=self.segment)

    @staticmethod
    def action(value):
        result=np.asarray(value,float)
        if result.shape!=(50,) or not np.isfinite(result).all():raise ValueError('50 finite action values required')
        return result.copy()

    def measurement(self,packet,motion):
        clean=proprioception_packet(step=packet['step'],time_s=packet['time_s'],joint_names=packet['joint_names'],
            q=packet['q_rad'],dq=packet['dq_rad_s'],tau_est=packet['tau_est_nm'],
            gyro=packet['gyro_rad_s'],accel=packet['specific_force_m_s2'])
        if set(packet)!=set(clean) or packet['version']!=1:raise ValueError('unexpected sensor packet')
        if (motion['status']=='unavailable' or motion['segment']!=self.segment
                or abs(motion['time_s']-packet['time_s'])>1e-8):
            raise ValueError('scene_wrist_measurement_unavailable')
        body=pin.SE3(_rigid(motion[self.frame_key]))
        positions=dict(zip(packet['joint_names'],packet['q_rad'],strict=True))
        required=set(self.arm_names)|{'waist_yaw_joint','waist_roll_joint','waist_pitch_joint'}
        if not required<=positions.keys():raise ValueError('missing wrist ancestor measurements')
        q=pin.neutral(self.model)
        for name,value in positions.items():
            if not self.model.existJointName(name):raise ValueError('unknown measured joint')
            q[self.model.joints[self.model.getJointId(name)].idx_q]=value
        return q,body

    def poses(self,q):
        data=self.model.createData();pin.framesForwardKinematics(self.model,data,q)
        return [data.oMf[f].copy() for f in self.frames]

    def raise_targets(self,offset_m):
        """Monotonic equal shift along control-frame up; caller uses floor frame.

        The offset is absolute from this hold's anchor, not an accumulated delta.
        It persists across idle/wait; a new IK solve still checks the envelope.
        """
        if self.invalid:raise ValueError('scene_wrist_hold_invalidated')
        if (type(offset_m) not in (int,float) or not np.isfinite(offset_m)
                or not self.vertical_lift_m<=offset_m<=.04):
            raise ValueError('clearance_lift_budget_exhausted')
        self.vertical_lift_m=offset_m
        self.targets=[p.copy() for p in self.anchor_targets]
        for target in self.targets:target.translation[2]+=offset_m

    def follow_heading(self,motion,up):
        """Enable persistent co-rotation once per grasp; never reanchor preload.

        Relative turn objectives may restart, but the paired targets continue
        following measured floor heading through completion, idle and wait.
        The caller provides the current control frame; no position is inferred.
        """
        from .measured_turn import HeadingReference
        if (self.invalid or self.frame_key!='body_in_control_frame'
                or motion['status']=='unavailable' or motion['segment']!=self.segment
                or not 0<=motion['time_s']-self.updated_at<=.020001):
            raise ValueError('turn_wrist_owner_unavailable')
        if self.heading_reference is None:
            body=pin.SE3(_rigid(motion[self.frame_key]))
            self.heading_reference=HeadingReference(body.rotation,up)
            self.heading_anchors=[p.copy() for p in self.anchor_targets]

    def update(self,packet,motion):
        """At synchronized encoder/control-frame time; failed solves invalidate.

        A propagated frame is an internal control estimate with aged depth,
        not a fresh camera measurement. The owner enforces its source age.
        """
        if self.invalid:raise ValueError('scene_wrist_hold_invalidated')
        self.invalid=True
        q,body=self.measurement(packet,motion);t=packet['time_s']
        if not 0<t-self.updated_at<=.150001:raise ValueError('scene_wrist_update_time_gap')
        if self.heading_reference is not None:
            angle=self.heading_reference.angle(body.rotation)
            rotation=pin.SE3(self.heading_reference.rotation(angle),np.zeros(3))
            self.anchor_targets=[rotation*p for p in self.heading_anchors]
            self.targets=[p.copy() for p in self.anchor_targets]
            for target in self.targets:target.translation[2]+=self.vertical_lift_m
        desired=[body.inverse()*p for p in self.targets]
        def residual(arms):
            candidate=q.copy();candidate[self.ids]=arms;error=[]
            for actual,target in zip(self.poses(candidate),desired):
                error.extend(actual.translation-target.translation)
                error.extend(.1*pin.log3(target.rotation.T@actual.rotation))
            return np.r_[error,1e-4*(arms-self.initial)]
        result=least_squares(residual,np.clip(self.solution,self.lo+1e-8,self.hi-1e-8),
            bounds=(self.lo,self.hi),max_nfev=80,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        q[self.ids]=result.x;current=self.poses(q)
        position=max(float(np.linalg.norm(a.translation-b.translation)) for a,b in zip(current,desired))
        angle=max(float(np.linalg.norm(pin.log3(b.rotation.T@a.rotation))) for a,b in zip(current,desired))
        adjustment=float(np.max(np.abs(result.x-self.initial)))
        self.last_solve=dict(status='candidate',time_s=t,segment=self.segment,
            max_position_error_m=position,max_rotation_error_rad=angle,max_joint_adjustment_rad=adjustment,
            optimizer_success=bool(result.success),nfev=result.nfev)
        # Explicit simulation development envelope, not hardware qualification.
        # Optimizer convergence also includes the weak posture regularizer.
        # Budget exhaustion can still yield a valid task-space solution; verify
        # the actual finite wrist residuals and envelope independently.
        if not np.isfinite(result.x).all() or position>.002 or angle>.02 or adjustment>.35:
            raise ValueError('scene_wrist_target_outside_envelope')
        self.solution=result.x;self.goal=self.reference.copy()
        self.goal[self.action_ids]=self.solution+self.preload
        self.updated_at=t;self.invalid=False
        self.last_solve.update(status='solved',goal_arm_reference_rad=self.goal[self.action_ids].tolist())
        return dict(self.last_solve)

    def command(self,now_s,previous):
        """50 Hz vector slew (1 rad/s per joint); other references pass through."""
        if self.invalid:raise ValueError('scene_wrist_hold_invalidated')
        dt=now_s-self.commanded_at
        if not 0<=dt<=.020001 or not 0<=now_s-self.updated_at<=.150001:
            self.invalid=True;raise ValueError('scene_wrist_command_time_gap')
        output=self.action(previous);delta=self.goal[self.action_ids]-output[self.action_ids]
        # One scale retains the direction of the paired 14-joint correction.
        # The earlier 0.5 rad/s cap lagged measured 0.75 rad/s target demand
        # and prolonged coupled oscillation in a matched physics comparison.
        # This 1 rad/s simulation limit is not hardware calibration.
        scale=min(1.,dt/max(float(np.max(np.abs(delta))),1e-12))
        output[self.action_ids]+=scale*delta;self.commanded_at=now_s
        return output.tolist()
