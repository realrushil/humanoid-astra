"""Bounded robot-arm gravity compensation through position-servo references.

Known robot mass/COM and measured body/finger angles determine gravity torque;
estimated pelvis-frame up supplies its direction. No object mass or contact
signal is used. Dividing by known motor stiffness converts N m into radians.
This compensates robot weight only, not load, acceleration, friction or delay.
The 0.12 rad / 0.4 rad/s bounds are simulation assumptions, not hardware limits.
"""
import math
import numpy as np
import pinocchio as pin

from .hand_sensors import measured_joint_positions


class ArmGravity:
    def __init__(self,model,names,stiffness,masses):
        self.model=model;self.data=model.createData();self.names=list(names)
        self.fk_names=list(model.names)[1:]
        self.arms=[n for n in names if any(k in n for k in ('shoulder','elbow','wrist'))]
        if (model.nq!=43 or model.nv!=43 or len(self.names)!=43 or
                set(self.names)!=set(self.fk_names) or len(self.arms)!=14 or set(stiffness)!=set(self.arms)):
            raise ValueError('complete G1 mapping and fourteen arm stiffnesses required')
        self.ids=[self.names.index(n) for n in self.arms]
        self.qids=[model.joints[model.getJointId(n)].idx_q for n in self.arms]
        self.kp=np.asarray([stiffness[n] for n in self.arms],float)
        if not np.isfinite(self.kp).all() or np.any(self.kp<=0):raise ValueError('positive finite stiffness required')
        required=set()
        for frame in model.frames:
            if frame.type!=pin.FrameType.BODY:continue
            joint=frame.parentJoint
            while joint:
                if model.names[joint] in self.arms:required.add(frame.name);break
                joint=model.parents[joint]
        if set(masses)!=required:raise ValueError('mass/COM must cover exactly the arm descendant links')
        self.bodies=[]
        for name,body in masses.items():
            mass=body['mass_kg'];com=np.asarray(body['com_m'],float)
            if not math.isfinite(mass) or mass<=0 or com.shape!=(3,) or not np.isfinite(com).all():
                raise ValueError('finite authored robot mass/COM required')
            self.bodies.append((model.getFrameId(name),mass,com))
        self.offset=np.zeros(14);self.time=self.step=None;self.failed=False;self.last=None

    def command(self,body,hands,up_body,targets):
        """Once per 50 Hz sample; return 43 targets with only arm entries changed."""
        if self.failed:raise ValueError('arm_gravity_invalidated')
        try:return self._command(body,hands,up_body,targets)
        except (KeyError,TypeError,ValueError):
            self.failed=True
            raise ValueError('arm_gravity_invalid_measurement') from None

    def _command(self,body,hands,up_body,targets):
        q=np.asarray(measured_joint_positions(body,hands,self.fk_names))
        up=np.asarray(up_body,float);output=np.asarray(targets,float).copy()
        if (up.shape!=(3,) or not np.isfinite(up).all() or abs(up@up-1.)>1e-5 or
                output.shape!=(43,) or not np.isfinite(output).all()):raise ValueError('invalid vectors')
        now=body['time_s'];dt=0. if self.time is None else now-self.time
        if self.time is not None and (body['step']!=self.step+1 or abs(dt-.02)>1e-6):
            raise ValueError('missing sample')
        pin.computeJointJacobians(self.model,self.data,q);pin.updateFramePlacements(self.model,self.data)
        gravity=np.zeros(self.model.nv)
        for frame,mass,com in self.bodies:
            r=self.data.oMf[frame].rotation@com
            j=pin.getFrameJacobian(self.model,self.data,frame,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            # COM linear Jacobian = origin translation minus [r]x rotation.
            gravity+=(j[:3]-pin.skew(r)@j[3:]).T@(9.81*mass*up)
        torque=gravity[self.qids]
        desired=np.clip(torque/self.kp,-.12,.12)
        self.offset+=np.clip(desired-self.offset,-.4*dt,.4*dt)
        before=output[self.ids].copy();requested=before+self.offset
        output[self.ids]=np.clip(requested,self.model.lowerPositionLimit[self.qids],self.model.upperPositionLimit[self.qids])
        self.time,self.step=now,body['step']
        self.last=dict(step=self.step,time_s=now,arm_names=self.arms,gravity_nm=torque.tolist(),
            desired_offset_rad=desired.tolist(),slew_offset_rad=self.offset.tolist(),
            applied_offset_rad=(output[self.ids]-before).tolist(),incoming_arm_rad=before.tolist(),
            outgoing_arm_rad=output[self.ids].tolist(),joint_limit_clipped=bool(np.any(output[self.ids]!=requested)))
        return output
