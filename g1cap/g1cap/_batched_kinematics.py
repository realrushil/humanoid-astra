"""Batched known-robot encoder kinematics for whole-body clearance screens."""
import numpy as np
import pinocchio as pin


def frames_in_body(model,frames,joint_positions,*,body='pelvis'):
    """Return capture-time T_body_frame matrices after one forward-kinematics pass.

    Joint readings are radians. Translations are metres in the specified body
    frame (pelvis by default: x forward, y left, z up). This uses known robot
    geometry only. Preserve the single-frame helper's ancestor validation and
    rejection of malformed readings; do not invent missing branch measurements.
    No state or transforms are cached across calls.
    """
    frames=list(frames)
    if any(j.nq!=1 for j in model.joints[1:]):
        raise ValueError('expected fixed-base model with one-coordinate G1 joints')
    required=set()
    for name in frames+[body]:
        if not model.existFrame(name):
            raise ValueError(f'unknown robot frame: {name}')
        joint_id=model.frames[model.getFrameId(name)].parentJoint
        while joint_id:
            required.add(model.names[joint_id])
            joint_id=model.parents[joint_id]
    missing=required-joint_positions.keys()
    if missing:
        raise ValueError(f'missing ancestor joint measurements: {sorted(missing)}')
    q=pin.neutral(model)
    for name,value in joint_positions.items():
        if not model.existJointName(name) or name=='universe':
            raise ValueError(f'unknown measured joint: {name}')
        if isinstance(value,bool) or not np.isfinite(value):
            raise ValueError(f'invalid measurement for {name}')
        q[model.joints[model.getJointId(name)].idx_q]=value
    data=model.createData()
    pin.framesForwardKinematics(model,data,q)
    body_inverse=data.oMf[model.getFrameId(body)].inverse()
    return {frame:(body_inverse*data.oMf[model.getFrameId(frame)]).homogeneous.copy()
            for frame in frames}
