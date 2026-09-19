"""Dex3-1 finger positions: radians, synchronized with the body packet.

Simulation assumption approved by the user; hardware timing/calibration remain
unconfirmed. DDS order is thumb 0/1/2, middle 0/1, index 0/1 per hand. Pressure,
finger torque and wrist force signals are deliberately outside this contract.
"""
from .arena_sensors import _finite,proprioception_packet

DEX3_JOINTS=tuple(f'{side}_hand_{finger}_{joint}_joint'
    for side in ('left','right')
    for finger,count in (('thumb',3),('middle',2),('index',2))
    for joint in range(count))


def hand_position_packet(*,step,time_s,left,right):
    if type(step) is not int or step<0:raise ValueError('invalid hand sample step')
    _finite([time_s],1,'hand time')
    if time_s<0:raise ValueError('invalid hand sample time')
    return dict(version=1,model='unitree_dex3_1',step=step,time_s=time_s,
        joint_names=list(DEX3_JOINTS),q_rad=_finite(left,7,'left q')+_finite(right,7,'right q'))


def validated_hand_packet(packet):
    if set(packet)!={'version','model','step','time_s','joint_names','q_rad'}:
        raise ValueError('unexpected hand packet fields')
    q=_finite(packet['q_rad'],14,'hand q')
    clean=hand_position_packet(step=packet['step'],time_s=packet['time_s'],left=q[:7],right=q[7:])
    if packet!=clean:raise ValueError('invalid hand model, joint order or version')
    return clean


def measured_joint_positions(body,hands,names):
    """Assemble native joint order from measurements, never command/default poses."""
    hand=validated_hand_packet(hands)
    clean=proprioception_packet(step=body['step'],time_s=body['time_s'],
        joint_names=body['joint_names'],q=body['q_rad'],dq=body['dq_rad_s'],
        tau_est=body['tau_est_nm'],gyro=body['gyro_rad_s'],accel=body['specific_force_m_s2'])
    if body!=clean:raise ValueError('unexpected body packet fields or version')
    if hand['step']!=body['step'] or abs(hand['time_s']-body['time_s'])>1e-8:
        raise ValueError('body/hand timestamps differ')
    pairs=list(zip(body['joint_names'],body['q_rad']))+list(zip(hand['joint_names'],hand['q_rad']))
    joints=dict(pairs)
    if len(joints)!=len(pairs) or len(names)!=len(joints) or set(names)!=set(joints):
        raise ValueError('native joint mapping must exactly cover the measured joints')
    return [joints[n] for n in names]


def acquisition_joint_sample(body,hands,camera,names):
    """Current encoders with explicitly aged 10 Hz RGB for native acquisition.

    The geometry frontend pairs images/encoders at image time separately. This
    learned-policy interface uses current encoders and the latest available RGB;
    it never relabels that image as current or accepts an expired/future frame.
    """
    q=measured_joint_positions(body,hands,names)
    _finite([camera['time_s']],1,'camera time')
    age=body['time_s']-camera['time_s']
    if (type(camera['step']) is not int or not 0<=camera['step']<=body['step'] or
            camera['time_s']<0 or not 0<=age<=.150001):
        raise ValueError('native acquisition camera is stale or inconsistent')
    return dict(step=body['step'],time_s=body['time_s'],camera_step=camera['step'],
        camera_time_s=camera['time_s'],camera_age_s=age,joint_names=list(names),q_rad=q)
