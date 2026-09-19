"""Experimental hold observation v2, based on measured pose motion.

Positions: simulation-world metres. Orientations: WXYZ body-to-world unit
quaternions. Times: simulation seconds. Raw solver velocity is retained by the
caller as a separate diagnostic; it is not substituted for pose derivatives.
"""
import math

def angular_distance(a,b):
    norm=lambda q: math.sqrt(sum(x*x for x in q))
    dot=sum(x*y for x,y in zip(a,b))/(norm(a)*norm(b))
    return 2*math.acos(min(1.,abs(dot)))

def assess_hold(samples):
    if len(samples)<2 or samples[-1]['time']-samples[0]['time']<.999:
        return {'ready':False,'reason':'need_one_second'}
    speeds=[]; angular=[]; root_speeds=[]
    for a,b in zip(samples,samples[1:]):
        dt=b['time']-a['time']
        if dt<=0 or dt>.020001:
            return {'ready':False,'reason':'invalid_or_missing_sample'}
        speeds.append(math.dist(a['box_pos'],b['box_pos'])/dt)
        angular.append(angular_distance(a['box_quat'],b['box_quat'])/dt)
        root_speeds.append(math.dist(a['root_pos'][:2],b['root_pos'][:2])/dt)
    stable=all(s['bilateral'] and s['clearance']>=.05 and s['tilt']<=math.radians(15) for s in samples)
    ok=stable and max(speeds)<=.05 and max(angular)<=.2 and max(root_speeds)<=.05
    return {'ready':ok,'reason':'stable_pose_window' if ok else 'moving_or_unsupported',
            'max_box_speed_m_s':max(speeds),'max_box_angular_speed_rad_s':max(angular),
            'max_root_planar_speed_m_s':max(root_speeds)}


def loaded_contact_summary(samples):
    """Extrema over executed physics substeps; empty means unavailable."""
    return dict(samples=len(samples),
                minimum_hand_N=min((v for s in samples for v in s['hand_forces_N'].values()),default=None),
                maximum_hand_N=max((v for s in samples for v in s['hand_forces_N'].values()),default=None),
                minimum_total_foot_N=min((sum(s['foot_upward_N'].values()) for s in samples),default=None))
