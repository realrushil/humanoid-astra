"""Temporary front-of-table lower-body limiter; metres, seconds, radians.

This limits commanded translation, not arm motion, and does not certify braking.
The 3 cm margin, 0.1 s lookahead and 0.5 s approach time are uncalibrated test
assumptions. Link orientation includes tilt; navigation rotates by root yaw only.
"""
import itertools,math
MARGIN=.03
LOOKAHEAD=.1
APPROACH_TIME=.5
GUARDED_BODIES=('pelvis','pelvis_contour_link',*[
    side+'_'+joint+'_link' for side in ['left','right']
    for joint in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']])

def rotate_xyzw(q,p):
    x,y,z,w=q;n=math.sqrt(sum(v*v for v in q))
    if n<1e-8 or not math.isfinite(n):raise ValueError('Invalid quaternion')
    x,y,z,w=[v/n for v in [x,y,z,w]]
    R=[[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
       [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
       [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]]
    return [sum(R[i][j]*p[j] for j in range(3)) for i in range(3)]

def front_clearance(geometry,poses):
    # The table's front plane is conservatively extended across all Y. Keeping
    # legs and feet behind it avoids moving sideways into the legs underneath.
    # This is a command limiter, not a guarantee about future gait/body motion.
    fronts=[]
    for name in GUARDED_BODIES:
        if name not in poses or name not in geometry['robot']:
            raise ValueError('Missing lower-body geometry/pose: '+name)
        pose=poses[name]
        for shape in geometry['robot'][name]:
            for p in itertools.product(*zip(shape['min'],shape['max'])):
                fronts.append(pose['pos'][0]+rotate_xyzw(pose['xyzw'],p)[0])
    if not fronts:raise ValueError('No lower-body collision bounds')
    return geometry['source_parts']['source_top']['min'][0]-max(fronts)

def limit_navigation(command,root_wxyz,clearance,closing_speed):
    if not all(math.isfinite(v) for v in [*command,*root_wxyz,clearance,closing_speed]):
        raise ValueError('Nonfinite approach observation')
    w,x,y,z=root_wxyz;yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    c,s=math.cos(yaw),math.sin(yaw)
    wx,wy=c*command[0]-s*command[1],s*command[0]+c*command[1]
    cap=max(0.,(clearance-MARGIN-LOOKAHEAD*max(0.,closing_speed))/APPROACH_TIME)
    safe_wx=min(wx,cap)  # Always preserve requested world-X retreat.
    # Fade lateral/yaw commands when the lower-body front margin is exhausted;
    # preserving them there could drive a knee sideways into a source leg.
    side_scale=min(1.,max(0.,clearance-MARGIN-LOOKAHEAD*max(0.,closing_speed))/MARGIN)
    safe_wy=wy*side_scale
    limited=[c*safe_wx+s*safe_wy,-s*safe_wx+c*safe_wy,command[2]*side_scale]
    return limited,{'clearance_m':clearance,'closing_speed_m_s':closing_speed,
        'forward_world_cap_m_s':cap,'requested_world_vx_m_s':wx,
        'limited_world_vx_m_s':safe_wx,'lateral_yaw_scale':side_scale,
        'intervened':max(abs(a-b) for a,b in zip(limited,command))>1e-7}
