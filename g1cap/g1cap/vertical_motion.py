"""Conditional vertical displacement from measured floor heights and IMU.

Acceleration error includes calibration, gravity projection and all intersample
linear-reconstruction error; it is a declared simulation assumption, not just
sensor noise. No assumption of constant acceleration over an image interval.
"""
import numpy as np

def predict_vertical(now,camera_time,heights,accelerations,*,error=.5,history_s=.5,horizon=.02,maximum_acceleration=20.):
    heights=np.asarray(heights,float);acc=np.asarray(accelerations,float)
    if (heights.ndim!=2 or heights.shape[1]!=3 or acc.ndim!=2 or acc.shape[1]!=2
            or not np.isfinite(heights).all() or not np.isfinite(acc).all()
            or np.any(heights[:,2]<0) or np.any(np.diff(acc[:,0])<=0)
            or not np.isfinite([now,camera_time,error,history_s,horizon,maximum_acceleration]).all()
            or min(error,history_s,horizon,maximum_acceleration)<=0):raise ValueError('invalid vertical evidence')
    age=now-camera_time
    if not 0<=age<=.150001:raise ValueError('vertical_camera_stale')
    current=heights[abs(heights[:,0]-camera_time)<1e-8]
    older=heights[heights[:,0]<=camera_time-history_s+1e-8]
    if len(current)!=1 or not len(older):raise ValueError('need_height_history')
    old=older[-1];current=current[0];t0=old[0];T=camera_time-t0
    if T>history_s+.150001:raise ValueError('height_history_gap')
    if not np.any(abs(acc[:,0]-t0)<1e-8) or not np.any(abs(acc[:,0]-camera_time)<1e-8) or not np.any(abs(acc[:,0]-now)<1e-8):raise ValueError('acceleration_endpoints_missing')
    used=acc[(acc[:,0]>=t0-1e-8)&(acc[:,0]<=now+1e-8)]
    if np.max(np.diff(used[:,0]))>.020001:raise ValueError('acceleration_time_gap')
    def integral(start,end,weight):
        points=used[(used[:,0]>=start-1e-8)&(used[:,0]<=end+1e-8)]
        t,a=points.T;dt=np.diff(t);w=weight(t)
        return float(np.sum(dt/6*((2*w[:-1]+w[1:])*a[:-1]+(w[:-1]+2*w[1:])*a[1:])))
    velocity=(current[1]-old[1])/T+integral(t0,camera_time,lambda t:t-t0)/T
    velocity_error=(current[2]+old[2])/T+error*T/2
    delta=velocity*age+integral(camera_time,now,lambda t:now-t)
    delta_error=age*velocity_error+error*age**2/2
    velocity_now=velocity+integral(camera_time,now,lambda t:np.ones_like(t))
    lower_velocity=velocity_now-velocity_error-error*age
    if abs(used[-1,1])+error>maximum_acceleration:raise ValueError('vertical_acceleration_envelope_exceeded')
    future_down=max(0.,-lower_velocity)*horizon+.5*maximum_acceleration*horizon**2
    return dict(delta_m=delta,error_m=delta_error,lower_delta_m=delta-delta_error,
        lower_velocity_m_s=lower_velocity,future_down_m=future_down,
        downward_allowance_m=max(0.,-(delta-delta_error))+future_down,
        camera_time_s=camera_time,age_s=age,history_s=T,acceleration_reconstruction_error_m_s2=error,
        maximum_future_acceleration_m_s2=maximum_acceleration)
