"""Current straight-retreat certificate, within the declared source-only workspace."""
import numpy as np
from .scene_motion import _rigid
from .source_stop import SourceSides


def tableward_navigation(source,now,rotation,up_body):
    """Measured segment-plane direction in native horizontal forward/left axes.

    Memory normals point toward free space. Negate to retain SourceDeparture's
    tableward-normal convention; that controller commands the negative normal.
    Camera translation stays at its measured time. Only orientation propagates
    through the same IMU rotation used by the source clearance screen.
    """
    if not isinstance(source,SourceSides):raise ValueError('departure_source_direction_unavailable')
    memory=source.memory;motion=source.motion
    if (type(now) not in (int,float) or not np.isfinite(now) or memory.invalid
            or now<memory.time
            or motion.get('status')!='tracked_local_segment'
            or motion.get('segment')!=memory.segment
            or not 0<=now-motion.get('time_s',float('nan'))<=.150001):
        raise ValueError('departure_source_direction_unavailable')
    memory.pose_uncertainty(now)
    pose=_rigid(motion['body_in_segment']);current=np.eye(4);current[:3,:3]=rotation;_rigid(current)
    camera=np.eye(4);camera[:3,:3]=source.gyro;_rigid(camera)
    body_in_segment=pose[:3,:3]@camera[:3,:3].T@current[:3,:3]
    normal=-body_in_segment.T@memory.normal
    up=np.asarray(up_body,float)
    if up.shape!=(3,) or not np.isfinite(up).all() or abs(np.linalg.norm(up)-1)>1e-6:
        raise ValueError('departure_floor_direction_unavailable')
    forward=np.array([1.,0.,0.]);forward-=up*(forward@up)
    if np.linalg.norm(forward)<.5:raise ValueError('departure_floor_direction_unavailable')
    forward/=np.linalg.norm(forward);left=np.cross(up,forward)
    direction=np.array([normal@forward,normal@left])
    if not np.isfinite(direction).all() or np.linalg.norm(direction)<.5:
        raise ValueError('departure_source_direction_unavailable')
    return (direction/np.linalg.norm(direction)).tolist()


def screen_travel(world,packet,probe,proposed,previous):
    """Screen actual paired arms at the maximum 0.08 m/s retreat command.

    The reused travel envelope bounds any planar direction up to that speed;
    the state machine may emit a smaller ramped speed or zero after this check.
    This certifies source geometry only, not unseen rear free space.
    """
    stop=world.loaded_stop_clearance;rotation=world.box_perception.motion.rotation
    def evaluate():
        if stop.up_world is None:raise ValueError('departure_floor_direction_unavailable')
        normal=tableward_navigation(stop.plane,packet['time_s'],rotation,
            np.asarray(rotation).T@stop.up_world)
        maximum=list(proposed);maximum[43:46]=[-.08*normal[0],-.08*normal[1],0.]
        result=probe.screen(stop,packet,world.sensor_recorder.latest_hands,rotation,
            world.box_perception.carry_seed,maximum,previous)
        return dict(result,front_normal_xy=normal,direction_basis='admitted_measured_source_front')
    return world.screen_measured_source(packet,evaluate)
