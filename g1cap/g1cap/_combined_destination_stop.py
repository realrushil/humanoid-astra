"""Temporary zero-navigation full-geometry plus paired-arm stopping screen."""
import numpy as np
from ._final_destination_guard import screen_final_destination_command
from ._source_reference_screen import screen_references

def screen_destination_stop(world,now_s,proposed,previous):
    """Conjoin both full-body/box envelopes with both vertical-aware arm paths.

    Uses the unchanged conservative travel envelope for whole-body/box motion;
    the existing floor/IMU predictor adds the explicit paired-arm stop check.
    A failure of either obstacle remains a failure; no source-only fallback.
    This is an offline prototype and does not qualify physical stopping.
    """
    command=np.asarray(proposed,float)
    if command.shape!=(50,) or not np.isfinite(command).all() or np.any(abs(command[43:46])>1e-12):
        raise ValueError('destination_stop_requires_zero_navigation')
    result=screen_final_destination_command(world,now_s,proposed,previous,stopping=True)
    stop=world.loaded_stop_clearance;packet=world.sensor_recorder.latest_packet
    rotation=world.box_perception.motion.rotation
    def evaluate():return stop.screen(packet,world.sensor_recorder.latest_hands,rotation,proposed)
    arms={}
    arms['source']=screen_references(stop,world.measured_source_references,
        world.box_perception.source_plane,packet,rotation,evaluate,allow_top=True)
    original_plane,original_memory=stop.plane,stop.front_memory
    try:
        stop.plane=None;stop.front_memory=None
        arms['destination']=screen_references(stop,world.measured_destination_references,
            world.destination_observation,packet,rotation,evaluate)
    finally:stop.plane,stop.front_memory=original_plane,original_memory
    components={'full_body_box_and_path':result['lower_m'],**{k+'_stopped_arm_path':v['lower_m'] for k,v in arms.items()}}
    lower=min(components.values())
    return dict(status='clear' if lower>=.03 else 'insufficient_margin',time_s=now_s,lower_m=lower,
        required_margin_m=.03,limiting_component=min(components,key=components.get),
        full_geometry=result,stopped_arm_paths=arms,physical_stopping_qualified=False)
