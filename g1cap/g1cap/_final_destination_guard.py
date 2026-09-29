"""Temporary final paired-command screen; no motors or public motion API."""
import math
import numpy as np
from ._source_reference_screen import screen_references
from ._two_table_guard import screen_both

def screen_final_destination_command(world,now_s,proposed,previous,*,stopping=False):
    """Bind both table screens to the current owner's actual 50-value proposal.

    Call after paired wrist control and before float32 action publication.
    The same pickup proof and owner must already be bound by the runtime.
    The destination uses its measured facing side, never the source tabletop.
    Full body/box/reference-path bounds retain the existing declared motion
    assumptions; this calculation does not qualify forward gait or stopping.
    """
    owner=world.scene_wrist_hold;gate=world.control.carry_admission
    if owner is None or gate is None:raise ValueError('destination_carry_owner_unavailable')
    gate.require_owner(owner)
    if getattr(owner,'invalid',True):raise ValueError('destination_carry_owner_invalid')
    stamp=owner.commanded_at;packet=world.sensor_recorder.latest_packet
    if (type(now_s) not in (int,float) or not math.isfinite(now_s)
            or type(stamp) not in (int,float) or not math.isfinite(stamp)
            or abs(stamp-now_s)>1e-8 or packet['time_s']!=now_s):
        raise ValueError('destination_paired_command_stale')
    command=np.asarray(proposed,float)
    if (command.shape!=(50,) or not np.isfinite(command).all()
            or not np.array_equal(command,np.asarray(owner.last_command))):
        raise ValueError('destination_paired_proposal_mismatch')
    if not gate.observe(now_s,world.box_perception.retention_feedback(now_s)):
        raise ValueError(gate.failure or 'carry_pickup_not_verified')
    result=screen_destination_geometry(world,now_s,proposed,previous,stopping=stopping)
    gate.require_owner(world.scene_wrist_hold)
    result.update(paired_command_time_s=stamp,forward_motion_qualified=False,final_paired_command_checked=True)
    return result

def screen_destination_geometry(world,now_s,proposed,previous,*,stopping=False):
    """Both-table pre-pair geometry only; this never certifies final ownership.

    The approach controller calls this before the paired wrist correction.
    screen_final_destination_command MUST run again on the actual final proposal.
    """
    packet=world.sensor_recorder.latest_packet
    if (type(now_s) not in (int,float) or not math.isfinite(now_s)
            or packet['time_s']!=now_s):raise ValueError('destination_geometry_packet_time')
    command=np.asarray(proposed,float)
    if command.shape!=(50,) or not np.isfinite(command).all():
        raise ValueError('destination_geometry_command_invalid')
    from .scene_uncertainty_runtime import navigation_forecast
    forecast=navigation_forecast(world,stopping=stopping)
    stop=world.loaded_stop_clearance;rotation=world.box_perception.motion.rotation
    def evaluate(command):
        return world.loaded_travel_probe.screen(stop,packet,world.sensor_recorder.latest_hands,
            rotation,world.box_perception.carry_seed,command,previous)
    def source(command):
        return screen_references(stop,world.measured_source_references,
            world.box_perception.source_plane,packet,rotation,lambda:evaluate(command),forecast=forecast)
    def destination(command):
        original_plane,original_memory=stop.plane,stop.front_memory
        try:
            stop.plane=None;stop.front_memory=None
            return screen_references(stop,world.measured_destination_references,
                world.destination_observation,packet,rotation,lambda:evaluate(command),forecast=forecast)
        finally:
            stop.plane,stop.front_memory=original_plane,original_memory
    result=screen_both(now_s,command.tolist(),source,destination)
    result.update(forward_motion_qualified=False,final_paired_command_checked=False)
    return result
