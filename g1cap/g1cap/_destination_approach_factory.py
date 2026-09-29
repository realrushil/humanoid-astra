"""Temporary retained-carry factory; caller must wire final-command/stop routing."""
from pathlib import Path
from ._retained_destination import SensorDestinationApproach
from ._final_destination_guard import screen_destination_geometry
from ._loaded_travel_probe import SourceTurnProbe

def make_destination_approach(world,now_s,distance_m):
    """Build a bounded controller without resetting proof, owner or observations.

    This factory does not install a public tool or publish a motor command.
    The eventual runtime caller must run the existing final paired-command guard
    and combined stopping callback; preliminary geometry alone is insufficient.
    """
    def wrist_ready():
        return (world.scene_wrist_hold is not None and world.scene_wrist_failure is None
                and world.floor_hold_active and not world.floor_hold_pending)
    if not wrist_ready():raise ValueError('destination_wrist_owner_unavailable')
    if getattr(world,'destination_travel_failure',None):
        raise ValueError(world.destination_travel_failure)
    gate=world.control.carry_admission
    if gate is None:raise ValueError('carry_pickup_not_verified')
    gate.require_owner(world.scene_wrist_hold)
    if not hasattr(world,'loaded_travel_probe'):
        import g1cap
        assets=Path(g1cap.__file__).parent/'assets'
        world.loaded_travel_probe=SourceTurnProbe(assets/'arena_g1_rev1_0_kinematics.urdf',
            assets/'arena_g1_rev1_0_bounds.json',world.names)
    def preliminary(now,sample,proposed):
        gate.require_owner(world.scene_wrist_hold)
        if world.loaded_stop_evidence_error:raise ValueError(world.loaded_stop_evidence_error)
        result=screen_destination_geometry(world,now,proposed,world.control.last_action)
        return dict(result,observed_at_s=sample['destination']['observed_at_s'])
    return SensorDestinationApproach(now_s,distance_m,world.sensor_destination_observation,wrist_ready,
        screen=preliminary,admission=gate,owner=lambda:world.scene_wrist_hold)
