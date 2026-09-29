"""Sensor-only Arena wiring for bounded source departure; no evaluator input."""
from pathlib import Path
import json
from .source_departure_adapter import SensorDeparture
from .departure_probe import SourceTurnProbe
from ._loaded_travel_probe import SourceTurnProbe as TravelProbe

def screen_departure_sweep(world,packet,evaluate,*,required_margin_m=.03):
    """Require remembered-front clearance even if the top leaves camera view.

    Temporarily remove the fresh top alternative only for full-yaw admission.
    Restore it for the shared stopped-arm controller, including on failure.
    The selector still applies original plane ages and uncertainty.
    """
    stop=world.loaded_stop_clearance
    original=stop.plane
    try:
        stop.plane=None
        def planning_screen():
            row=dict(evaluate())
            if row['status']=='clear' and row['lower_m']<required_margin_m:
                row['status']='insufficient_margin'
            return row
        result=world.screen_measured_source(packet,planning_screen)
        return dict(result,source_clearance_basis='remembered_front_only')
    finally:
        stop.plane=original

class RuntimeDeparture(SensorDeparture):
    def __init__(self,world,now,yaw,initial):
        self.world=world;self.evidence=None
        assets=Path(__file__).parent/'assets'
        self.probe=SourceTurnProbe(assets/'arena_g1_rev1_0_kinematics.urdf',assets/'arena_g1_rev1_0_bounds.json',world.names)
        self.travel_probe=TravelProbe(assets/'arena_g1_rev1_0_kinematics.urdf',assets/'arena_g1_rev1_0_bounds.json',world.names)
        # Additional 5 cm development reserve rounds up twice the measured
        # 2.14 cm handoff loss. Runtime collision threshold remains 3 cm.
        margin=.08 if getattr(world,'scene_uncertainty',None) is not None else .03
        # Static sensitivity needs about 0.2 m extra retreat (>=2.5 s at
        # the unchanged 0.08 m/s cap); allow bounded settling headroom.
        duration=16. if getattr(world,'scene_uncertainty',None) is not None else 12.
        super().__init__(now,yaw,initial,self.measure,planning_margin_m=margin,max_duration_s=duration)
    def measure(self,now,proposed,previous):
        w=self.world;packet=w.sensor_recorder.latest_packet
        if abs(packet['time_s']-now)>1e-8:raise ValueError('departure_observation_clock_invalid')
        if w.loaded_stop_evidence_error:raise ValueError(w.loaded_stop_evidence_error)
        retained=w.control.carry_feedback(now)
        if retained['status']!='available':raise ValueError('departure_retention_discontinuity')
        frame=w.box_perception.control_frame(now)
        if frame['segment']!=retained['segment']:raise ValueError('departure_retention_discontinuity')
        # Supported-arm stopping remains mandatory. Translation additionally
        # screens body, measured fingers, box and the proposed paired-arm path.
        # Source-only controlled-workspace scope does not certify rear space.
        stopped=w.screen_measured_source(packet,lambda:w.loaded_stop_clearance.screen(
            packet,w.sensor_recorder.latest_hands,w.box_perception.motion.rotation,proposed),stopping=True)
        if stopped['status']!='clear':raise ValueError('coupled_stop_path_margin_insufficient')
        sweep=screen_departure_sweep(w,packet,lambda:self.probe.screen(
            w.loaded_stop_clearance,packet,w.sensor_recorder.latest_hands,
            w.box_perception.motion.rotation,w.box_perception.carry_seed,proposed,previous,
            yaw_rad=self.controller.yaw),required_margin_m=self.controller.planning_margin)
        front=w.approach.feedback(now);hand=w.observed_hand.feedback(now)
        live=front['status']=='available' and hand['status']=='available'
        if live:
            # Preserve the already-qualified visible-source retreat path.
            travel=dict(status='legacy_live_checks',front_normal_xy=front.get('normal_navigation_xy',[]),
                direction_basis='live_source_front')
        else:
            from .departure_travel import screen_travel
            travel=screen_travel(w,packet,self.travel_probe,proposed,previous)
        capture=retained['time_s']
        if abs(sweep['observed_at_s']-capture)>1e-8:raise ValueError('departure_observation_clock_invalid')
        self.evidence=dict(time_s=capture,track_epoch=retained['track_epoch'],segment=retained['segment'],
            retained=retained['retained'],settled=retained['settled'],
            direction_available=True,front_normal_xy=travel['front_normal_xy'],
            direction_basis=travel['direction_basis'],drive_clear=live or travel['status']=='clear',
            live_front_available=front['status']=='available',live_hand_available=hand['status']=='available',
            sweep=dict(status=sweep['status'],lower_m=sweep['lower_m'],time_s=capture,yaw_rad=self.controller.yaw),
            source_observed_at_s=sweep.get('source_observed_at_s'))
        w.files['source-departure'].write(json.dumps(dict(time_s=now,sweep=sweep,stop=stopped,travel=travel,evidence=self.evidence),allow_nan=False)+'\n')
        w.files['source-departure'].flush()
        return self.evidence
    def screened_command(self,now,proposed,previous):
        result=super().screened_command(now,proposed,previous)
        if self.latest['outcome']=='completed':
            # Future hold/turn must continue screening source memory; never
            # return to a live-edge-only guard after the edge has left view.
            self.world.control.source_turn_geometry=self.world.probe_source_geometry
        elif self.latest['outcome']=='failed':
            result=self.world.loaded_stop_command(now,result)
            result[43:46]=[0.,0.,0.]
        return result

def begin(world,now,previous,yaw):
    if (world.scene_wrist_hold is None or world.scene_wrist_failure is not None
            or not world.floor_hold_active or world.floor_hold_pending):
        raise ValueError('turn_wrist_owner_unavailable')
    retained=world.control.carry_feedback(now)
    world.control.carry_admission.arm(now,retained,world.scene_wrist_hold)
    initial=dict(time_s=retained['time_s'],track_epoch=retained['track_epoch'],segment=retained['segment'],retained=retained['retained'])
    from .departure_reference import initialize_reference
    initialize_reference(world,now)
    world.box_perception.begin_carry(now)
    if 'source-departure' not in world.files:
        world.files['source-departure']=(world.output/'source-departure.jsonl').open('w')
    world.files['source-departure'].write(json.dumps(dict(type='stage_reference',time_s=now,evidence=world.departure_reference_admission),allow_nan=False)+'\n')
    world.files['source-departure'].flush()
    controller=RuntimeDeparture(world,now,yaw,initial)
    world.departure_controller=controller
    return controller

def public_evidence(world,now):
    c=getattr(world,'departure_controller',None)
    if c is None or c.evidence is None:return dict(status='unavailable')
    e=c.evidence;age=now-e['time_s']
    if not 0<=age<=.150001:return dict(status='unavailable',reason='departure_evidence_stale')
    state=c.controller
    return dict(status='ready' if state.outcome=='completed' else 'not_ready',time_s=e['time_s'],age_s=age,
        yaw_rad=state.yaw,clearance_lower_m=e['sweep']['lower_m'],required_margin_m=.03,
        planning_margin_m=state.planning_margin,
        track_epoch=e['track_epoch'],segment=e['segment'],retained=e['retained'],settled=e['settled'],
        evidence_start_s=state.ready_since,source_observed_at_s=e['source_observed_at_s'])
