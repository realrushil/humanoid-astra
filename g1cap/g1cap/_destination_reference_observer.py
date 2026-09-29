"""Temporary camera-time destination edge observer; no navigation permission."""
import json,math
import numpy as np
from g1cap.destination_surface import green_destination_mask
from g1cap.sensor_kinematics import camera_in_body
from ._connected_front import front_candidates
from ._source_references import SourceFrontReferences

def observe_destination_reference(world,visual,packet,camera):
    """Retain measured facing-plane evidence under one destination observer.

    A view epoch change is logged, never equated with obstacle identity. The
    existing reference bank independently checks original-boundary association.
    A movement controller must still stop on its own view-epoch discontinuity.
    All points are camera RGB-D transformed with camera-time robot encoders.
    """
    owner=world.destination_observation
    bank=getattr(world,'measured_destination_references',None)
    if bank is None:
        bank=world.measured_destination_references=SourceFrontReferences(owner,uncertainty=getattr(world,'scene_uncertainty',None))
    if packet['step']!=camera['step'] or abs(packet['time_s']-camera['time_s'])>1e-8:
        bank.invalidate('destination_camera_encoder_mismatch')
        raise ValueError('destination_camera_encoder_mismatch')
    t=packet['time_s'];destination=owner.observe(t);candidates=[]
    if destination['status']=='observed_destination_candidate':
        stamp=destination.get('observed_at_s')
        if type(stamp) not in (int,float) or not math.isfinite(stamp) or abs(stamp-t)>1e-8:
            bank.invalidate('destination_fit_camera_mismatch')
            raise ValueError('destination_fit_camera_mismatch')
        support=dict(destination,status='observed_candidate')
        result=front_candidates(camera['depth_m'],camera['calibration']['intrinsic'],support,
            green_destination_mask(camera['rgb']),
            lambda points:world.approach_geometry.exclude(points,packet,camera['calibration']),min_span=0.)
        C=camera_in_body(world.approach_geometry.model,
            dict(zip(packet['joint_names'],packet['q_rad'],strict=True)),camera['calibration'])
        candidates=[dict(line=line,normal_body=(C[:3,:3]@line['normal_camera']).tolist(),
            anchor_body=(np.mean(line['endpoints_camera_m'],axis=0)@C[:3,:3].T+C[:3,3]).tolist())
            for line in result.get('lines',[])]
    feedback=bank.observe(t,candidates,owner,visual['motion'])
    row=dict(time_s=t,step=packet['step'],destination_status=destination['status'],
        destination_reason=destination.get('reason'),view_epoch=destination.get('view_epoch'),
        candidates=candidates,feedback=feedback)
    if 'destination-references' not in world.files:
        world.files['destination-references']=(world.output/'destination-references.jsonl').open('w')
    world.files['destination-references'].write(json.dumps(row,allow_nan=False)+'\n')
    world.files['destination-references'].flush()
    return row
