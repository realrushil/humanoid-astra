"""One fresh source reference at departure admission, not a tracker retry/reset."""
import math
from ._source_references import SourceFrontReferences

def initialize_reference(world,now):
    existing=getattr(world,'departure_reference',None)
    if existing is not None:return existing
    data=getattr(world,'departure_source_input',None)
    owner=world.box_perception.source_plane
    if data is None or owner is None or data['owner'] is not owner or owner.plane is None:
        raise ValueError('departure_source_identity_unavailable')
    stamp=data['time_s']
    if (type(now) not in (int,float) or not math.isfinite(now)
            or type(stamp) not in (int,float) or not math.isfinite(stamp)
            or not 0<=now-stamp<=.050001):
        raise ValueError('departure_source_frame_stale')
    bank=SourceFrontReferences(owner,uncertainty=getattr(world,'scene_uncertainty',None))
    result=bank.observe(stamp,data['candidates'],owner,data['motion'])
    if result['status']!='observed_source_front':raise ValueError('departure_source_reference_unavailable')
    # This admission creates a local reference after verified pickup/hold. Keep
    # its capture time and the shared episode uncertainty ledger. Later requests reuse this
    # object even after failure; they cannot manufacture a fresh lifetime.
    world.departure_reference=bank
    world.measured_source_references=bank
    world.departure_reference_admission=result
    return bank
