"""Temporary whole-screen selection for turn and supported-stop guards."""
import math
from g1cap.source_stop import SourceSides
from g1cap._prepared_source_sides import prepare

def screen_references(stop,bank,owner,packet,rotation,evaluate,*,allow_top=False,forecast=None):
    """Stop at the first complete passing independently aged source plane.

    Candidates retain their deterministic bank order. A complete passing
    screen is sufficient; reported margins need not be the largest available.
    Rejections evaluate all choices, preserving the best complete rejection.
    The callback screens one complete proposed command at current encoder/gyro
    time. It must not publish a command. Only supported stopping may fall back
    to a fresh top alone; its ordinary guard still enforces top freshness.
    SourceSides.time keeps the camera/motion timestamp used by the vertical
    predictor; current evaluation time must not invent an intermediate height.
    Original guard state is restored even on rejection, preserving the one
    wrist owner and the camera's original measurement objects.
    """
    now=packet['time_s']
    memories=bank.available(now,owner,stop.motion or {})
    original_plane,original_memory=stop.plane,stop.front_memory
    top=original_plane.top if isinstance(original_plane,SourceSides) else original_plane
    choices=list(memories)
    if allow_top and top is not None:choices.append(None)
    if not choices:raise ValueError('source_reference_unavailable')
    results=[]
    try:
        for memory in choices:
            future=None
            if forecast is not None:
                if memory is None or memory.uncertainty is None:
                    raise ValueError('source forecast requires a motion ledger')
                try:future=memory.uncertainty.forecast(memory.capture,now,**forecast)
                except ValueError as error:
                    if str(error)=='scene uncertainty future budget exhausted':continue
                    raise
            stop.front_memory=memory
            stop.plane=(SourceSides(top,memory,stop.motion,stop.front_gyro,stop.motion['time_s'],pose_errors=future)
                        if memory is not None else top)
            if memory is not None:stop.plane=prepare(stop.plane,now,rotation)
            result=dict(evaluate())
            lower=result.get('lower_m')
            if (not isinstance(lower,(int,float)) or not math.isfinite(lower)
                    or result.get('status') not in ('clear','insufficient_margin')):
                raise ValueError('source_screen_invalid')
            if future is not None:result['source_uncertainty_forecast']=future
            result['source_reference']='measured_front_with_optional_top' if memory is not None else 'fresh_top'
            if memory is not None:
                result.update(source_observed_at_s=memory.time,source_age_s=now-memory.time)
            results.append(result)
            if result['status']=='clear':break
    finally:
        stop.plane,stop.front_memory=original_plane,original_memory
    # Each result already contains its worst body part and complete arm path.
    # Never choose favorable parts from different reference screens.
    if not results:raise ValueError('source_future_budget_exhausted')
    chosen=dict(max(results,key=lambda r:r['lower_m']))
    chosen['source_reference_candidates']=[{k:r[k] for k in
        ('source_reference','source_observed_at_s','lower_m','status') if k in r} for r in results]
    chosen['source_reference_selection']='first_complete_clear_else_best_rejection'
    chosen['source_reference_available_count']=len(choices)
    chosen['source_reference_evaluated_count']=len(results)
    return chosen
