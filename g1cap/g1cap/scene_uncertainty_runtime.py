"""One episode ledger shared by both task-table observers and stopped holding."""
def update_scene_uncertainty(world,motion):
    ledger=getattr(world,'scene_uncertainty',None)
    if ledger is None:return None
    world.loaded_stop_clearance.uncertainty=ledger
    # Before localization starts there is no historical frame to preserve.
    # Once started, a gap or segment change latches failure; never reinitialize.
    if ledger.time is None and motion.get('status')!='tracked_local_segment':
        return dict(status='uninitialized')
    try:
        ledger.observe(motion['time_s'],motion)
        return dict(status='available',time_s=ledger.time,segment=ledger.segment,
                    distance_m=ledger.distance,rotation_rad=ledger.angle)
    except (ValueError,KeyError) as error:
        return dict(status='unavailable',reason=str(error))


def configure_scene_uncertainty(recipe):
    from .scene_uncertainty import SceneUncertainty,validate_forecast
    config=recipe.get('scene_uncertainty');forecast=recipe.get('scene_uncertainty_forecast')
    if config is None:
        if forecast is not None:raise ValueError('forecast requires scene uncertainty')
        return None,None
    if not isinstance(forecast,dict):raise ValueError('scene uncertainty requires explicit forecast')
    return SceneUncertainty(**config),validate_forecast(**forecast)


def navigation_forecast(world,*,stopping=False):
    if stopping or getattr(world,'scene_uncertainty',None) is None:return None
    forecast=getattr(world,'scene_uncertainty_forecast',None)
    if forecast is None:raise ValueError('scene uncertainty forecast unavailable')
    return forecast
