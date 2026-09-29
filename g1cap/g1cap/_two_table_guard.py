"""Temporary composition of two current-command obstacle screens."""
import math

def finite(value):
    return type(value) in (int,float) and math.isfinite(value)

def screen_both(now,command,source_screen,destination_screen):
    """Require both complete screens for the identical proposed command.

    Callbacks evaluate current encoder/gyro time, including complete robot,
    observed box, arm-reference path and declared stopping envelope. Older
    measured references retain their own uncertainty/timestamps inside those
    screens; time_s here identifies evaluation time, not observation freshness.
    This function owns no motors and supplies no missing geometry.
    """
    if not finite(now) or now<0 or len(command)!=50 or not all(finite(v) for v in command):
        raise ValueError('two_table_command_invalid')
    results={}
    for name,callback in [('source',source_screen),('destination',destination_screen)]:
        proposed=list(command)
        result=callback(proposed)
        if proposed!=list(command):raise ValueError(name+'_proposal_mutated')
        if not isinstance(result,dict) or result.get('status') not in ('clear','insufficient_margin'):
            raise ValueError(name+'_screen_unavailable')
        if not finite(result.get('time_s')) or abs(result['time_s']-now)>1e-8:
            raise ValueError(name+'_screen_time_mismatch')
        lower=result.get('lower_m')
        if (not finite(lower) or (result['status']=='clear')!=(lower>=.03)):
            raise ValueError(name+'_screen_invalid')
        results[name]=result
    limiting=min(results,key=lambda name:results[name]['lower_m'])
    lower=results[limiting]['lower_m']
    return dict(status='clear' if lower>=.03 else 'insufficient_margin',time_s=now,
        lower_m=lower,required_margin_m=.03,limiting_obstacle=limiting,
        source_only=False,source=results['source'],destination=results['destination'])
