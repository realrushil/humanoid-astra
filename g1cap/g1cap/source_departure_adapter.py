"""Departure adapter: decide navigation only after paired wrist references exist."""
from .source_departure import SourceDeparture

class SensorDeparture:
    post_wrist_required=True
    def __init__(self,now,yaw,sample,observe,*,planning_margin_m=.03,max_duration_s=12.):
        self.controller=SourceDeparture(now,yaw,sample,planning_margin_m=planning_margin_m,max_duration_s=max_duration_s)
        self.observe=observe;self.phase='source_departure';self.screened_at=None
        self.latest=None
    def command(self,now,previous):
        # The wrist owner runs next. No navigation is emitted until its proposed
        # arm references have been screened by screened_command at this timestamp.
        result=list(previous);result[43:46]=[0.,0.,0.]
        return result
    def screened_command(self,now,proposed,previous):
        result=list(proposed);result[43:46]=[0.,0.,0.]
        try:
            sample=self.observe(now,result,previous)
            self.latest=self.controller.update(now,sample)
        except ValueError as error:
            result=list(previous);result[43:46]=[0.,0.,0.]
            self.controller._finish(str(error));self.latest=self.controller.result()
        self.screened_at=now
        self.phase='source_departure' if self.latest['phase']=='drive' else 'source_departure_settle'
        result[43:46]=self.latest['navigation']
        return result
    def update(self,now):
        if self.screened_at is None or not 0<=now-self.screened_at<=.050001:
            raise ValueError('departure_post_arm_screen_missing')
        if self.latest['outcome']:
            return self.latest['outcome'],self.latest['reason']
        return None
    def measurements(self,now):
        return dict(self.latest or {},screened_at_s=self.screened_at)
