"""A partial lift followed by sustained loss ends that acquisition attempt.

Only camera-time grasp hypotheses enter here. A 2 cm observed lift on two
images marks an attempted grasp, not success. Subsequently being within 5 mm
of the fresh observed under-box/source-height plane with no opposed wrists for 0.3 s is a failure
signal. These ideal-sensor thresholds are simulation assumptions, not hardware
calibration or proof of support/contact. Pregrasp navigation is not prohibited.
"""
import math


class AcquisitionRetention:
    def __init__(self):
        self.last_time=None
        self.lift_count=0
        self.lift_seen=False
        self.loss_started=None
        self.reference=None

    def update(self,now_s,grasp):
        t,gap=grasp.get('time_s'),grasp.get('gap_m')
        reference=grasp.get('gap_reference')
        if reference!=self.reference:
            # A different height reference must earn its own partial-lift
            # evidence. Neither old lift samples nor loss dwell transfer.
            self.lift_count=0;self.lift_seen=False;self.loss_started=None
            self.reference=reference
        valid=(grasp.get('status')=='available' and
               reference in ('observed_under_box_plane','source_height_plane') and
               all(type(v) in (int,float) and math.isfinite(v) for v in (now_s,t,gap)) and
               0<=now_s-t<=.150001)
        if not valid:
            self.lift_count=0;self.loss_started=None
            return None
        if self.last_time is not None:
            if t==self.last_time:return None  # A repeated image earns no dwell.
            if t<self.last_time or t-self.last_time>.150001:
                self.lift_count=0;self.loss_started=None
            if t<self.last_time:return None
        self.last_time=t
        opposed=grasp.get('opposing_near_wrists')
        self.lift_count=self.lift_count+1 if opposed is True and gap>=.02 else 0
        self.lift_seen=self.lift_seen or self.lift_count>=2
        if self.lift_seen and opposed is False and gap<=.005:
            if self.loss_started is None:self.loss_started=t
            if t-self.loss_started>=.3-1e-9:return 'acquisition_grasp_attempt_lost'
        else:self.loss_started=None
        return None
