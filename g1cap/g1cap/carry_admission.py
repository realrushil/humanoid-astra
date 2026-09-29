"""Episode-local pickup proof and owner identity; no motion or clearance grant."""
import math

class CarryAdmission:
    def __init__(self):
        self.reset()

    def reset(self):
        self.identity=None;self.pickup_time=None;self.last_time=None
        self.owner=None;self.failure=None

    def fail(self,reason):
        self.failure=reason
        return False

    @staticmethod
    def identity_at(now,sample):
        stamp=sample.get('time_s')
        if (type(now) not in (int,float) or not math.isfinite(now) or now<0
                or type(stamp) not in (int,float) or not math.isfinite(stamp)
                or not 0<=now-stamp<=.150001 or sample.get('status')!='available'
                or sample.get('retained') is not True):
            raise ValueError('carry_retention_unavailable')
        identity=(sample.get('track_epoch'),sample.get('segment'))
        if any(type(v) is not int or v<0 for v in identity):
            raise ValueError('carry_identity_unavailable')
        return identity

    def record_pickup(self,now,sample):
        """Call only on actual completed pickup, never on a hold or observer flag."""
        self.reset()
        try:identity=self.identity_at(now,sample)
        except ValueError as error:return self.fail(str(error))
        if sample.get('settled') is not True:return self.fail('carry_pickup_not_settled')
        self.identity=identity;self.pickup_time=self.last_time=now
        return True

    def observe(self,now,sample):
        """Monitor through idle/wait as well as motion; a loss latches."""
        if self.failure or self.identity is None:return False
        try:identity=self.identity_at(now,sample)
        except ValueError as error:return self.fail(str(error))
        if not 0<=now-self.last_time<=.150001:return self.fail('carry_monitor_time_gap')
        self.last_time=now
        if identity!=self.identity:return self.fail('carry_identity_changed')
        return True

    def arm(self,now,sample,owner):
        """Bind the active floor-frame owner after the initial lift handoff.

        The caller must separately validate that this is the current paired
        floor-frame owner and screen every proposed command and stopping path.
        """
        if not self.observe(now,sample):raise ValueError(self.failure or 'carry_pickup_not_verified')
        if sample.get('settled') is not True:raise ValueError('carry_not_settled')
        if self.owner is not None:self.require_owner(owner)
        if owner is None:raise ValueError('carry_owner_unavailable')
        self.owner=owner

    def require_owner(self,owner):
        if self.failure:raise ValueError(self.failure)
        if self.owner is None:raise ValueError('carry_not_armed')
        if owner is not self.owner:
            self.fail('carry_owner_changed')
            raise ValueError(self.failure)
