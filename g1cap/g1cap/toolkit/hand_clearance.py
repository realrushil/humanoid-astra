"""Measured hand-clearance bias; world metres and simulation seconds.

This is an uncalibrated anticipatory reference adjustment, not a safety proof.
Only measured hand/wrist bounds near a tabletop determine the requested bias.
"""
from collections import deque
import math
from g1cap.arena_surfaces import transformed_bounds


class ClearanceBias:
    def __init__(self, geometry):
        self.geometry = geometry
        self.history = deque(maxlen=6)  # 0.1s at the existing 50 Hz action rate.
        self.bias = 0.
        self.time = None

    def update(self, observation):
        margins = []
        for name, shapes in self.geometry['robot'].items():
            if 'hand' not in name and 'wrist' not in name:
                continue
            for shape in shapes:
                bounds = transformed_bounds(shape, observation['body_poses'][name])
                for surface in observation['surfaces'].values():
                    table = surface['bounds']
                    if all(bounds['max'][i] >= table['min'][i]-.03 and
                           bounds['min'][i] <= table['max'][i]+.03 for i in range(2)):
                        margins.append(bounds['min'][2]-table['max'][2])
        gap = min(margins) if margins else None
        return self.update_gap(observation['time'], gap)

    def update_gap(self, now, gap):
        """Shared correction law; caller owns the clearance measurement source."""
        dt = .02 if self.time is None else now-self.time
        if not math.isfinite(now) or not 0 < dt <= .021:
            raise ValueError('Clearance bias requires consecutive 50 Hz observations')
        self.time = now
        closing = 0.
        if gap is None:
            self.history.clear()
            requested = 0.
        else:
            if not math.isfinite(gap):
                raise ValueError('Nonfinite hand geometry')
            self.history.append((now,gap))
            start, previous = self.history[0]
            if now > start:
                closing = max(0., (previous-gap)/(now-start))
            requested = max(0., min(.06, .015-gap+.12*closing))
        # Upward correction is bounded; decay is slower to avoid immediately
        # undoing the extra reference while the measured motion is settling.
        self.bias += max(-.02*dt, min(.20*dt, requested-self.bias))
        return dict(time_s=now, margin_m=gap, closing_speed_m_s=closing,
                    requested_bias_m=requested, bias_m=self.bias)
