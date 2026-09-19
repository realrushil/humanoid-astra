"""Small delayed estimated-state adapter for nominal sensitivity experiments.

Errors are synthetic assumptions, not hardware calibration.  Position and wrist
coordinates use the simulation's XY frame; yaw is radians and wrapped to [-pi, pi).
"""
from dataclasses import dataclass, replace
import math
import random
from collections import deque

from .models import State, wrap_yaw


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TypeError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class ObservationConfig:
    delay_s: float = 0.0
    position_bias_xy: tuple = (0.0, 0.0)
    yaw_bias: float = 0.0
    position_noise_std: float = 0.0
    yaw_noise_std: float = 0.0
    seed: int = 0

    def __post_init__(self):
        delay_s = _number(self.delay_s, "delay_s")
        if not 0 <= delay_s <= .5:
            raise ValueError("delay_s must be in [0, .5]")
        if not isinstance(self.position_bias_xy, (tuple, list)) or len(self.position_bias_xy) != 2:
            raise ValueError("position_bias_xy must contain two values")
        for value in self.position_bias_xy:
            _number(value, "position_bias_xy")
        _number(self.yaw_bias, "yaw_bias")
        _number(self.position_noise_std, "position_noise_std")
        _number(self.yaw_noise_std, "yaw_noise_std")
        if not 0 <= self.position_noise_std <= .1:
            raise ValueError("invalid position noise")
        if not 0 <= self.yaw_noise_std <= .2:
            raise ValueError("invalid yaw noise")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be int")
        object.__setattr__(self, "position_bias_xy", tuple(self.position_bias_xy))

    @classmethod
    def from_dict(cls, values):
        return cls(**dict(values))


class ObservationStream:
    metadata_label = "assumed_synthetic_observation_error"

    def __init__(self, config=None, history_size=128):
        self.config = config or ObservationConfig()
        if not isinstance(self.config, ObservationConfig):
            raise TypeError('config must be ObservationConfig')
        if isinstance(history_size, bool) or not isinstance(history_size, int) or not 1 <= history_size <= 12000:
            raise ValueError('history_size must be an integer in [1, 12000]')
        self._samples = deque(maxlen=history_size)
        self._rng = random.Random(self.config.seed)
        self._last = None

    def push(self, state):
        if not isinstance(state, State):
            raise TypeError("state must be State")
        if self._last is not None:
            if state == self._last:
                return
            if state.sim_time <= self._last.sim_time:
                raise ValueError("timestamps must increase")
            if state.sequence <= self._last.sequence:
                raise ValueError("sequences must increase")
        px, py = (self._rng.gauss(0, self.config.position_noise_std) + b
                  for b in self.config.position_bias_xy)
        yaw = self._rng.gauss(0, self.config.yaw_noise_std) + self.config.yaw_bias
        p = tuple(state.pelvis_position)
        rw = tuple(state.right_wrist_position)
        lw = tuple(state.left_wrist_position)
        captured = replace(state, pelvis_position=(p[0]+px, p[1]+py, p[2]),
                           right_wrist_position=(rw[0]+px, rw[1]+py, rw[2]),
                           left_wrist_position=(lw[0]+px, lw[1]+py, lw[2]),
                           pelvis_yaw=wrap_yaw(state.pelvis_yaw+yaw))
        self._samples.append(captured)
        self._last = state

    def read(self, current_state):
        if not isinstance(current_state, State):
            raise TypeError("state must be State")
        cutoff = current_state.sim_time - self.config.delay_s
        sample = next((s for s in reversed(self._samples) if s.sim_time <= cutoff), None)
        if sample is None:
            return replace(current_state, ready=False)
        # Synthetic delay uses simulation time. Retain the larger measured
        # transport age, without adding the same transport staleness twice.
        age = current_state.sim_time - sample.sim_time + max(sample.state_age_s, current_state.state_age_s)
        return replace(sample, ready=sample.ready and current_state.ready,
                       state_age_s=age,
                       backend_ok=sample.backend_ok and current_state.backend_ok)
