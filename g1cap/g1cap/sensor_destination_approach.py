"""Bounded loaded approach driven by a visible RGB-D destination candidate.

All distances are metres and timestamps are simulation seconds.  The callback
must provide a current destination centroid/range, a camera-calibrated planar
direction in the local stance segment, and the same retained-grasp/frame
signals used by loaded carry.  This controller owns only bounded navigation;
the existing paired wrist and whole-body balance owners remain responsible for
the load.
"""

from collections import deque
import math


class SensorDestinationApproach:
    def __init__(self, now_s, distance_m, observe, wrist_ready):
        if (isinstance(distance_m, bool) or not isinstance(distance_m, (int, float)) or
                not math.isfinite(distance_m) or not .20 <= distance_m <= .50):
            raise ValueError('distance_outside_envelope')
        self.observe, self.wrist_ready = observe, wrist_ready
        self.distance = float(distance_m)
        self.started = self.previous = float(now_s)
        self.segment = None
        self.anchor_translation = None
        self.direction = None
        self.phase = 'destination_approach'
        self.speed = 0.0
        self.history = deque(maxlen=51)
        sample = self.sample(now_s)
        self.segment = sample['frame']['segment']
        self.anchor_translation = tuple(sample['frame']['translation_segment_m'])
        self.direction = tuple(sample['destination']['direction_segment_xy'])

    def sample(self, now_s):
        sample = self.observe(now_s)
        try:
            destination, grasp, frame = sample['destination'], sample['grasp'], sample['frame']
        except (KeyError, TypeError):
            raise ValueError('destination_observation_unavailable') from None
        if destination.get('status') != 'observed_destination_candidate':
            raise ValueError('destination_observation_unavailable')
        observed = destination.get('observed_at_s')
        if (not isinstance(observed, (int, float)) or isinstance(observed, bool) or
                not math.isfinite(observed) or not 0 <= now_s - observed <= .150001):
            raise ValueError('destination_observation_stale')
        if (grasp.get('status') != 'available' or not grasp.get('raised') or
                not grasp.get('ready')):
            raise ValueError('destination_grasp_not_ready')
        if (frame.get('status') != 'tracked_local_segment' or
                not 0 <= now_s - frame.get('time_s', float('nan')) <= .150001):
            raise ValueError('destination_motion_unavailable')
        if self.segment is not None and frame.get('segment') != self.segment:
            raise ValueError('destination_motion_unavailable')
        direction = destination.get('direction_segment_xy')
        translation = frame.get('translation_segment_m')
        if (not isinstance(direction, (list, tuple)) or len(direction) != 2 or
                not isinstance(translation, (list, tuple)) or len(translation) != 3 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in [*direction, *translation])):
            raise ValueError('destination_geometry_invalid')
        norm = math.hypot(*direction)
        if not .5 <= norm <= 1.1:
            raise ValueError('destination_direction_invalid')
        return sample

    def _progress(self, sample):
        translation = sample['frame']['translation_segment_m']
        delta = [translation[i] - self.anchor_translation[i] for i in range(3)]
        return sum(delta[i] * self.direction[i] for i in range(2))

    def command(self, now_s, previous):
        sample = self.sample(now_s)
        if now_s < self.previous or now_s - self.previous > .150001:
            raise ValueError('destination_motion_time_gap')
        progress = self._progress(sample)
        remaining = self.distance - progress
        if progress < -.03 or progress > self.distance + .03:
            raise ValueError('destination_approach_overshoot')
        desired = 0.0 if remaining <= .02 else .08
        dt = min(now_s - self.previous, .02)
        self.speed = max(0.0, min(desired, self.speed + .30 * dt))
        self.previous = now_s
        action = list(previous)
        action[43:46] = [self.speed * self.direction[0], self.speed * self.direction[1], 0.0]
        self.history.append((now_s, progress))
        self.phase = 'destination_settle' if desired == 0.0 else 'destination_approach'
        return action

    def update(self, now_s):
        sample = self.sample(now_s)
        progress = self._progress(sample)
        self.history.append((now_s, progress))
        if now_s - self.started > 12:
            raise ValueError('destination_approach_timeout')
        if progress < -.03 or progress > self.distance + .03:
            raise ValueError('destination_approach_overshoot')
        if progress >= self.distance - .02:
            self.phase = 'destination_settle'
            if now_s - self.started >= 1.0 and sample['grasp']['ready']:
                return 'completed', 'destination_approach_and_hold'
        return None

    def measurements(self, now_s):
        sample = self.sample(now_s)
        return dict(time_s=now_s, progress_m=self._progress(sample), target_m=self.distance,
                    direction_segment_xy=list(self.direction), phase=self.phase)
