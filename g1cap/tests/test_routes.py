import dataclasses
import unittest

from g1cap.models import Evaluator, State
from g1cap.tasks import make_task


class RouteTests(unittest.TestCase):
    def route(self):
        return dataclasses.replace(make_task(), name='route', deadline=30.,
                                   target_position=(.5, .5, 0.), target_yaw=.5,
                                   waypoints=((.5, 0.), (.5, .5)))

    def sample(self, time, xy, yaw=0.):
        return State(sim_time=time, pelvis_position=(*xy, .75), pelvis_yaw=yaw)

    def test_final_target_cannot_skip_first_waypoint(self):
        evaluator = Evaluator(self.route())
        for time in (0., .5, 1.):
            self.assertIsNone(evaluator.update(self.sample(time, (.5, .5), .5)))
        self.assertEqual(evaluator.metrics['waypoints_completed'], 0)

    def test_waypoints_need_ordered_dwell_and_final_heading(self):
        evaluator = Evaluator(self.route())
        for time in (0., .5):
            self.assertIsNone(evaluator.update(self.sample(time, (.5, 0.))))
        self.assertEqual(evaluator.metrics['waypoints_completed'], 1)
        for time in (.6, 1.1):
            self.assertIsNone(evaluator.update(self.sample(time, (.5, .5))))
        self.assertIsNone(evaluator.update(self.sample(1.2, (.5, .5), .5)))
        self.assertEqual(evaluator.update(self.sample(1.7, (.5, .5), .5)), 'success')
        self.assertEqual(evaluator.metrics['waypoints_completed'], 2)

    def test_task_roundtrip_preserves_explicit_targets(self):
        from g1cap.tasks import task_from_dict
        task = self.route()
        self.assertEqual(task_from_dict(task.to_dict()), task)

    def test_route_still_fails_when_leaving_workspace(self):
        evaluator = Evaluator(self.route())
        self.assertEqual(evaluator.update(self.sample(0., (2.1, 0.))), 'left_workspace')
