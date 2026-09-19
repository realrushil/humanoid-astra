"""Software contracts only: these tests do not measure a humanoid controller."""
import dataclasses
import json
import math
import unittest

from g1cap.models import Evaluator, State
from g1cap.tasks import make_task
from g1cap.sonic import planner_command


class TaskAndEvaluatorTests(unittest.TestCase):
    def test_seeded_immutable_instance_and_detached_json(self):
        task = make_task('waypoint', 4)
        self.assertEqual(task, make_task('waypoint', 4))
        self.assertNotEqual(task.target_position, make_task('waypoint', 5).target_position)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            task.seed = 8
        copy = task.to_dict()
        copy['target_position'][0] = 900
        self.assertLess(task.target_position[0], 1.01)
        json.dumps(task.to_dict(), allow_nan=False)
        with self.assertRaises(ValueError):
            make_task('missing')

    def test_waypoint_requires_continuous_settled_dwell(self):
        task = dataclasses.replace(make_task(), target_position=(0., 0., 0.), target_yaw=0.)
        ev = Evaluator(task)
        self.assertIsNone(ev.update(State()))
        self.assertIsNone(ev.update(State(sim_time=.4)))
        self.assertIsNone(ev.update(State(sim_time=.45, planar_velocity=(.1, 0.))))
        self.assertIsNone(ev.update(State(sim_time=.5)))
        self.assertIsNone(ev.update(State(sim_time=.99)))
        self.assertEqual(ev.update(State(sim_time=1.)), 'success')

    def test_fall_and_bounds_beat_success(self):
        task = dataclasses.replace(make_task('reach'), target_position=(.25,-.2,.9))
        ev = Evaluator(task)
        self.assertIsNone(ev.update(State(sim_time=0.)))
        self.assertIsNone(ev.update(State(sim_time=.4, pelvis_position=(0.,0.,.3))))
        self.assertEqual(ev.update(State(sim_time=.5, pelvis_position=(0.,0.,.3))), 'fell')
        ev = Evaluator(task)
        self.assertEqual(ev.update(State(pelvis_position=(.21,0.,.75))), 'base_displacement_exceeded')
        ev = Evaluator(make_task())
        self.assertEqual(ev.update(State(pelvis_position=(2.01,0.,.75))), 'left_workspace')

    def test_success_at_deadline_beats_timeout_and_invalid_state_beats_success(self):
        task = dataclasses.replace(make_task(), target_position=(0.,0.,0.), target_yaw=0., deadline=.5)
        ev = Evaluator(task)
        ev.update(State())
        self.assertEqual(ev.update(State(sim_time=.5)), 'success')
        ev = Evaluator(make_task())
        self.assertEqual(ev.update(State(sim_time=20.)), 'timeout')
        ev = Evaluator(task)
        ev.update(State())
        self.assertEqual(ev.update(State(sim_time=.5, pelvis_yaw=float('nan'))), 'backend_failed')

    def test_reach_base_conditions_and_nonmonotonic_state(self):
        task = dataclasses.replace(make_task('reach'), target_position=(.25,-.2,.9))
        ev = Evaluator(task)
        ev.update(State(pelvis_position=(.15,0.,.75)))
        self.assertIsNone(ev.update(State(sim_time=.6, pelvis_position=(.15,0.,.75))))
        self.assertEqual(ev.update(State(sim_time=.4)), 'backend_failed')
        self.assertEqual(Evaluator(task).update(State(state_age_s=.26)), 'state_stale')

    def test_wrapped_heading_is_used_for_success(self):
        task = dataclasses.replace(make_task(), target_position=(0.,0.,0.), target_yaw=-math.pi+.02)
        ev = Evaluator(task)
        ev.update(State(pelvis_yaw=math.pi-.02))
        self.assertEqual(ev.update(State(sim_time=.5, pelvis_yaw=math.pi-.02)), 'success')


class PlannerFieldsTests(unittest.TestCase):
    def test_rotated_forward_and_idle_semantics(self):
        fields = planner_command(.2, 0., 0., math.pi/2, .1)
        self.assertAlmostEqual(fields['movement'][0], 0.)
        self.assertAlmostEqual(fields['movement'][1], 1.)
        self.assertEqual(fields['mode'], 1)  # upstream SLOW_WALK
        self.assertEqual(fields['speed'], .2)
        self.assertEqual(planner_command(0,0,0,0,.1)['mode'], 0)

    def test_yaw_integration_and_invalid_requests(self):
        fields = planner_command(0,0,.4,math.pi-.01,.1)
        self.assertAlmostEqual(fields['heading'], -math.pi+.03)
        self.assertEqual(fields['mode'], 0)
        self.assertEqual(fields['movement'], [0.,0.,0.])
        for args in [(float('nan'),0,0,0,.1), (.4,0,0,0,.1), (0,0,0,0,-1)]:
            with self.assertRaises(ValueError):
                planner_command(*args)


if __name__ == '__main__':
    unittest.main()
