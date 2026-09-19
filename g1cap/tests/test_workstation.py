"""Task semantics and real worker/scene boundaries, without a controller process."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from test_session import Publisher, sample


class WorkstationTests(unittest.TestCase):
    def task(self, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec('g1cap.workstation_task'),
                             'workstation task is not implemented')
        from g1cap.workstation_task import WorkstationTask
        from g1cap.scene import workstation_scene
        return WorkstationTask(workstation_scene(), **kwargs)

    def evaluator(self, **kwargs):
        task = self.task(**kwargs)
        from g1cap.workstation_task import WorkstationEvaluator
        return WorkstationEvaluator(task, sample(0.))

    def test_layout_translation_moves_geometry_and_goal_together(self):
        from g1cap.scene import workstation_scene
        from g1cap.session_runtime import task_from_recipe
        base = workstation_scene()
        shifted = task_from_recipe({'task':'workstation_reach', 'target_id':'blue_upper',
                                    'layout_offset_xy':[.12, -.04]})
        for old, new in zip(base.boxes, shifted.scene.boxes):
            self.assertAlmostEqual(new.center_world[0]-old.center_world[0], .12)
            self.assertAlmostEqual(new.center_world[1]-old.center_world[1], -.04)
            self.assertEqual(new.half_size, old.half_size)
        for old, new in zip(base.markers, shifted.scene.markers):
            self.assertAlmostEqual(new.position_world[0]-old.position_world[0], .12)
            self.assertAlmostEqual(new.position_world[1]-old.position_world[1], -.04)
            self.assertEqual(new.position_world[2], old.position_world[2])
        from g1cap.workstation_task import WorkstationEvaluator
        e = WorkstationEvaluator(shifted, sample(0.))
        for i in range(1, 8):
            e.update(sample(i*.1, right_wrist_position=[.78, -.24, .86]))
        self.assertIsNone(e.terminal_reason)
        target = shifted.scene.marker('blue_upper').position_world
        for i in range(8, 16):
            e.update(sample(i*.1, right_wrist_position=list(target)))
        self.assertEqual(e.terminal_reason, 'success')

    def test_bad_layout_translation_rejected_before_startup(self):
        from g1cap.scene import workstation_scene
        for value in ([0], [0, float('nan')], 'ab', [False, 0]):
            with self.assertRaises(ValueError): workstation_scene(value)

    def test_goal_does_not_prescribe_base_pose_or_height(self):
        task = self.task().to_dict()
        self.assertEqual(task['target_id'], 'blue_lower')
        for key in ('approach_world_xy', 'height', 'reference_code', 'ideal_base_pose'):
            self.assertNotIn(key, task)
        for base in ([.19, -.02, .70], [.29, .03, .79]):
            e = self.evaluator()
            for i in range(1, 8):
                e.update(sample(i*.1, pelvis_position=base,
                                right_wrist_position=[.78, -.24, .70]))
            self.assertEqual(e.terminal_reason, 'success')

    def test_wrong_marker_cannot_succeed(self):
        e = self.evaluator()
        for i in range(1, 12):
            e.update(sample(i*.1, right_wrist_position=[.78, .42, .70]))
        self.assertIsNone(e.terminal_reason)
        self.assertAlmostEqual(e.metrics['wrist_error_m'], .66)

    def test_interrupted_dwell_and_fast_wrist_do_not_accumulate(self):
        e = self.evaluator()
        goal = dict(right_wrist_position=[.78, -.24, .70])
        for i in range(1, 5): e.update(sample(i*.1, **goal))
        e.update(sample(.5, **goal, right_wrist_velocity_world=[.2, 0., 0.]))
        for i in range(6, 11):
            self.assertIsNone(e.update(sample(i*.1, **goal)))
        self.assertEqual(e.update(sample(1.1, **goal)), 'success')

    def test_contact_and_missing_history_override_goal(self):
        for bad, reason in (({'forbidden_contacts': [{}]}, 'forbidden_contact'),
                            ({'no_support': False}, 'invalid_state')):
            e = self.evaluator()
            e.update(sample(.1, right_wrist_position=[.78, -.24, .70], **bad))
            self.assertEqual(e.terminal_reason, reason)
        e = self.evaluator()
        self.assertEqual(e.update(sample(.3)), 'state_gap')

    def test_deadline_remains_across_a_program_boundary(self):
        e = self.evaluator(deadline=1.)
        for i in range(1, 12): e.update(sample(i*.1))
        self.assertEqual(e.terminal_reason, 'timeout')
        self.assertEqual(e.update(sample(1.2, right_wrist_position=[.78,-.24,.70])), 'timeout')

    def test_single_foot_does_not_earn_goal_dwell(self):
        e = self.evaluator()
        for i in range(1, 12):
            e.update(sample(i*.1, right_wrist_position=[.78,-.24,.70],
                            foot_normal_forces={'left':0., 'right':100.}))
        self.assertIsNone(e.terminal_reason)

    def test_unknown_target_or_invalid_budget_fails_before_execution(self):
        self.task()
        for kwargs in ({'target_id':'missing'}, {'deadline':float('nan')}, {'deadline':0.}):
            with self.assertRaises(ValueError): self.task(**kwargs)

    def test_recipe_selection_and_generation_keep_the_named_goal(self):
        from g1cap import session_runtime, interactive
        self.assertTrue(hasattr(session_runtime, 'task_from_recipe'))
        task = session_runtime.task_from_recipe({'task':'workstation_reach','target_id':'blue_upper'})
        self.assertEqual(task.target_id, 'blue_upper')
        with self.assertRaises(ValueError):
            session_runtime.task_from_recipe({'task':'typo'})
        with self.assertRaises(ValueError):
            session_runtime.task_from_recipe({'task':'workstation_reach','height':.7})
        self.assertIsNone(session_runtime.task_from_recipe({}))
        self.assertTrue(hasattr(interactive, 'generation_request'))
        status = dict(task=task.to_dict(), observation=sample(0.), scene=task.scene.to_dict())
        request = interactive.generation_request(status, 'old code', [{'round':0}])
        self.assertEqual(request['instruction'], task.to_dict()['instruction'])
        self.assertEqual(request['task']['target_id'], 'blue_upper')
        self.assertEqual(request['scene']['markers'][1]['position_world'], [.78,-.24,.86])
        self.assertEqual(request['previous_source'], 'old code')

    def test_scene_rejects_invalid_geometry_and_missing_owners(self):
        self.task()
        from g1cap.scene import Box, Marker, Scene
        with self.assertRaises(ValueError):
            Scene('bad', (Box('box',(0.,0.,0.),(-1.,1.,1.),(1.,1.,1.,1.)),), ())
        with self.assertRaises(ValueError):
            Scene('bad', (), (Marker('goal','missing',(0.,0.,1.),'goal'),))

    def test_plinth_contact_cannot_count_as_floor_support(self):
        from g1cap.sonic_sim import summarize_contacts
        result = summarize_contacts([dict(body1='blue_workstation',
            body2='right_ankle_roll_link',normal_force=100.,distance=-.001,self_contact=False)])
        self.assertEqual(result['foot_normal_forces']['right'], 0.)
        self.assertEqual(len(result['forbidden_contacts']), 1)

    def test_scene_query_and_planning_use_real_worker_without_moving_robot(self):
        task = self.task()
        from g1cap.session import Session
        with tempfile.TemporaryDirectory() as root:
            p = Publisher()
            session = Session(p, None, task, root).start()
            try:
                first = session.execute('''def run(robot, task):
    scene = robot.observe_scene()
    target = next(m for m in scene['markers'] if m['name'] == task['target_id'])
    assert target['position_world'] == [.78, -.24, .70]
    result = robot.check_reach(target['position_world'])
    assert result['reason'] == 'planner_unavailable'
    scene['markers'][0]['position_world'][0] = 99
    assert robot.observe_scene()['markers'][0]['position_world'][0] == .78
    raise ValueError('continue the same scene')
''')
                self.assertEqual(first['execution']['status'], 'policy_error')
                second = session.execute('''def run(robot, task):
    assert robot.observe_scene()['markers'][0]['position_world'][0] == .78
''')
                self.assertEqual(second['execution']['status'], 'completed')
                self.assertEqual(first['session_id'], second['session_id'])
                self.assertGreater(second['start_observation']['elapsed'], first['start_observation']['elapsed'])
                self.assertTrue(all(c[1] == 'idle' for c in p.commands))
                self.assertIsNone(session.status()['terminal_reason'])
            finally:
                session.close()


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'MuJoCo is installed on Linux')
class SceneModelTests(unittest.TestCase):
    def test_scene_has_physical_boxes_but_markers_do_not_contact(self):
        self.assertIsNotNone(importlib.util.find_spec('g1cap.scene'))
        import mujoco as mj
        from g1cap.scene import workstation_scene, write_scene_model
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root/'robot.xml'
            # A free sphere overlaps the independently specified blue box.
            source.write_text('''<mujoco><worldbody>
              <body name="robot" pos=".96 -.24 .4"><freejoint/>
                <geom type="sphere" size=".1" mass="1"/>
              </body></worldbody></mujoco>''')
            plain = mj.MjModel.from_xml_path(str(source))
            destination = root/'elsewhere'/'scene.xml'
            write_scene_model(source, destination, workstation_scene())
            model = mj.MjModel.from_xml_path(str(destination))
            data = mj.MjData(model)
            mj.mj_forward(model, data)
            self.assertEqual((model.nq,model.nv,model.nu), (plain.nq,plain.nv,plain.nu))
            self.assertGreater(data.ncon, 0)
            self.assertEqual(model.body(model.geom('blue_workstation').bodyid[0]).name,
                             'blue_workstation')
            self.assertEqual(model.site('blue_lower').pos.tolist(), [.78,-.24,.70])
            self.assertGreater(model.geom('blue_workstation').contype[0], 0)


if __name__ == '__main__':
    unittest.main()
