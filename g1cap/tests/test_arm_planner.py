"""Real MuJoCo kinematics tests; Linux SONIC environment has the dependency."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

AVAILABLE = importlib.util.find_spec('mujoco') is not None


@unittest.skipUnless(AVAILABLE, 'requires MuJoCo/NumPy; exercised on Linux')
class ArmPlannerTests(unittest.TestCase):
    def setUp(self):
        import mujoco
        import numpy as np
        from g1cap.toolkit.arm_planner import ArmPlanner
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # A simple planar seven-joint chain with an independent left joint.
        suffixes = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                    'wrist_roll', 'wrist_pitch', 'wrist_yaw')
        bodies = ''.join(f'<body name="right_{s}_link" pos=".08 0 0"><joint name="right_{s}_joint" axis="0 0 1" range="-1 1"/><geom type="sphere" size=".015" mass=".1" contype="0" conaffinity="0"/>' for s in suffixes)
        xml = '<mujoco><compiler angle="radian"/><worldbody><body name="pelvis" pos="0 0 1"><freejoint/><geom type="sphere" size=".02" mass="1" contype="0" conaffinity="0"/>' + bodies + '</body>'*7 + '<body name="left"><joint name="left_joint"/><geom type="sphere" size=".01" mass=".1" contype="0" conaffinity="0"/></body></body></worldbody></mujoco>'
        self.path = Path(self.temp.name)/'model.xml'
        self.path.write_text(xml)
        self.planner = ArmPlanner(self.path)
        model = self.planner.model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.sample = dict(pelvis_position=data.qpos[:3].tolist(),
                           pelvis_quaternion_wxyz=data.qpos[3:7].tolist(),
                           joint_names=[model.joint(i).name for i in range(1, model.njnt)],
                           joint_positions=data.qpos[7:].tolist())
        data.qpos[7] = .15
        mujoco.mj_forward(model, data)
        self.target = data.body('right_wrist_yaw_link').xpos.copy()

    def test_reaches_target_with_fixed_base_and_bounded_joint_steps(self):
        import numpy as np
        result = self.planner.plan(self.sample, self.target.tolist())
        self.assertEqual(result['status'], 'planned', result)
        self.assertLess(result['position_error'], .01)
        path = result['joint_path']
        self.assertEqual(len(path[0]), 7)
        self.assertTrue(all(max(abs(a-b) for a,b in zip(x,y)) <= .025001 for x,y in zip(path,path[1:])))
        self.assertEqual(self.planner.data.qpos[:7].tolist(), [0.,0.,1.,1.,0.,0.,0.])
        self.assertEqual(self.planner.data.joint('left_joint').qpos[0], 0.)

    def test_far_target_and_missing_joint_are_explicit_failures(self):
        result = self.planner.plan(self.sample, [9., 0., 1.])
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(result['reason'], 'target_outside_local_envelope')
        self.sample['joint_names'][0] = 'wrong_name'
        with self.assertRaises(ValueError):
            self.planner.plan(self.sample, self.target.tolist())

    def test_session_reach_check_plans_without_mutating_measured_state_or_command(self):
        import copy
        from types import SimpleNamespace
        from test_session import sample
        from g1cap.session_tools import SessionTools
        raw=sample(1.,**self.sample)
        before=copy.deepcopy(raw)
        command=('velocity',{'vx':.1})
        # Deliberately no command methods: a compute-only check must not need them.
        session=SimpleNamespace(observe=lambda:raw,desired=command)
        result=SessionTools(session,self.planner).check_reach(self.target.tolist())
        self.assertEqual(result['status'],'planned',result)
        self.assertLess(result['position_error'],.01)
        self.assertEqual(result['sequence'],200)
        self.assertEqual(raw,before)
        self.assertEqual(session.desired,command)

    def test_local_unreachable_target_does_not_return_fake_plan(self):
        result = self.planner.plan(self.sample, [.66,0.,1.])
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(result['reason'], 'ik_not_converged')
        self.assertEqual(result['joint_path'], [])

    def test_path_contact_rejects_an_otherwise_reachable_target(self):
        from g1cap.toolkit.arm_planner import ArmPlanner
        xml = self.path.read_text().replace('contype="0" conaffinity="0"','contype="1" conaffinity="1"')
        x,y,z = self.target
        xml = xml.replace('<worldbody>',f'<worldbody><geom type="sphere" pos="{x} {y} {z}" size=".025"/>')
        self.path.write_text(xml)
        planner = ArmPlanner(self.path)
        result = planner.plan(self.sample,self.target.tolist())
        self.assertEqual(result['status'],'rejected')
        self.assertEqual(result['reason'],'path_model_collision')
        self.assertLess(result['collision']['distance_m'],-.002)
        self.assertEqual(len(result['collision']['bodies']),2)

    def test_saved_g1_paths_avoid_thumb_hip_with_dense_check(self):
        import json
        import mujoco
        import numpy as np
        from g1cap.toolkit.arm_planner import ArmPlanner
        root=Path(__file__).resolve().parents[1]
        model_path=root/'deps/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'
        if not model_path.exists():
            self.skipTest('requires pinned G1/Dex3 assets; exercised on Linux')
        cases=json.loads((root/'tests/fixtures/g1_arm_collision.json').read_text())
        for case in cases:
            with self.subTest(case=case['case']):
                planner=ArmPlanner(model_path)
                result=planner.plan(case['sample'],case['target_world'])
                self.assertEqual(result['status'],'planned',result)
                self.assertGreater(result['collision_corrections'],0)
                data=mujoco.MjData(planner.model)
                data.qpos[:]=planner.data.qpos[:]
                for a,b in zip(result['joint_path'],result['joint_path'][1:]):
                    for alpha in np.linspace(0,1,26):
                        data.qpos[planner.q_indices]=np.array(a)+(np.array(b)-a)*alpha
                        mujoco.mj_forward(planner.model,data)
                        for c in data.contact:
                            bodies={planner.model.body(int(planner.model.geom_bodyid[g])).name for g in c.geom}
                            if any(bodies=={'world',s+'_ankle_roll_link'} for s in ('left','right')):
                                continue
                            self.assertGreaterEqual(c.dist,-.002)
