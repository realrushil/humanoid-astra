"""Paired wrist pose planning contracts; kinematics, not physical qualification."""
import importlib.util
import unittest
from test_arm_planner import AVAILABLE


@unittest.skipUnless(AVAILABLE,'MuJoCo installed on Linux')
class BimanualTests(unittest.TestCase):
    def test_both_pose_goals_and_frozen_root(self):
        self.assertIsNotNone(importlib.util.find_spec('g1cap.toolkit.bimanual'))
        import mujoco as mj
        import numpy as np
        from g1cap.toolkit.bimanual import BimanualPlanner
        # Two independent seven-joint arms in a tiny real MuJoCo model.
        import tempfile
        from pathlib import Path
        from g1cap.toolkit.arm_planner import ARM_SUFFIXES
        chains=[]
        for side,y in (('left',.2),('right',-.2)):
            chain=f'<body name="{side}_base" pos="0 {y} 0">'
            for i,s in enumerate(ARM_SUFFIXES):
                axis=('0 0 1','0 1 0','1 0 0')[i%3]
                chain+=f'<body name="{side}_{s}_link" pos=".06 0 0"><joint name="{side}_{s}_joint" axis="{axis}" range="-2 2"/><geom type="sphere" size=".01" mass=".1" contype="0" conaffinity="0"/>'
            chains.append(chain+'</body>'*8)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.xml'
            path.write_text('<mujoco><compiler angle="radian"/><worldbody><body name="pelvis" pos="0 0 1"><freejoint/><geom size=".02" mass="1" contype="0" conaffinity="0"/>'+''.join(chains)+'</body></worldbody></mujoco>')
            planner=BimanualPlanner(path)
            m=planner.model; d=mj.MjData(m); mj.mj_forward(m,d)
            sample=dict(pelvis_position=d.qpos[:3].tolist(),pelvis_quaternion_wxyz=d.qpos[3:7].tolist(),
                joint_names=[m.joint(j).name for j in range(1,m.njnt)],joint_positions=d.qpos[7:].tolist())
            d.qpos[m.joint('left_shoulder_pitch_joint').qposadr]=.15
            d.qpos[m.joint('right_shoulder_pitch_joint').qposadr]=-.15
            mj.mj_forward(m,d)
            goals={side:dict(position=d.body(side+'_wrist_yaw_link').xpos.tolist(),
                             quaternion=d.body(side+'_wrist_yaw_link').xquat.tolist()) for side in ('left','right')}
            result=planner.plan(sample,goals)
            self.assertEqual(result['status'],'planned',result)
            self.assertTrue(all(e['position_m']<.005 and e['orientation_rad']<.03 for e in result['errors'].values()))
            self.assertEqual(planner.data.qpos[:7].tolist(),[0,0,1,1,0,0,0])
            self.assertEqual(len(result['joint_path'][-1]),14)
            self.assertTrue(all(np.max(np.abs(np.array(b)-a))<=.025001 for a,b in zip(result['joint_path'],result['joint_path'][1:])))
            # A collision correction must handle all 14 joints, not accidentally
            # reuse the right-arm-only seven-element gradient.
            x,y,z=goals['left']['position']
            xml=path.read_text().replace('contype="0" conaffinity="0"','contype="1" conaffinity="1"')
            xml=xml.replace('<worldbody>',f'<worldbody><geom type="sphere" pos="{x} {y} {z}" size=".02"/>')
            path.write_text(xml)
            collision_result=BimanualPlanner(path).plan(sample,goals)
            self.assertEqual(collision_result['status'],'rejected',collision_result)
            goals['left']['quaternion']=[0,0,0,0]
            with self.assertRaises(ValueError): planner.plan(sample,goals)
