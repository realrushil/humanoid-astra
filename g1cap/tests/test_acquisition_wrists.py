"""Actual native PINK projection: preserve ownership and measured/world parity."""
import unittest
try:
 import numpy as np
 from scipy.spatial.transform import Rotation
 from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.g1 import instantiate_g1_robot_model
 from g1cap.toolkit.acquisition_wrists import AcquisitionHandClearance
except ImportError:
 instantiate_g1_robot_model=None

@unittest.skipIf(instantiate_g1_robot_model is None,'native Arena/PINK model required')
class AcquisitionWristTests(unittest.TestCase):
 def test_measured_up_matches_world_guard_and_preserves_nonarm_references(self):
  robot=instantiate_g1_robot_model();names=list(robot.joint_names)
  joints=robot.q_default.tolist();proposal=joints+[.02,.03,.04,.75,0.,0.,0.]
  geometry={'robot':{'left_hand_link':[dict(min=[-.02]*3,max=[.02]*3)]}}
  a=AcquisitionHandClearance(robot,names,geometry);b=AcquisitionHandClearance(robot,names,None)
  rotation=Rotation.from_euler('xyz',[.04,-.06,.2]);xyzw=rotation.as_quat()
  for step in range(8):
   t=step*.02;gap=.04-step*.005
   obs=dict(time=t,joint_pos=joints,root_quat=[xyzw[3],*xyzw[:3]],
    body_poses={'left_hand_link':dict(pos=[0.,0.,gap+.02],xyzw=[0.,0.,0.,1.])},
    surfaces={'table':{'bounds':dict(min=[-.5,-.5,-.1],max=[.5,.5,0.])}})
   feedback=dict(status='available',time_s=t,margin_m=gap,up_body=(rotation.as_matrix().T@np.array([0.,0.,1.])).tolist())
   actual=b.command_measured(joints,proposal,feedback)
   np.testing.assert_allclose(actual,a.command(obs,proposal),atol=1e-10,rtol=0)
   for i in range(50):
    if i not in b.indices:self.assertEqual(actual[i],proposal[i])
   self.assertLessEqual(max(abs(x) for x in b.extra),.20)
  self.assertTrue(b.last['corrected'])
  with self.assertRaisesRegex(ValueError,'sensor_hand_clearance_unavailable'):
   b.command_measured(joints,proposal,dict(status='unavailable'))
