import json
from pathlib import Path
import unittest
try:
 import numpy as np
 import pinocchio as pin
except ImportError:pin=None
if pin is not None:
 from g1cap.arm_gravity import ArmGravity
from g1cap.arena_sensors import proprioception_packet
from g1cap.hand_sensors import DEX3_JOINTS,hand_position_packet


@unittest.skipIf(pin is None,'Pinocchio is required')
class ArmGravityTests(unittest.TestCase):
 def setUp(self):
  self.model=pin.buildModelFromUrdf('g1cap/assets/arena_g1_rev1_0_kinematics.urdf')
  self.names=list(self.model.names)[1:];self.body=[n for n in self.names if n not in DEX3_JOINTS]
  self.mass=json.loads(Path('g1cap/assets/arena_g1_arm_mass.json').read_text())['links']
  self.arms=[n for n in self.names if any(s in n for s in ('shoulder','elbow','wrist'))]
  self.kp={n:40. for n in self.arms}
  self.control=ArmGravity(self.model,self.names,self.kp,self.mass)

 def sample(self,step):
  b=proprioception_packet(step=step,time_s=step*.02,joint_names=self.body,
   q=[0.]*29,dq=[0.]*29,tau_est=[0.]*29,gyro=[0.]*3,accel=[0.,0.,9.81])
  h=hand_position_packet(step=step,time_s=step*.02,left=[0.]*7,right=[0.]*7)
  return b,h

 def test_only_arms_change_with_bounded_magnitude_slew_and_zero_initial_jump(self):
  target=np.zeros(43);previous=np.zeros(14)
  for step in range(30):
   b,h=self.sample(step);result=self.control.command(b,h,[0.,0.,1.],target)
   offset=np.array(self.control.last['slew_offset_rad'])
   self.assertLessEqual(max(abs(offset)),.12+1e-12)
   self.assertLessEqual(max(abs(offset-previous)),.008+1e-12)
   if step==0:np.testing.assert_array_equal(result,target)
   other=[i for i,n in enumerate(self.names) if n not in self.arms]
   np.testing.assert_array_equal(result[other],target[other]);previous=offset
  self.assertGreater(max(abs(offset)),.001)

 def test_gravity_is_the_potential_gradient_in_the_estimated_body_frame(self):
  b,h=self.sample(0);up=np.array([.1,.2,1.]);up/=np.linalg.norm(up)
  self.control.command(b,h,up.tolist(),np.zeros(43));gravity=self.control.last['gravity_nm']
  data=self.model.createData()
  def potential(q):
   pin.framesForwardKinematics(self.model,data,q)
   return sum(9.81*v['mass_kg']*float(up@(data.oMf[self.model.getFrameId(n)]*np.array(v['com_m']))) for n,v in self.mass.items())
  for name,tau in zip(self.arms,gravity):
   i=self.model.joints[self.model.getJointId(name)].idx_q;dq=np.zeros(43);dq[i]=1e-5
   self.assertAlmostEqual(tau,(potential(dq)-potential(-dq))/2e-5,places=6)

 def test_motor_stiffness_scales_only_the_reference_offset(self):
  other=ArmGravity(self.model,self.names,{n:2*k for n,k in self.kp.items()},self.mass)
  for c in (self.control,other):
   b,h=self.sample(0);c.command(b,h,[0.,0.,1.],np.zeros(43))
  np.testing.assert_allclose(other.last['gravity_nm'],self.control.last['gravity_nm'])
  np.testing.assert_allclose(other.last['desired_offset_rad'],np.array(self.control.last['desired_offset_rad'])/2)

 def test_invalid_mapping_mass_or_stiffness_rejects_before_use(self):
  for kp,mass in (({},self.mass),(self.kp,{}),({n:0. for n in self.arms},self.mass)):
   with self.assertRaises(ValueError):ArmGravity(self.model,self.names,kp,mass)
  bad=json.loads(json.dumps(self.mass));next(iter(bad.values()))['com_m']=[float('nan'),0.,0.]
  with self.assertRaises(ValueError):ArmGravity(self.model,self.names,self.kp,bad)

 def test_missing_hand_timing_or_sample_gap_invalidates(self):
  b,h=self.sample(0);self.control.command(b,h,[0.,0.,1.],np.zeros(43))
  b,h=self.sample(2)
  with self.assertRaises(ValueError):self.control.command(b,h,[0.,0.,1.],np.zeros(43))
  b,h=self.sample(1)
  with self.assertRaises(ValueError):self.control.command(b,h,[0.,0.,1.],np.zeros(43))
  self.setUp();b,h=self.sample(1);h['step']=0
  with self.assertRaises(ValueError):self.control.command(b,h,[0.,0.,1.],np.zeros(43))

 def test_joint_limit_clipping_is_reported(self):
  b,h=self.sample(0);self.control.command(b,h,[0.,0.,1.],np.zeros(43))
  target=np.zeros(43)
  for name,tau in zip(self.arms,self.control.last['gravity_nm']):
   i=self.names.index(name);j=self.model.joints[self.model.getJointId(name)].idx_q
   target[i]=self.model.upperPositionLimit[j] if tau>0 else self.model.lowerPositionLimit[j]
  b,h=self.sample(1);result=self.control.command(b,h,[0.,0.,1.],target)
  self.assertTrue(self.control.last['joint_limit_clipped'])
  np.testing.assert_array_equal(result,target)


if __name__=='__main__':unittest.main()
