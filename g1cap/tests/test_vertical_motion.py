import unittest
try:
 import numpy as np
except ImportError:
 np=None
if np is not None:
 from g1cap.vertical_motion import predict_vertical
@unittest.skipIf(np is None,"NumPy runtime required")
class VerticalTests(unittest.TestCase):
 def test_integrates_time_varying_acceleration(self):
  t=np.arange(31)*.02;a=3*t
  # z=.5*t^3 + .1*t; acceleration=3t exactly linear.
  h=np.c_[[0,.5],[0,.5*.5**3+.1*.5],[0,0]]
  r=predict_vertical(.58,.5,h,np.c_[t,a],error=.001)
  actual=.5*(.58**3-.5**3)+.1*.08
  self.assertAlmostEqual(r['delta_m'],actual,places=12)
 def test_correlated_endpoint_and_acceleration_errors_covered(self):
  t=np.arange(31)*.02;actual=.5*(.58**2-.5**2)
  for e0 in [-.003,.003]:
   for e1 in [-.003,.003]:
    for bias in [-.5,.5]:
     h=[[0,e0,.003],[.5,.125+e1,.003]]
     r=predict_vertical(.58,.5,h,np.c_[t,np.ones_like(t)+bias])
     self.assertLessEqual(abs(r['delta_m']-actual),r['error_m']+1e-12)
 def test_missing_acceleration_and_stale_camera_rejected(self):
  h=[[0,0,.003],[.5,0,.003]];t=np.arange(31)*.02
  with self.assertRaises(ValueError):predict_vertical(.58,.5,h,np.c_[t[::2],t[::2]*0])
  with self.assertRaises(ValueError):predict_vertical(.7,.5,h,np.c_[t,t*0])
 def test_future_tick_allows_downward_velocity_and_acceleration(self):
  t=np.arange(31)*.02;r=predict_vertical(.58,.5,[[0,0,0],[.5,-.1,0]],np.c_[t,t*0],error=.001)
  self.assertGreater(r['future_down_m'],.2*.02)
  self.assertGreaterEqual(r['downward_allowance_m'],.2*.1)
if __name__=='__main__':unittest.main()
