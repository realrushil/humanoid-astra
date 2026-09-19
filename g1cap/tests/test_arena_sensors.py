"""Sensor packets must exclude oracle state and preserve physical conventions."""
import importlib
import math
import unittest


class ArenaSensorsTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('g1cap.arena_sensors')
        except ModuleNotFoundError:
            self.fail('The explicit RGB-D/proprioception packet implementation is missing')

    def args(self):
        return dict(step=7, time_s=.14, joint_names=['left_elbow_joint'],
                    q=[.4], dq=[-.2], tau_est=[1.5], gyro=[.1,.2,.3], accel=[0.,0.,9.81])

    def test_packet_has_only_sensor_quantities_and_detached_values(self):
        args=self.args();p=self.module().proprioception_packet(**args)
        self.assertEqual(set(p),{'version','step','time_s','joint_names','q_rad','dq_rad_s',
                                 'tau_est_nm','gyro_rad_s','specific_force_m_s2'})
        self.assertEqual(p['specific_force_m_s2'],[0.,0.,9.81])
        args['tau_est'][0]=999
        self.assertEqual(p['tau_est_nm'],[1.5])
        # No mass/true contact inputs are accepted at this measurement boundary.
        with self.assertRaises(TypeError):self.module().proprioception_packet(**args,box_mass_kg=.5)

    def test_rejects_missing_nonfinite_misaligned_or_duplicate_measurements(self):
        m=self.module()
        for key,value in [('step',True),('time_s',-.1),('q',[math.nan]),('tau_est',[]),
                          ('gyro',[0.,0.]),('accel',[0.,0.,math.inf]),('joint_names',['x','x'])]:
            args=self.args();args[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.proprioception_packet(**args)

    def test_depth_projection_is_optical_z_not_radial_distance(self):
        # At u=150 with fx=100,cx=50 and Z=2m, X=2m and Z=2m (range=sqrt(8)).
        m=self.module()
        self.assertEqual(m.optical_point(150,60,2.,[[100,0,50],[0,100,60],[0,0,1]]),[2.,0.,2.])
        self.assertEqual(m.optical_point(50,10,2.,[[100,0,50],[0,100,60],[0,0,1]]),[0.,-1.,2.])

    def test_invalid_depth_is_unknown_not_zero_distance(self):
        m=self.module();k=[[100,0,50],[0,100,60],[0,0,1]]
        for z in [0.,-1.,math.inf,math.nan]:
            self.assertIsNone(m.optical_point(50,60,z,k))
        with self.assertRaises(ValueError):m.optical_point(50,60,2.,[[0,0,50],[0,100,60],[0,0,1]])
