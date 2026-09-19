import importlib
import itertools
from pathlib import Path
import unittest

try:
    import numpy as np
    import pinocchio as pin
except ImportError:
    pin=None


@unittest.skipIf(pin is None,'Pinocchio is required')
class RobotGeometryTests(unittest.TestCase):
    def test_unknown_fingers_are_enclosed_without_inventing_encoder_angles(self):
        try:module=importlib.import_module('g1cap.sensor_robot_geometry')
        except ModuleNotFoundError:self.fail('calibrated sensor robot bounds are missing')
        from g1cap.sensor_kinematics import frame_in_body
        assets=Path('g1cap/assets')
        model=pin.buildModelFromUrdf(str(assets/'arena_g1_rev1_0_kinematics.urdf'))
        fingers=[n for n in list(model.names)[1:] if '_hand_' in n]
        names=[n for n in list(model.names)[1:] if n not in fingers]
        geometry=module.RobotGeometry(assets/'arena_g1_rev1_0_kinematics.urdf',
                                     assets/'arena_g1_rev1_0_bounds.json',names)
        self.assertEqual(len(names),29);self.assertEqual(len(geometry.spheres),6)
        packet=dict(joint_names=names,q_rad=[0.]*len(names))
        camera=dict(parent_frame='pelvis',offset_convention='ros',
                    offset_position_m=[0,0,0],offset_quaternion_xyzw=[0,0,0,1])
        unknown=set(geometry.bounds)-set(geometry.rigid)
        rng=np.random.default_rng(2)
        for _ in range(3):
            full=dict(zip(names,packet['q_rad']))
            full.update({name:rng.uniform(-1.,1.) for name in fingers})
            points=[]
            for link in unknown:
                pose=frame_in_body(model,link,full)
                for shape in geometry.bounds[link]:
                    corners=np.array(list(itertools.product(*zip(shape['min'],shape['max']))))
                    points.extend(corners@pose[:3,:3].T+pose[:3,3])
            self.assertTrue(geometry.exclude(np.array(points),packet,camera).all())
        # Clearance uses calibrated lower-body extents; no scene/world pose.
        n,offset,clearance=geometry.front_clearance(
            dict(normal_camera=[1.,0,0],offset_m=-1.),packet,camera)
        self.assertGreater(clearance,.5);self.assertLess(clearance,1.)
        np.testing.assert_array_equal(n,[1,0,0]);self.assertEqual(offset,-1.)
        missing=dict(joint_names=names[:-1],q_rad=[0.]*(len(names)-1))
        with self.assertRaises(ValueError):geometry.exclude(np.zeros((1,3)),missing,camera)
