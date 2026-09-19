import unittest
from types import SimpleNamespace

try:
    import numpy as np
except ImportError:
    np=None


@unittest.skipIf(np is None,'NumPy runtime required')
class CarryPerceptionTests(unittest.TestCase):
    def frontend(self):
        from g1cap.arena_perception import ArenaBoxPerception
        from g1cap.carry_frame import FloorCarryFrame
        frontend=ArenaBoxPerception(None)
        # This estimator-boundary fixture contains external planar points;
        # full recorded sensor replay separately exercises robot exclusion/FK.
        frontend.robot_geometry=SimpleNamespace(exclude=lambda points,packet,calibration:
            np.zeros(len(points),bool))
        x,y=np.meshgrid(np.linspace(-.5,.5,35),np.linspace(-.5,.5,35))
        points=np.c_[x.ravel(),y.ravel(),np.full(x.size,.7)]
        floor=points.copy();floor[:,:2]*=4;floor[:,2]=0
        frame=FloorCarryFrame().update(1.,np.eye(3),
            dict(status='observed_candidate',normal_body=[0,0,1],offset_m=.75))
        # Feed a permitted geometry result at the explicit estimator boundary.
        # Camera/kinematic accuracy is checked separately by full sensor replay.
        frontend.carry_seed=dict(time_s=1.,frame=frame,camera_transform=np.eye(4),
            box=dict(status='accepted',center_camera_m=[0,0,.9],axes_camera=np.eye(3).tolist(),dimensions_m=[.2]*3),
            initial=dict(status='observed_candidate',normal_camera=[0,0,1],offset_m=-.7,gap_m=.1),
            points=points,floor_points=floor,packet={},calibration={})
        return frontend

    def test_carry_reference_requires_fresh_observation_and_explicit_start(self):
        frontend=self.frontend()
        self.assertIsNone(frontend.source_plane)
        with self.assertRaises(ValueError):frontend.begin_carry(1.2)
        frontend.begin_carry(1.1)
        self.assertIsNotNone(frontend.source_plane)
        original=frontend.source_plane
        frontend.begin_carry(1.1)
        self.assertIs(frontend.source_plane,original)

    def test_missing_or_unassociated_table_cannot_begin_carry(self):
        frontend=self.frontend();frontend.carry_seed=None
        with self.assertRaises(ValueError):frontend.begin_carry(1.)
        frontend=self.frontend();frontend.carry_seed['initial']={'status':'unavailable'}
        with self.assertRaises(ValueError):frontend.begin_carry(1.)
        self.assertIsNone(frontend.source_plane)

    def test_acquisition_reset_discards_carry_plane_but_not_scene_history(self):
        frontend=self.frontend();motion=frontend.motion
        frontend.begin_carry(1.)
        frontend.begin_acquisition()
        self.assertIsNone(frontend.source_plane)
        self.assertIs(frontend.motion,motion)

    def test_initial_table_must_match_the_level_floor_model(self):
        frontend=self.frontend()
        frontend.carry_seed['points'][:,2]+=.05*frontend.carry_seed['points'][:,0]
        with self.assertRaises(ValueError):frontend.begin_carry(1.)
        self.assertIsNone(frontend.source_plane)

    def test_initial_association_requires_current_floor_depth(self):
        frontend=self.frontend();frontend.carry_seed['floor_points']=np.empty((0,3))
        with self.assertRaises(ValueError):frontend.begin_carry(1.)
        self.assertIsNone(frontend.source_plane)


if __name__=='__main__':unittest.main()
