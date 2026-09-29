import json,unittest
from pathlib import Path
import numpy as np
import pinocchio as pin
from g1cap.arena_perception import ArenaBoxPerception

class InitialSupportRobotExclusionTests(unittest.TestCase):
    def test_recorded_rotated_frame_does_not_treat_forearms_as_support(self):
        root=Path(__file__).parent/'fixtures/rotated-support'
        fixture=json.loads((root/'measurement.json').read_text());arrays=np.load(root/'head.npz')
        packet=fixture['packet'];model=pin.buildModelFromUrdf(str(Path(__file__).resolve().parents[1]/'g1cap/assets/arena_g1_rev1_0_kinematics.urdf'))
        frontend=ArenaBoxPerception(model)
        prior=fixture['prior_box']
        frontend.track.initialize(time_s=prior['observed_at_s'],now_s=prior['observed_at_s'],center=prior['center_camera_m'],rotation=prior['axes_camera'],size=prior['dimensions_m'])
        frontend.advance_imu(packet['time_s'],packet['gyro_rad_s'])
        camera=dict(step=packet['step'],time_s=packet['time_s'],calibration=fixture['calibration'],rgb=arrays['rgb'],depth_m=arrays['depth_m'])
        result=frontend.update(packet,camera,fixture['up_body'])
        self.assertEqual(result['box']['status'],'accepted')
        self.assertEqual(result['support']['status'],'observed_candidate',result['support'])
        self.assertLess(abs(result['support']['gap_m']),.003)
        self.assertEqual(result['floor']['status'],'observed_candidate')
