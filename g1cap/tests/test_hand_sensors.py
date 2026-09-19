from copy import deepcopy
import importlib
import unittest
from g1cap.arena_sensors import proprioception_packet

class HandSensorTests(unittest.TestCase):
    def module(self):
        try:return importlib.import_module('g1cap.hand_sensors')
        except ModuleNotFoundError:self.fail('explicit Dex3 position packet is missing')

    def body(self):
        return proprioception_packet(step=5,time_s=.1,joint_names=['left_elbow_joint'],
            q=[.3],dq=[0.],tau_est=[1.],gyro=[0.,0.,0.],accel=[0.,0.,9.81])

    def test_seven_per_hand_mapping_and_position_only_packet(self):
        m=self.module();left=list(range(7));right=list(range(10,17))
        packet=m.hand_position_packet(step=5,time_s=.1,left=left,right=right)
        self.assertEqual(set(packet),{'version','model','step','time_s','joint_names','q_rad'})
        self.assertEqual(packet['model'],'unitree_dex3_1')
        self.assertEqual(packet['joint_names'][:3],[f'left_hand_thumb_{i}_joint' for i in range(3)])
        self.assertEqual(packet['joint_names'][3:7],['left_hand_middle_0_joint','left_hand_middle_1_joint','left_hand_index_0_joint','left_hand_index_1_joint'])
        self.assertEqual(packet['q_rad'],left+right)
        left[0]=999;self.assertEqual(packet['q_rad'][0],0)
        with self.assertRaises(TypeError):m.hand_position_packet(step=5,time_s=.1,left=left,right=right,tactile=[])

    def test_joint_assembly_requires_exact_synchronized_measurements(self):
        m=self.module();hands=m.hand_position_packet(step=5,time_s=.1,left=list(range(7)),right=list(range(7,14)))
        body=self.body();names=list(reversed(hands['joint_names']+body['joint_names']))
        result=m.measured_joint_positions(body,hands,names)
        self.assertEqual(result,[.3]+list(reversed(range(14))))
        for key,value in [('time_s',.12),('step',6),('q_rad',[0.]*13),('model','unknown'),('contact',True)]:
            bad=deepcopy(hands);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.measured_joint_positions(body,bad,names)
        for bad in [names[:-1],names+[names[0]],names+['unknown']]:
            with self.assertRaises(ValueError):m.measured_joint_positions(body,hands,bad)

    def test_rejects_nonfinite_missing_or_future_packet_fields(self):
        m=self.module();args=dict(step=5,time_s=.1,left=[0.]*7,right=[0.]*7)
        for key,value in [('step',True),('time_s',-.1),('left',[0.]*6),('right',[float('nan')]*7)]:
            bad=dict(args);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.hand_position_packet(**bad)
        packet=m.hand_position_packet(**args);body=self.body();body['box_mass']=.1
        with self.assertRaises(ValueError):m.measured_joint_positions(body,packet,list(body['joint_names'])+packet['joint_names'])

    def test_native_sample_preserves_camera_age_and_rejects_stale_images(self):
        m=self.module();body=self.body();hands=m.hand_position_packet(step=5,time_s=.1,left=[0.]*7,right=[0.]*7)
        names=body['joint_names']+hands['joint_names'];camera=dict(step=0,time_s=0.)
        sample=m.acquisition_joint_sample(body,hands,camera,names)
        self.assertEqual(sample['camera_age_s'],.1)
        self.assertEqual(sample['camera_time_s'],0.)
        self.assertEqual(sample['q_rad'],[.3]+[0.]*14)
        for bad in [dict(step=6,time_s=.12),dict(step=0,time_s=-.1),dict(step=6,time_s=.1)]:
            with self.assertRaises(ValueError):m.acquisition_joint_sample(body,hands,bad,names)
