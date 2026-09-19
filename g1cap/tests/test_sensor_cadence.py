"""Recorder capture timing with small native-shaped arrays, without Isaac."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib
import math
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

def recorder_class():
    # The isolated scientific venv has NumPy but no video stack; file contents
    # are immaterial to cadence, so provide a deterministic PNG writer stand-in.
    if 'g1cap.arena_sensor_recording' not in sys.modules:
        imageio=SimpleNamespace(imwrite=lambda path,rgb:Path(path).write_bytes(rgb.tobytes()))
        with patch.dict(sys.modules,{'imageio':SimpleNamespace(v2=imageio),'imageio.v2':imageio}):
            return importlib.import_module('g1cap.arena_sensor_recording').SensorRecorder
    return sys.modules['g1cap.arena_sensor_recording'].SensorRecorder


class Tensor:
    def __init__(self, value):self.value=np.asarray(value)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value


def native(period_s=.02):
    names=[f'body_{i}' for i in range(29)]
    actuator=SimpleNamespace(joint_names=names,effort_limit=Tensor([[10.]*29]),
                             stiffness=Tensor([[100.]*29]),applied_effort=Tensor([[0.]*29]))
    class ActuatorMap(dict):pass
    actuators=ActuatorMap(body=actuator)
    actuators.applied_effort=Tensor([[0.]*29])
    robot=SimpleNamespace(joint_names=names,actuators=actuators,
                          data=SimpleNamespace(joint_pos=Tensor([[0.]*29]),joint_vel=Tensor([[0.]*29])))
    camera=SimpleNamespace(cfg=SimpleNamespace(update_period=period_s,width=2,height=2,
        offset=SimpleNamespace(pos=(0,0,0),rot=(0,0,0,1),convention='world')),
        data=SimpleNamespace(intrinsic_matrices=Tensor([np.eye(3)]),
            output={'rgb':Tensor(np.zeros((1,2,2,4),dtype=np.uint8)),
                    'distance_to_image_plane':Tensor(np.ones((1,2,2,1)))}))
    imu=SimpleNamespace(data=SimpleNamespace(ang_vel_b=Tensor([[0,0,0]]),lin_acc_b=Tensor([[0,0,9.81]])))
    model=SimpleNamespace(joint_names=names,get_joint_group_indices=lambda group:list(range(29)))
    return SimpleNamespace(scene={'robot':robot,'robot_head_cam':camera,'body_imu':imu}),model


@unittest.skipIf(np is None, 'NumPy is required for native-shaped sensor arrays')
class SensorCadenceTests(unittest.TestCase):
    def test_recorder_captures_at_configured_steps_and_preserves_timestamp(self):
        SensorRecorder=recorder_class()
        source,model=native()
        with tempfile.TemporaryDirectory() as root:
            recorder=SensorRecorder(Path(root)/'sensor',source,model,rgbd_period_steps=2)
            for step in range(5):
                recorder.capture(step,step*.02)
                packet,camera=recorder.measurements().values()
                self.assertEqual(packet['step'],step)
                self.assertEqual(camera['step'],step-step%2)
                self.assertAlmostEqual(camera['time_s'],camera['step']*.02)
            recorder.close()
            frames=[json.loads(line) for line in (Path(root)/'sensor/rgbd.jsonl').read_text().splitlines()]
            self.assertEqual([frame['step'] for frame in frames],[0,2,4])
            metadata=json.loads((Path(root)/'sensor/metadata.json').read_text())
            self.assertEqual(metadata['rgbd_hz'],25)
            self.assertEqual(metadata['rgbd_period_steps'],2)
            self.assertEqual(metadata['native_camera_update_period_s'],.02)
            self.assertIn('zero added',metadata['rgbd_latency_assumption'])

    def test_invalid_period_and_slow_native_camera_fail_before_output(self):
        SensorRecorder=recorder_class()
        source,model=native()
        with tempfile.TemporaryDirectory() as root:
            for period in (True,0,-1,2.0,'2'):
                with self.subTest(period=period),self.assertRaises(ValueError):
                    SensorRecorder(Path(root)/'sensor',source,model,rgbd_period_steps=period)
            for native_period in (.1,.03):
                slow,model=native(native_period)
                with self.assertRaises(ValueError):
                    SensorRecorder(Path(root)/'sensor',slow,model,rgbd_period_steps=2)
            for invalid in (-.01,math.nan):
                source,model=native(invalid)
                with self.assertRaises(ValueError):
                    SensorRecorder(Path(root)/'sensor',source,model)
            self.assertFalse((Path(root)/'sensor').exists())

    def test_world_consumes_only_the_recorders_new_frame(self):
        path=Path(__file__).resolve().parents[1]/'g1cap/arena_world.py'
        tree=ast.parse(path.read_text())
        world=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='ArenaWorld')
        capture=next(node for node in world.body if isinstance(node,ast.FunctionDef) and node.name=='capture')
        predicate=next(node.test for node in ast.walk(capture) if isinstance(node,ast.If)
                       and isinstance(node.test,ast.Compare)
                       and isinstance(node.test.left,ast.Subscript)
                       and isinstance(node.test.left.value,ast.Name)
                       and node.test.left.value.id=='camera')
        code=compile(ast.Expression(predicate),str(path),'eval')
        for frame_step,packet_step,expected in ((0,0,True),(0,1,False),(2,2,True),(2,3,False)):
            self.assertEqual(eval(code,{},dict(camera={'step':frame_step},packet={'step':packet_step})),expected)
