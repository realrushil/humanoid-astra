"""Measured-coordinate, lease and local transport contracts; no policy claims."""
import math
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

from g1cap.sonic_backend import EpisodeFrame, LeasedPlanner, SimulatorClient, SonicBackend, SonicPublisher, SampleHistory


def measured(**changes):
    sample = dict(sim_time=10., sequence=100, pelvis_position=[5., 7., .8],
                  pelvis_yaw=math.pi/2, planar_velocity=[0., 0.], yaw_rate=0., tilt=0.,
                  right_wrist_position=[5., 6.8, .9], left_wrist_position=[5., 7.2, .9],
                  right_wrist_quaternion_wxyz=[math.sqrt(.5), 0., 0., math.sqrt(.5)],
                  state_age_s=0., ready=True, backend_ok=True, no_support=True,
                  dds_domain=1, dds_interface='lo')
    sample.update(changes)
    return sample


class EpisodeFrameTests(unittest.TestCase):
    def test_frozen_origin_transforms_pelvis_wrists_and_velocity_not_height(self):
        frame = EpisodeFrame(measured())
        state = frame.convert(measured(sim_time=10.5, sequence=101,
                                      pelvis_position=[5., 7.2, .75],
                                      planar_velocity=[0., .2],
                                      right_wrist_position=[5.2, 7.5, 1.]))
        self.assertAlmostEqual(state.sim_time, .5)
        self.assertAlmostEqual(state.pelvis_position[0], .2)
        self.assertAlmostEqual(state.pelvis_position[1], 0.)
        self.assertEqual(state.pelvis_position[2], .75)
        self.assertAlmostEqual(state.planar_velocity[0], .2)
        self.assertAlmostEqual(state.planar_velocity[1], 0.)
        for actual, expected in zip(state.right_wrist_position, [.5, -.2, 1.]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(state.wrist_orientation, [1., 0., 0., 0.]):
            self.assertAlmostEqual(actual, expected)

    def test_support_and_nonisolated_or_invalid_samples_cannot_start_episode(self):
        for changes in ({'no_support': False}, {'dds_domain': 0}, {'dds_interface': 'eth0'},
                        {'state_age_s': .3}, {'ready': False}, {'tilt': math.nan}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                EpisodeFrame(measured(**changes))

    def test_reset_time_cannot_be_converted_into_a_valid_episode(self):
        frame = EpisodeFrame(measured())
        with self.assertRaises(ValueError):
            frame.convert(measured(sim_time=9.))


class LeasedPlannerTests(unittest.TestCase):
    def test_fixed_hand_profile_is_sent_on_regular_and_shutdown_messages(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import inspect
        self.assertIn('hand_posture',inspect.signature(SonicPublisher).parameters)
        self.assertIn('arm_posture',inspect.signature(SonicPublisher).parameters)
        publisher=SonicPublisher('/unused','/unused.sock',hand_posture='tucked_thumb',arm_posture='travel')
        publisher.client=RecordedClient()
        messages=[]
        class Transport:
            def setsockopt(self,*args): pass
            def bind(self,*args): pass
            def close(self,**kwargs): pass
            def send(self,data): publisher.stopped.set()
        context=SimpleNamespace(socket=lambda *_:Transport(),term=lambda:None)
        zmq=SimpleNamespace(Context=lambda:context,PUB=1,LINGER=2,SNDHWM=3)
        def build(**fields): messages.append(fields);return b'planner'
        with patch.dict('sys.modules',zmq=zmq), patch('g1cap.toolkit.sonic_backend._load_builders',
                return_value=(lambda **kw:b'command',build)):
            publisher._run()
        self.assertIsNone(publisher.error)
        self.assertEqual(len(messages),2)
        for msg in messages:
            self.assertEqual(msg['left_hand_position'],[0.,0.,1.7,0.,0.,0.,0.])
            self.assertEqual(msg['right_hand_position'],[0.,0.,-1.7,0.,0.,0.,0.])
            self.assertEqual(msg['upper_body_position'],[0.,0.,0.,0.,0.,.6,-.6,0.,0.,.65,.65,0.,0.,0.,0.,0.,0.])
        self.assertEqual(messages[-1]['mode'],0)
        self.assertEqual(publisher.latest_state()['hand_posture'],'tucked_thumb')
        self.assertEqual(publisher.latest_state()['arm_posture'],'travel')
        reference=[.1]*17
        self.assertEqual(publisher._message_fields({'upper_body_position':reference})['upper_body_position'],reference)
        self.assertEqual(SonicPublisher('/unused','/unused.sock')._message_fields({'mode':0}),{'mode':0})
        with self.assertRaises(ValueError):
            SonicPublisher('/unused','/unused.sock',hand_posture='typo')

    def test_velocity_uses_measured_yaw_and_facing_integrates_independently(self):
        planner = LeasedPlanner(heading=0., now=0., lease_s=.25)
        planner.command(.2, 0., .4, now=0.)
        fields = planner.fields(measured_yaw=math.pi/2, now=.1)
        self.assertAlmostEqual(fields['movement'][0], 0.)
        self.assertAlmostEqual(fields['movement'][1], 1.)
        self.assertAlmostEqual(math.atan2(fields['facing'][1], fields['facing'][0]), .04)
        fields = planner.fields(measured_yaw=math.pi/2, now=.2)
        self.assertAlmostEqual(math.atan2(fields['facing'][1], fields['facing'][0]), .08)

    def test_expiry_idles_without_worker_and_does_not_integrate_unleased_gap(self):
        planner = LeasedPlanner(heading=0., now=0., lease_s=.25)
        planner.command(.2, 0., .4, now=0.)
        fields = planner.fields(measured_yaw=0., now=1.)
        self.assertEqual(fields['movement'], [0., 0., 0.])
        self.assertEqual(fields['mode'], 0)
        self.assertEqual(fields['speed'], -1.)
        self.assertAlmostEqual(math.atan2(fields['facing'][1], fields['facing'][0]), .1)
        self.assertEqual(planner.fields(measured_yaw=1., now=2.)['facing'], fields['facing'])

    def test_turn_in_place_and_out_of_bounds_rejection(self):
        planner = LeasedPlanner(heading=0., now=0.)
        planner.command(0., 0., .4, now=0.)
        fields = planner.fields(measured_yaw=0., now=.1)
        self.assertEqual(fields['mode'], 0)
        self.assertGreater(fields['facing'][1], 0.)
        for values in ((.3, .2, 0.), (0., 0., .5), (math.nan, 0., 0.), (True, 0., 0.)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                planner.command(*values, now=.1)

    def test_normalized_lateral_velocity_accepts_float_roundoff_not_real_excess(self):
        planner = LeasedPlanner(heading=0., now=0.)
        planner.command(0., .20000000000000004, 0., now=0.)
        fields = planner.fields(measured_yaw=0., now=.01)
        self.assertLessEqual(fields['speed'], .2)
        with self.assertRaises(ValueError):
            planner.command(0., .200001, 0., now=.02)


class LocalStateTransportTests(unittest.TestCase):
    def test_real_unix_socket_reads_measured_sample_and_rejects_error_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'sim.sock')
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(path)
                listener.listen()
                requests = []
                def reply():
                    for response in ({'ok': True, 'state': measured()},
                                     {'ok': False, 'error': 'physics failed'}):
                        connection, _ = listener.accept()
                        with connection, connection.makefile('rb') as reader:
                            requests.append(json.loads(reader.readline()))
                            connection.sendall(json.dumps(response).encode()+b'\n')
                thread = threading.Thread(target=reply)
                thread.start()
                client = SimulatorClient(path)
                self.assertEqual(client.state()['sequence'], 100)
                with self.assertRaisesRegex(RuntimeError, 'physics failed'):
                    client.state()
                thread.join(2.)
                self.assertFalse(thread.is_alive())
                self.assertEqual(requests, [{'command': 'state'}, {'command': 'state'}])


class RecordedStatePublisher:
    """External publisher boundary with a recorded sensor snapshot, no dynamics."""
    error = None
    def __init__(self):
        self.raw = measured()
    def latest_state(self):
        return self.raw.copy()
    def idle(self):
        pass


class MeasuredBackendTests(unittest.TestCase):
    def test_observe_refreshes_actual_sample_after_idle_without_fabricating_step(self):
        publisher = RecordedStatePublisher()
        backend = SonicBackend(publisher)
        publisher.raw = measured(sim_time=10.4, sequence=180,
                                 pelvis_position=[5., 7.08, .8])
        self.assertAlmostEqual(backend.state.sim_time, .4)
        self.assertAlmostEqual(backend.state.pelvis_position[0], .08)

    def test_step_uses_observed_elapsed_time_instead_of_requested_increment(self):
        publisher = RecordedStatePublisher()
        backend = SonicBackend(publisher, step_timeout=.03)
        publisher.raw = measured(sim_time=10.035, sequence=107,
                                 pelvis_position=[5., 7.07, .8])
        state = backend.step(.02)
        self.assertAlmostEqual(state.sim_time, .035)
        self.assertAlmostEqual(state.pelvis_position[0], .07)
        self.assertEqual(state.sequence, 107)
        self.assertEqual(state.pelvis_position[2], .8)

    def test_stalled_physics_fails_in_bounded_time_and_never_advances_pose(self):
        backend = SonicBackend(RecordedStatePublisher(), step_timeout=.03)
        initial = backend.state
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            backend.step(.02)
        self.assertLess(time.monotonic()-started, .2)
        self.assertEqual(backend.state.sim_time, initial.sim_time)
        self.assertEqual(backend.state.pelvis_position, initial.pelvis_position)
        self.assertFalse(backend.state.backend_ok)

    def test_reach_fails_explicitly_instead_of_interpolating_wrist(self):
        backend = SonicBackend(RecordedStatePublisher())
        before = backend.state.right_wrist_position
        with self.assertRaises(NotImplementedError):
            backend.reach([.3, -.2, .9])
        backend.hold_wrist()
        self.assertEqual(backend.state.right_wrist_position, before)


class RecordedClient:
    def __init__(self):
        self.raw = measured()
    def state(self):
        return self.raw.copy()


class ContinuousEvidenceTests(unittest.TestCase):
    def publisher(self):
        publisher = SonicPublisher('/unused-in-unit-test', '/unused-in-unit-test.sock')
        publisher.client = RecordedClient()
        publisher._capture_sample()
        return publisher

    def test_fall_then_upright_history_is_not_hidden_by_latest_observation(self):
        publisher = self.publisher()
        backend = SonicBackend(publisher)
        for seq, timestamp, height in ((104, 10.02, .1), (108, 10.04, .8)):
            publisher.client.raw = measured(sequence=seq, sim_time=timestamp,
                                            pelvis_position=[5., 7., height])
            publisher._capture_sample()
            publisher._capture_sample()  # repeated captures must not double-score
        self.assertEqual(backend.state.pelvis_position[2], .8)
        drained = backend.consume_samples()
        self.assertEqual([state.sequence for state in drained], [104, 108])
        self.assertEqual([state.pelvis_position[2] for state in drained], [.1, .8])
        self.assertEqual(backend.consume_samples(), [])

    def test_history_overflow_fails_only_if_unconsumed_evidence_was_dropped(self):
        history = SampleHistory(limit=2)
        for seq, timestamp in ((100, 10.), (104, 10.02), (108, 10.04)):
            history.append(measured(sequence=seq, sim_time=timestamp))
        self.assertEqual([raw['sequence'] for raw in history.after(100)], [104, 108])
        history.append(measured(sequence=112, sim_time=10.06))
        with self.assertRaisesRegex(RuntimeError, 'history'):
            history.after(100)

    def test_gap_fails_instead_of_granting_unobserved_dwell(self):
        publisher = self.publisher()
        backend = SonicBackend(publisher)
        publisher.client.raw = measured(sequence=140, sim_time=10.2)
        publisher._capture_sample()
        with self.assertRaisesRegex(RuntimeError, 'gap'):
            backend.consume_samples()
        self.assertFalse(backend.state.backend_ok)

    def test_controller_failure_latches_even_when_simulator_keeps_publishing(self):
        publisher = self.publisher()
        backend = SonicBackend(publisher)
        publisher.client.raw = measured(sequence=104, sim_time=10.02)
        def failed_controller():
            raise RuntimeError('controller exited')
        publisher.health_check = failed_controller
        with self.assertRaisesRegex(RuntimeError, 'controller exited'):
            publisher._capture_sample()
        self.assertIn('controller exited', publisher.error)
        self.assertFalse(backend.state.backend_ok)
        with self.assertRaisesRegex(RuntimeError, 'controller exited'):
            backend.consume_samples()


if __name__ == '__main__':
    unittest.main()
