import math
import unittest
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

from g1cap.sonic_sim import SonicSimulation, support_release_ready, unsupported_standing


class StartupContactTests(unittest.TestCase):
    def test_precommand_wait_keeps_state_publication_live_and_never_rearms(self):
        sim=SonicSimulation.__new__(SonicSimulation)
        bridge=SimpleNamespace(low_cmd_lock=threading.Lock(),low_cmd_received=False,PublishLowState=Mock())
        sim.env=SimpleNamespace(unitree_bridge=bridge,prepare_obs=Mock(return_value={'time':0}),sim_step=Mock())
        sim.physics_started=False
        self.assertFalse(sim._advance())
        self.assertFalse(sim._advance())
        sim.env.sim_step.assert_not_called()
        self.assertEqual(bridge.PublishLowState.call_count,2)
        bridge.low_cmd_received=True
        self.assertTrue(sim._advance())
        bridge.low_cmd_received=False  # Lost commands must not freeze consequences.
        self.assertTrue(sim._advance())
        self.assertEqual(sim.env.sim_step.call_count,2)

    def test_declared_initial_pose_uses_named_joints_and_keeps_defaults_explicit(self):
        from g1cap.sonic_sim import initial_joint_targets
        self.assertEqual(initial_joint_targets('upstream_default','upstream_default'),{})
        targets=initial_joint_targets('travel','tucked_thumb')
        self.assertEqual(len(targets),31)
        self.assertEqual(targets['left_shoulder_roll_joint'],.6)
        self.assertEqual(targets['right_shoulder_roll_joint'],-.6)
        self.assertEqual(targets['left_hand_thumb_2_joint'],1.7)
        self.assertEqual(targets['right_hand_thumb_2_joint'],-1.7)
        self.assertFalse(any('hip' in name for name in targets))
        with self.assertRaises(ValueError):initial_joint_targets('unknown','tucked_thumb')

    def test_startup_contact_latches_transient_external_collision(self):
        from g1cap.sonic_sim import latch_environment_contact
        self_touch=dict(body1='hand',body2='hip',self_contact=True,distance=-.003)
        table=dict(body1='hand',body2='table',self_contact=False,distance=-.003)
        self.assertIsNone(latch_environment_contact(None,dict(sim_time=1,forbidden_contacts=[self_touch])))
        first=latch_environment_contact(None,dict(sim_time=2,forbidden_contacts=[table]))
        self.assertEqual(first,dict(sim_time=2,contact=table))
        self.assertEqual(latch_environment_contact(first,dict(sim_time=3,forbidden_contacts=[])),first)
        other=dict(sim_time=4,forbidden_contacts=[dict(table,body2='box')])
        self.assertEqual(latch_environment_contact(first,other),first)

    def test_runtime_rejects_startup_contact_even_after_contact_disappears(self):
        from g1cap.sonic_runtime import check_startup_contact
        check_startup_contact(dict(first_environment_contact=None))
        with self.assertRaisesRegex(ValueError,'history unavailable'):
            check_startup_contact({})
        with self.assertRaisesRegex(RuntimeError,'startup_environment_contact'):
            check_startup_contact(dict(first_environment_contact=dict(sim_time=2,contact=dict(body1='hand',body2='table')),
                                       forbidden_contacts=[]))

    def state(self, **updates):
        state = dict(ready=True, backend_ok=True, state_age_s=.01,
                     support_enabled=True, no_support=False, fallen=False,
                     foot_contacts={'left': True, 'right': True},
                     foot_normal_forces={'left':150., 'right':150.},
                     pelvis_position=[0., 0., .79], tilt=.02, planar_velocity=[.01, 0.], yaw_rate=.01)
        state.update(updates)
        return state

    def test_airborne_or_single_foot_cannot_release_support(self):
        self.assertTrue(support_release_ready(self.state()))
        for feet in ({'left': False, 'right': False}, {'left': True, 'right': False}, {}):
            self.assertFalse(support_release_ready(self.state(foot_contacts=feet)))

    def test_handover_does_not_require_balance_under_the_spring(self):
        self.assertTrue(support_release_ready(self.state(planar_velocity=[.2,0.],yaw_rate=.4)))
        for updates in (dict(tilt=math.radians(20)), dict(pelvis_position=[0.,0.,.3]),
                        dict(foot_normal_forces={'left':0.,'right':300.})):
            self.assertFalse(support_release_ready(self.state(**updates)))

    def test_task_readiness_requires_unsupported_stable_load_bearing_feet(self):
        state=self.state(support_enabled=False,no_support=True)
        self.assertTrue(unsupported_standing(state))
        for update in (dict(support_enabled=True),dict(no_support=False),
                       dict(planar_velocity=[.06,0.]),dict(yaw_rate=.11),
                       dict(foot_normal_forces={'left':0.,'right':300.}),
                       dict(state_age_s=.3),dict(backend_ok=False),dict(tilt=math.radians(16))):
            self.assertFalse(unsupported_standing(dict(state,**update)))

    def test_stale_unhealthy_or_already_released_state_is_not_ready(self):
        for updates in (dict(state_age_s=.3), dict(backend_ok=False), dict(ready=False),
                        dict(support_enabled=False), dict(fallen=True)):
            self.assertFalse(support_release_ready(self.state(**updates)))

    def test_release_rpc_rejects_airborne_state_without_scheduling(self):
        simulation = SonicSimulation.__new__(SonicSimulation)
        simulation.lock = threading.Lock()
        simulation.env = SimpleNamespace(elastic_band=SimpleNamespace(enable=True))
        simulation.release_at = None
        simulation.snapshot = self.state(foot_contacts={'left':False, 'right':False}, sim_time=1.,
                                         measured_monotonic=time.monotonic())
        with self.assertRaises(ValueError):
            simulation.request({'command':'release_support'})
        self.assertIsNone(simulation.release_at)
        self.assertTrue(simulation.env.elastic_band.enable)

    def test_release_rpc_only_schedules_supported_contact(self):
        simulation = SonicSimulation.__new__(SonicSimulation)
        simulation.lock = threading.Lock()
        simulation.env = SimpleNamespace(elastic_band=SimpleNamespace(enable=True))
        simulation.release_at = None
        simulation.snapshot = self.state(sim_time=1., measured_monotonic=time.monotonic(),
                                         planar_velocity=[.2,0.],yaw_rate=.4)
        simulation.request({'command':'release_support'})
        self.assertEqual(simulation.release_at, 1.)
        self.assertTrue(simulation.env.elastic_band.enable)

    def test_internal_snapshot_uses_capture_clock_for_age(self):
        snapshot = self.state(measured_monotonic=time.monotonic())
        snapshot.pop('state_age_s')  # Internal snapshots acquire age only at RPC read.
        self.assertTrue(support_release_ready(snapshot))
        snapshot['measured_monotonic'] -= 1.
        self.assertFalse(support_release_ready(snapshot))
