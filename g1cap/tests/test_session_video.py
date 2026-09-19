import unittest
from g1cap.session_video import video_schedule, checked_source


class SessionVideoTests(unittest.TestCase):
    def test_overview_only_accelerates_outside_executed_rounds(self):
        events = [{'type':'round_start','sim_time':10},
                  {'type':'round_end','sim_time':12},
                  {'type':'round_start','sim_time':30},
                  {'type':'terminal','sim_time':32}]
        schedule = video_schedule(0, 32, events, fps=10, idle_speed=8)
        for t, speed in schedule:
            if 10 <= t < 12 or 30 <= t < 32:
                self.assertEqual(speed, 1)
        self.assertTrue(any(speed == 8 for _, speed in schedule))
        self.assertEqual(schedule[-1][0], 32)

    def test_startup_failure_is_never_accelerated(self):
        schedule = video_schedule(0, 3, [], fps=10, idle_speed=8)
        self.assertTrue(all(speed in (0, 1) for _, speed in schedule))

    def test_source_must_match_executed_digest(self):
        import hashlib
        source = 'def run(robot, task):\n    return robot.observe()\n'
        self.assertEqual(checked_source(source, hashlib.sha256(source.encode()).hexdigest()), source)
        with self.assertRaises(ValueError): checked_source(source, 'incorrect')


class VideoCollectionTests(unittest.TestCase):
    def test_render_failure_still_collects_physics_and_source(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from g1cap.interactive import RemoteSession
        with tempfile.TemporaryDirectory() as folder:
            remote = RemoteSession('host', '/project', 0, Path(folder)/'remote',
                                   recipe={'task':'workstation_reach'})
            with patch.object(remote, '_ssh', side_effect=RuntimeError('renderer unavailable')):
                with patch('g1cap.interactive.subprocess.run') as run:
                    remote.close()
            self.assertEqual(run.call_args.args[0][0], 'scp')
            self.assertIn('renderer unavailable', json.loads((remote.output/'video_error.json').read_text())['error'])


class RecordingSectionsTests(unittest.TestCase):
    def test_task_starts_at_admission_without_dropping_editing_time(self):
        from g1cap.session_video import recording_sections
        frames=[{'sim_time':0,'no_support':False}, {'sim_time':3,'no_support':True},
                {'sim_time':4,'no_support':True}, {'sim_time':20,'no_support':True}]
        initial={'sim_time':4,'no_support':True,'support_enabled':False}
        self.assertEqual(recording_sections(frames,initial,20),
                         [('setup',0,4,1),('full',4,20,1),('overview',4,20,8)])

    def test_failed_initialization_has_no_task_video(self):
        from g1cap.session_video import recording_sections
        self.assertEqual(recording_sections([{'sim_time':0,'no_support':False}],None,5),
                         [('setup',0,5,1)])

    def test_support_during_task_cannot_be_hidden_by_trimming(self):
        from g1cap.session_video import recording_sections
        initial={'sim_time':4,'no_support':True,'support_enabled':False}
        for frames in ([{'sim_time':4,'no_support':False}],
                       [{'sim_time':4,'no_support':True},{'sim_time':5,'no_support':False}]):
            with self.assertRaises(ValueError): recording_sections(frames,initial,6)
        with self.assertRaises(ValueError):
            recording_sections([{'sim_time':4,'no_support':True}],dict(initial,support_enabled=True),6)
