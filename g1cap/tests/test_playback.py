import json
from pathlib import Path
import tempfile
import unittest

from g1cap.playback import goal_markers, read_recording


def write_recording(path, frames):
    rows = [{'record_type': 'metadata', 'model_path': '/tmp/model.xml'}]
    rows.extend({'record_type': 'frame', **frame} for frame in frames)
    path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')


class PlaybackTests(unittest.TestCase):
    def test_goal_markers_use_frozen_episode_origin_and_yaw(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            (episode/'task.json').write_text(json.dumps({'target_position':[1.,0.,0.], 'waypoints':[]}))
            (episode/'manifest.json').write_text(json.dumps({'episode_frame':{
                'origin_xy':[2.,3.], 'origin_yaw':1.5707963267948966}}))
            points = goal_markers(episode)
            self.assertAlmostEqual(points[0][0], 2.)
            self.assertAlmostEqual(points[0][1], 4.)
            self.assertEqual(points[0][2], .025)

    def test_read_recording_filters_start_and_preserves_qpos(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'poses.jsonl'
            write_recording(path, [
                {'sim_time':0., 'qpos':[1,2], 'no_support':False},
                {'sim_time':.1, 'qpos':[3,4], 'no_support':True},
            ])
            metadata, frames = read_recording(path, start_time=.05)
            self.assertEqual(metadata['model_path'], '/tmp/model.xml')
            self.assertEqual([frame['sim_time'] for frame in frames], [.1])
            self.assertEqual(frames[0]['qpos'], [3.,4.])

    def test_read_recording_rejects_nonmonotonic_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'poses.jsonl'
            write_recording(path, [
                {'sim_time':.2, 'qpos':[1], 'no_support':False},
                {'sim_time':.1, 'qpos':[2], 'no_support':False},
            ])
            with self.assertRaisesRegex(ValueError, 'monotonic'):
                read_recording(path)
