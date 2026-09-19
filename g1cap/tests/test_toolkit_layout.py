import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g1cap.runner import run_episode
from g1cap.tasks import make_task


class ToolkitLayoutTests(unittest.TestCase):
    def test_toolkit_is_public_and_episode_hashes_cover_its_sources(self):
        self.assertIsNotNone(importlib.util.find_spec('g1cap.toolkit'))
        from g1cap.toolkit import Robot
        self.assertEqual(Robot.__module__, 'g1cap.toolkit.robot')
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)/'episode'
            with patch('g1cap.runner.execute_policy', return_value={
                    'status': 'completed', 'stderr': '', 'error': ''}):
                run_episode('def run(robot, task): pass', make_task(), output)
            hashes = json.loads((output/'manifest.json').read_text())['package_source_sha256']
            for name in ('robot.py', 'sonic_backend.py', 'sonic.py'):
                source = Path(__file__).resolve().parents[1]/'g1cap/toolkit'/name
                self.assertEqual(hashes['toolkit/'+name], hashlib.sha256(source.read_bytes()).hexdigest())
