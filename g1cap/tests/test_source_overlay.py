"""Deployment must include the package's static assets, independent of CWD."""
import json
from pathlib import Path
import tempfile
import unittest
import os
import subprocess
import sys


class SourceOverlayTests(unittest.TestCase):
    def test_simulator_child_uses_verified_overlay_despite_competing_cwd(self):
        from g1cap.source_overlay import stage_overlay
        from g1cap.sonic_runtime import simulator_launch_command
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();source=root/'source'/'g1cap';source.mkdir(parents=True)
            (source/'__init__.py').write_text('')
            from g1cap import source_overlay
            (source/'source_overlay.py').write_text(Path(source_overlay.__file__).read_text())
            (source/'sonic_sim.py').write_text("from pathlib import Path\nprint(Path(__file__).resolve())\n")
            overlay=root/'overlay';stage_overlay(overlay,package=source)
            competitor=root/'checkout'/'g1cap';competitor.mkdir(parents=True)
            (competitor/'__init__.py').write_text('')
            (competitor/'sonic_sim.py').write_text("print('WRONG SIMULATOR')\n")
            record=root/'simulator-source-record.json'
            command=simulator_launch_command(sys.executable,competitor.parent,
                                             overlay,record,[])
            output=subprocess.check_output(command,cwd=competitor.parent,text=True)
            self.assertEqual(output.strip(),str(overlay/'g1cap/sonic_sim.py'))
            evidence=json.loads(record.read_text())
            self.assertEqual(evidence['module_path'],str(overlay/'g1cap/sonic_sim.py'))

    def test_child_resolves_overlay_even_when_working_directory_has_competing_package(self):
        from g1cap.source_overlay import stage_overlay, overlay_command
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();source=root/'source'/'g1cap';source.mkdir(parents=True)
            (source/'__init__.py').write_text('')
            # Exercise the real bootstrap helper in the snapshotted package.
            from g1cap import source_overlay
            (source/'source_overlay.py').write_text(Path(source_overlay.__file__).read_text())
            (source/'probe.py').write_text("from pathlib import Path\nprint(Path(__file__).resolve())\n")
            overlay=root/'overlay';stage_overlay(overlay,package=source)
            competitor=root/'cwd'/'g1cap';competitor.mkdir(parents=True)
            (competitor/'__init__.py').write_text('')
            (competitor/'probe.py').write_text("print('WRONG PACKAGE')\n")
            env=dict(os.environ,PYTHONPATH=str(overlay))
            old=subprocess.check_output([sys.executable,'-m','g1cap.probe'],cwd=competitor.parent,env=env,text=True)
            self.assertIn('WRONG PACKAGE',old)
            record=root/'imports.json'
            command=overlay_command(sys.executable,'g1cap.probe',overlay,record)
            output=subprocess.check_output(command,cwd=competitor.parent,env=env,text=True)
            self.assertEqual(output.strip(),str(overlay/'g1cap/probe.py'))
            evidence=json.loads(record.read_text())
            self.assertEqual(evidence['module_path'],str(overlay/'g1cap/probe.py'))
            self.assertEqual(evidence['manifest'],json.loads((overlay/'source-manifest.json').read_text()))

    def test_snapshot_includes_assets_and_docs_but_not_bytecode(self):
        from g1cap.source_overlay import stage_overlay, verify_overlay
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); package=root/'source'/'g1cap'
            (package/'assets').mkdir(parents=True)
            (package/'tool_docs').mkdir()
            (package/'__pycache__').mkdir()
            for name,text in [('__init__.py','x=1'),('assets/robot.json','{}'),
                              ('tool_docs/contract.md','sensor contract'),('__pycache__/bad.pyc','cache')]:
                (package/name).write_text(text)
            overlay=root/'staged'
            manifest=stage_overlay(overlay,package=package)
            self.assertEqual(set(manifest),{'g1cap/__init__.py','g1cap/assets/robot.json','g1cap/tool_docs/contract.md'})
            self.assertEqual(verify_overlay(overlay),manifest)
            (package/'assets/robot.json').write_text('changed after snapshot')
            self.assertEqual((overlay/'g1cap/assets/robot.json').read_text(),'{}')
            (overlay/'g1cap/assets/robot.json').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'manifest'):
                verify_overlay(overlay)

    def test_default_package_comes_from_module_not_working_directory(self):
        import os
        from g1cap.source_overlay import stage_overlay
        with tempfile.TemporaryDirectory() as folder:
            old=Path.cwd()
            try:
                os.chdir(folder)
                manifest=stage_overlay(Path(folder)/'overlay')
            finally:os.chdir(old)
            self.assertIn('g1cap/arena_world.py',manifest)
            self.assertIn('g1cap/assets/arena_g1_rev1_0_kinematics.urdf',manifest)

    def test_added_file_or_missing_manifest_prevents_launch(self):
        from g1cap.source_overlay import stage_overlay, verify_overlay
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);package=root/'source'/'g1cap';package.mkdir(parents=True)
            (package/'__init__.py').write_text('')
            overlay=root/'staged';stage_overlay(overlay,package=package)
            (overlay/'g1cap/extra.py').write_text('unexpected')
            with self.assertRaisesRegex(ValueError,'manifest'):verify_overlay(overlay)
            (overlay/'g1cap/extra.py').unlink()
            (overlay/'source-manifest.json').unlink()
            with self.assertRaisesRegex(ValueError,'manifest'):verify_overlay(overlay)
