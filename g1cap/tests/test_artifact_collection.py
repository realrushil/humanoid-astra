"""A completed physics episode must survive slow or failed evidence transfer."""
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from g1cap.interactive import RemoteSession

class ArtifactCollectionTests(unittest.TestCase):
    def archive(self,name='physics/states.jsonl'):
        data=io.BytesIO()
        with tarfile.open(fileobj=data,mode='w:gz') as tar:
            content=b'{"step":0}\n';entry=tarfile.TarInfo(name);entry.size=len(content)
            tar.addfile(entry,io.BytesIO(content))
        return data.getvalue()
    def test_compressed_transfer_has_separate_bound_and_preserves_nested_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            remote=RemoteSession('testhost','/owned/root',0,Path(folder)/'local')
            archive=self.archive()
            def transfer(args,**kwargs):
                self.assertEqual(args[0],'ssh');self.assertIn('tar -czf -',args[-1])
                self.assertIn('--exclude=./physics/sensors',args[-1])
                self.assertEqual(kwargs['timeout'],600);kwargs['stdout'].write(archive)
                return subprocess.CompletedProcess(args,0)
            with patch('g1cap.interactive.subprocess.run',side_effect=transfer):remote._collect_artifacts()
            self.assertEqual((remote.output/'artifacts/physics/states.jsonl').read_text(),'{"step":0}\n')
            self.assertFalse((remote.output/'evidence.tar.gz').exists())   # extracted, so not kept
    def test_failed_transfer_retains_partial_archive_without_publishing_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            remote=RemoteSession('testhost','/owned/root',0,Path(folder)/'local')
            def transfer(args,**kwargs):
                kwargs['stdout'].write(b'partial');raise subprocess.TimeoutExpired(args,600)
            with patch('g1cap.interactive.subprocess.run',side_effect=transfer):
                with self.assertRaises(subprocess.TimeoutExpired):remote._collect_artifacts()
            self.assertFalse((remote.output/'artifacts').exists())
            self.assertEqual((remote.output/'evidence.tar.gz').read_bytes(),b'partial')
    def test_archive_cannot_write_outside_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            remote=RemoteSession('testhost','/owned/root',0,Path(folder)/'local')
            archive=self.archive('../escaped')
            def transfer(args,**kwargs):kwargs['stdout'].write(archive)
            with patch('g1cap.interactive.subprocess.run',side_effect=transfer):
                with self.assertRaises(tarfile.FilterError):remote._collect_artifacts()
            self.assertFalse((remote.output/'escaped').exists())
