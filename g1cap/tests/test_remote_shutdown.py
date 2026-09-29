"""An exited SSH client does not prove the remote simulator stopped."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from g1cap.interactive import RemoteSession

class ShutdownTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.remote=RemoteSession('testhost','/owned/root',0,Path(self.temp.name)/'out',recipe={'backend':'arena'})
  self.remote.process=Mock();self.remote.process.poll.return_value=255
  self.remote._ssh=Mock();self.remote._json=Mock(return_value={'owned_groups_stopped':True})
  self.remote._collect_artifacts=Mock()
 def test_dead_ssh_client_still_requests_remote_stop(self):
  with patch('g1cap.arena_video.render_session'):self.remote.close()
  self.assertTrue(any('/stop' in x.args[0] for x in self.remote._ssh.call_args_list))
  self.remote._collect_artifacts.assert_called_once()
 def test_unreachable_shutdown_never_collects_changing_files(self):
  self.remote._ssh.side_effect=RuntimeError('Connection closed')
  with self.assertRaises(RuntimeError):self.remote.close()
  self.remote._collect_artifacts.assert_not_called()
  self.assertTrue((self.remote.output/'shutdown_error.json').exists())
 def test_cleanup_marker_is_required_before_collecting(self):
  self.remote._json.side_effect=[None,{'owned_groups_stopped':True}]
  with patch('g1cap.interactive.time.sleep'),patch('g1cap.arena_video.render_session'):self.remote.close()
  self.assertEqual(self.remote._json.call_count,2)
  self.remote._collect_artifacts.assert_called_once()
 def test_collection_is_refused_when_cleanup_cannot_be_verified(self):
  self.remote._json.side_effect=RuntimeError('Connection closed')
  with self.assertRaises(RuntimeError):self.remote.close()
  self.remote._collect_artifacts.assert_not_called()
if __name__=='__main__':unittest.main()

class ShutdownExitTests(unittest.TestCase):
 def test_cli_fails_when_cleanup_is_unverified(self):
  from g1cap.interactive import main
  with tempfile.TemporaryDirectory() as folder:
   args=['interactive','--ssh-host','testhost','--remote-root','/owned/root','--gpu','0',
         '--out',str(Path(folder)/'out'),'--policy',str(Path(folder)/'unused.py')]
   with patch('sys.argv',args),patch('g1cap.interactive.run_interactive',return_value={'error':None,'cleanup_error':'unverified shutdown'}):
    self.assertEqual(main(),1)
