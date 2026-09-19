import hashlib,json,tempfile,unittest
from pathlib import Path
from g1cap.arena_video import load_evidence
class VideoEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.root=Path(self.folder.name)
        (self.root/'physics').mkdir();(self.root/'session/round-00').mkdir(parents=True)
        self.source='def run(robot, task):\n    robot.wait(1)\n'
        (self.root/'session/round-00/policy.py').write_text(self.source)
        self.write('session/round-00/result.json',{'source_sha256':hashlib.sha256(self.source.encode()).hexdigest()})
        self.write('session/summary.json',{'task':{'task_kind':'source_return'},'metrics':{'success':False},'terminal_reason':'round_limit'})
        (self.root/'session/tool-trace.jsonl').write_text('')
        (self.root/'physics/states.jsonl').write_text('\n'.join(json.dumps({'step':i,'time':i*.02,'phase':'idle'}) for i in range(2))+'\n')
        (self.root/'physics/actions.jsonl').write_text(json.dumps({'step':1})+'\n')
    def tearDown(self):self.folder.cleanup()
    def write(self,name,value):(self.root/name).write_text(json.dumps(value))
    def test_source_and_physics_correspondence_is_required(self):
        evidence=load_evidence(self.root)
        self.assertEqual(len(evidence['states']),2);self.assertEqual(len(evidence['sources']),1)
        (self.root/'session/round-00/policy.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'source'):load_evidence(self.root)
    def test_missing_or_reordered_frames_are_rejected(self):
        (self.root/'physics/states.jsonl').write_text(json.dumps({'step':0,'time':0})+'\n'+json.dumps({'step':2,'time':.04})+'\n')
        with self.assertRaisesRegex(ValueError,'sequence'):load_evidence(self.root)
