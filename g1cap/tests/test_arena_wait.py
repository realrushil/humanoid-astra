"""A bounded physical dwell promises elapsed time, not grasp success."""
import math,unittest
from g1cap.arena_control import BoxControl,METHODS
from test_arena_control import observation

class WaitTests(unittest.TestCase):
    def setUp(self):
        self.action=[.03]*43+[.1,.02,.2,.75,.01,.02,.03]
        self.control=BoxControl(self.action,lambda obs:None,lambda *a:None)
        self.control.update(observation())

    def test_wait_runs_without_a_held_box_and_preserves_references(self):
        self.assertIn('wait',METHODS)
        row=observation();row.update(bilateral=False,clearance=0.,supported=True)
        result=self.control.start('wait',{'duration':.1},row)
        self.assertEqual(result['status'],'running')
        for i in range(1,6):
            sent=self.control.command(row)
            self.assertEqual(sent[43:46],[0.,0.,0.])
            self.assertEqual(sent[:43],self.control._action(self.action)[:43])
            self.assertEqual(sent[46:],self.control._action(self.action)[46:])
            row=observation(i*.02);row.update(bilateral=False,clearance=0.,supported=True)
            self.control.update(row)
        self.assertEqual(self.control.result['reason'],'dwell_elapsed')
        self.assertFalse(self.control.result['observation']['bilateral'])

    def test_invalid_duration_never_starts_a_wait(self):
        for duration in [0,-1,5.01,True,float('nan')]:
            result=self.control.start('wait',{'duration':duration},observation())
            self.assertEqual(result['status'],'rejected')
            self.assertIsNone(self.control.method)

    def test_wait_does_not_mask_collision_or_time_sequence_failure(self):
        self.assertEqual(self.control.start('wait',{'duration':3.},observation())['status'],'running')
        row=observation(.02);row['robot_source_peak_N']=6.
        self.control.update(row)
        self.assertEqual(self.control.result['reason'],'forbidden_source_contact')
        self.assertEqual(self.control.start('wait',{},row)['reason'],'episode_failed')

if __name__=='__main__':unittest.main()
