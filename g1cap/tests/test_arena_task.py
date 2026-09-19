import unittest
from g1cap.arena_task import ArenaBoxTask
from test_arena_control import observation


class ArenaTaskTests(unittest.TestCase):
    def setUp(self):
        self.score=ArenaBoxTask({'min':[.45,-.22,-.07],'max':[1.05,.58,-.03]})
    def sample(self,step,held):
        obs=observation(step*.02,.08 if held else 0.)
        obs.update(bilateral=held,supported=not held,box_pos=[.7,.18,.15 if held else .07],
                   hand_forces_N={'left':10. if held else 0.,'right':10. if held else 0.})
        self.score.update(obs)
        return obs
    def test_initial_table_support_cannot_pass(self):
        for i in range(150):self.sample(i,False)
        self.assertFalse(self.score.metrics()['success'])
    def test_hold_then_supported_release_and_late_fault(self):
        for i in range(51):self.sample(i,True)
        for i in range(51,110):self.sample(i,False)
        self.assertTrue(self.score.metrics()['success'])
        obs=self.sample(110,False);obs['robot_source_peak_N']=10.
        self.score.update(obs)
        self.assertFalse(self.score.metrics()['success'])
        self.assertEqual(self.score.metrics()['failure'],'forbidden_source_contact')
    def test_supported_but_outside_source_cannot_pass(self):
        for i in range(51):self.sample(i,True)
        for i in range(51,150):
            obs=observation(i*.02,0.)
            obs.update(box_pos=[1.2,.18,.07],bilateral=False,supported=True,hand_forces_N={'left':0.,'right':0.})
            self.score.update(obs)
        self.assertFalse(self.score.metrics()['success'])
