import unittest
from g1cap.arena_placement import PreparedPlacement
from test_arena_placement import obs,ACTION,DummyWrists


class Stage:
    def __init__(self,action,phase,result):
        self.last_action=list(action);self.phase=phase;self.result=result
    def command(self,row):
        self.last_action[0]=.321
        return list(self.last_action)
    def update(self,row):return self.result


class PreparedPlacementTests(unittest.TestCase):
    def test_current_state_and_last_command_are_transferred_between_stages(self):
        calls=[]
        def retract(row,action,surface):
            calls.append((row,list(action),surface));return Stage(action,'place_retract',('completed','ready'))
        c=PreparedPlacement(obs(50),ACTION,'destination',
            lambda r,a:Stage(a,'place_align',('completed','ready')),retract,lambda *a:DummyWrists())
        self.assertEqual(c.phase,'place_align')
        sent=c.command(obs(51));boundary=obs(52)
        self.assertIsNone(c.update(boundary))
        self.assertIs(calls[0][0],boundary)
        self.assertEqual(calls[0][1],sent);self.assertEqual(calls[0][2],'destination')
        self.assertEqual(c.phase,'place_retract')
        sent=c.command(obs(53));self.assertIsNone(c.update(obs(54)))
        self.assertEqual(c.phase,'place_lower');self.assertEqual(c.last_action,sent)

    def test_stage_failure_latches_without_starting_the_next_stage(self):
        def forbidden(*args):raise AssertionError('next stage started after a failed preparation')
        c=PreparedPlacement(obs(50),ACTION,'destination',
            lambda r,a:Stage(a,'place_align',('failed','preparation_force_limit')),forbidden,forbidden)
        self.assertEqual(c.phase,'place_align')
        sent=c.command(obs(51))
        self.assertEqual(c.update(obs(52)),('failed','preparation_force_limit'))
        self.assertEqual(c.update(obs(53)),('failed','preparation_force_limit'))
        self.assertEqual(c.command(obs(54)),sent)

    def test_stage_construction_rejection_is_failure_after_physical_action(self):
        def reject(*args):raise ValueError('no_feasible_clearance')
        c=PreparedPlacement(obs(50),ACTION,'destination',
            lambda r,a:Stage(a,'place_align',('completed','ready')),reject,lambda *a:None)
        self.assertEqual(c.phase,'place_align')
        sent=c.command(obs(51))
        self.assertEqual(c.update(obs(52)),('failed','no_feasible_clearance'))
        self.assertEqual(c.command(obs(53)),sent)


if __name__=='__main__':unittest.main()
