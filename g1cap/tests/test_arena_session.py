import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from g1cap.arena_session import ArenaSession


class FakeControl:
    def __init__(self):
        self.owner=threading.get_ident()
        self.method=None;self.phase='idle';self.result=None;self.terminal_reason=None
        self.ticks=0;self.box_x=.6;self.never_complete=False
    def start(self,method,args,obs):
        assert threading.get_ident()==self.owner
        self.method=method;self.phase='hold';self.result=None;self.ticks=0
        return {'status':'running'}
    def update(self,obs):
        assert threading.get_ident()==self.owner
        self.ticks+=1
        if self.method and self.ticks>=3 and not self.never_complete:
            self.box_x+=.1
            self.method=None;self.phase='idle';self.result={'status':'completed'}
    def cancel(self,reason,obs):
        assert threading.get_ident()==self.owner
        self.method=None;self.phase='idle'
        self.result={'status':'cancelled','reason':reason}
        return self.result


class ArenaSessionTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.control=FakeControl();self.step=0
        def executor(source,task,round_id,dispatch,**options):
            seen=dispatch('observe',[],{},round_id)
            result=dispatch('hold_box',[],{'duration':1.},round_id)
            return dict(status='completed',seen=seen,tool_result=result)
        self.session=ArenaSession(self.control,{},Path(self.folder.name),executor=executor)
        self.advance()
    def tearDown(self):
        self.session.close();self.folder.cleanup()
    def advance(self):
        self.step+=1
        self.session.tick({'time':self.step*.02,'step':self.step,'box_pos':[self.control.box_x,0,0]})
    def finish_round(self,count):
        end=time.monotonic()+2
        while len(self.session.rounds)<count and time.monotonic()<end:
            self.advance();time.sleep(.001)
        self.assertEqual(len(self.session.rounds),count)
    def test_two_workers_share_state_and_intervening_time(self):
        self.session.submit('first');self.finish_round(1)
        first_end=self.session.rounds[0]['end_observation']['time']
        for _ in range(20):self.advance()
        self.session.submit('second');self.finish_round(2)
        second=self.session.rounds[1]
        self.assertGreater(second['start_observation']['time'],first_end+.3)
        self.assertAlmostEqual(second['execution']['seen']['box_pos'][0],.7)
        self.assertAlmostEqual(self.control.box_x,.8)
        self.assertEqual(self.session.rounds[0]['session_id'],second['session_id'])
    def test_supported_lift_uses_existing_request_queue(self):
        def executor(source,task,round_id,dispatch,**options):
            self.assertIn('lift_supported_box',options['tools'])
            return dispatch('lift_supported_box',[],{},round_id)
        self.session.executor=executor
        self.session.submit('supported lift');self.finish_round(1)
        self.assertEqual(self.session.rounds[0]['execution']['status'],'completed')
        self.assertEqual(self.session.tool_results[-1]['method'],'lift_supported_box')
    def test_wait_uses_existing_request_queue(self):
        def executor(source,task,round_id,dispatch,**options):
            self.assertIn('wait',options['tools'])
            return dispatch('wait',[3.],{},round_id)
        self.session.executor=executor
        self.session.submit('wait');self.finish_round(1)
        self.assertEqual(self.session.rounds[0]['execution']['status'],'completed')
        self.assertEqual(self.session.tool_results[-1]['method'],'wait')
    def test_transfer_methods_use_existing_worker_queue(self):
        def executor(source,task,round_id,dispatch,**options):
            for method,args in [('turn_with_box',[-.5]),('move_with_box',[.3]),('place_box',['destination'])]:
                self.assertIn(method,options['tools'])
                self.assertEqual(dispatch(method,args,{},round_id)['status'],'completed')
            return {'status':'completed'}
        self.session.executor=executor
        self.session.submit('transfer');self.finish_round(1)
        self.assertEqual([r['method'] for r in self.session.tool_results],['turn_with_box','move_with_box','place_box'])

    def test_surface_observation_is_copied_with_the_same_step(self):
        self.session.tick({'time':.04,'step':2,'physics_step':8,'surfaces':{'destination':{'supported':True}},
                           'box_bounds':{'min':[0,0,0],'max':[1,1,1]},'turn_clearance_m':.3})
        observed=self.session.observe();observed['surfaces']['destination']['supported']=False
        fresh=self.session.observe()
        self.assertTrue(fresh['surfaces']['destination']['supported'])
        self.assertEqual(fresh['step'],2);self.assertEqual(fresh['physics_step'],8)

    def test_overlapping_program_rejected(self):
        self.control.never_complete=True
        self.session.submit('first')
        with self.assertRaises(RuntimeError):self.session.submit('second')
    def test_finish_resolves_waiting_worker(self):
        self.control.never_complete=True
        self.session.submit('first')
        for _ in range(20):self.advance();time.sleep(.001)
        self.assertIsNotNone(self.session.pending)
        self.session.finish('episode_deadline')
        self.session.thread.join(timeout=1)
        self.assertFalse(self.session.thread.is_alive())
        self.assertEqual(self.session.rounds[0]['execution']['tool_result']['reason'],'episode_deadline')
    def test_fault_survives_new_submission(self):
        self.control.terminal_reason='forbidden_source_contact'
        self.control.result={'status':'failed','reason':'forbidden_source_contact'}
        self.advance()
        with self.assertRaises(RuntimeError):self.session.submit('new program')
        self.assertEqual(self.session.status()['terminal_reason'],'forbidden_source_contact')
        self.assertAlmostEqual(self.session.observe()['box_pos'][0],.6)
    def test_stale_mailbox_identity_fails_without_worker(self):
        import json
        path=Path(self.folder.name)/'inbox/round-00.json'
        path.write_text(json.dumps({'session_id':'old','source':'program'}))
        self.session.poll_submission()
        self.assertEqual(self.session.terminal_reason,'submission_error')
        self.assertIsNone(self.session.thread)


class ArenaRoundBudgetTests(unittest.TestCase):
    """Real worker/mailbox ownership with a controllable wall clock, no physics."""
    def setUp(self):
        self.now=100.
        self.clock_patch=patch('g1cap.arena_session.time',SimpleNamespace(monotonic=lambda:self.now))
        self.clock_patch.start()
        self.folder=tempfile.TemporaryDirectory()
        self.control=FakeControl()
        self.control.never_complete=True
        self.session=ArenaSession(self.control,{},self.folder.name,worker_timeout=600.,executor=self.one_request)
        self.step=0
        self.advance()

    def tearDown(self):
        self.session.close()
        self.folder.cleanup()
        self.clock_patch.stop()

    def one_request(self,source,task,round_id,dispatch,**options):
        return dispatch('hold_box',[1.],{},round_id)

    def advance(self):
        self.step+=1
        self.session.tick({'time':self.step*.02,'step':self.step,'box_pos':[self.control.box_x,0,0]})

    def await_request(self):
        # Only synchronize real threads here; elapsed budget time is entirely fake.
        end=time.monotonic()+1.
        while self.session.pending is None and time.monotonic()<end:
            self.advance()
            time.sleep(.001)
        self.assertIsNotNone(self.session.pending)

    def await_round(self):
        self.session.thread.join(timeout=.3)
        self.assertFalse(self.session.thread.is_alive(),'round must return when its wall budget expires')
        return self.session.rounds[-1]['execution']

    def complete_request(self):
        self.control.result={'status':'completed'}
        self.advance()

    def test_request_can_wait_more_than_180_seconds_within_round_budget(self):
        self.session.submit('long request')
        self.await_request()
        self.now=301.  # 201 wall seconds since submission, within the 600 s budget.
        self.session.thread.join(timeout=.12)
        self.assertTrue(self.session.thread.is_alive(),'valid request was cut off at 180 seconds')
        self.complete_request()
        self.assertEqual(self.await_round()['status'],'completed')

    def test_successive_requests_share_budget_including_time_between_calls(self):
        first_done=threading.Event()
        next_call=threading.Event()
        def two_requests(source,task,round_id,dispatch,**options):
            first=dispatch('hold_box',[1.],{},round_id)
            first_done.set()
            next_call.wait(1.)
            second=dispatch('hold_box',[1.],{},round_id)
            return dict(first=first,second=second)
        self.session.executor=two_requests
        self.session.submit('two requests')
        self.await_request()
        self.now=250.
        self.complete_request()
        self.assertTrue(first_done.wait(.3))
        self.now=650.  # Computation between calls also consumes this round's budget.
        next_call.set()
        self.await_request()
        self.now=701.
        result=self.await_round()
        self.assertEqual(result['first']['status'],'completed')
        self.assertEqual(result['second']['reason'],'request_wall_timeout')
        self.assertTrue(self.session.pending['cancelled'])
        self.advance()  # Only the physics owner performs cancellation.
        self.assertEqual(self.control.result['reason'],'worker_request_cancelled')

    def test_expired_round_does_not_admit_a_new_action(self):
        release=threading.Event()
        started=threading.Event()
        def delayed(source,task,round_id,dispatch,**options):
            started.set()
            release.wait(1.)
            return dispatch('hold_box',[1.],{},round_id)
        self.session.executor=delayed
        self.session.submit('delayed request')
        self.assertTrue(started.wait(.3))
        self.now=700.
        release.set()
        self.assertEqual(self.await_round()['reason'],'request_wall_timeout')
        self.advance()
        self.assertIsNone(self.control.method)
        self.assertIsNone(self.session.pending)
        self.assertEqual(self.session.tool_results,[])

    def test_new_revision_has_own_budget_without_resetting_episode(self):
        self.session.submit('first')
        self.await_request()
        self.now=701.
        self.assertEqual(self.await_round()['reason'],'request_wall_timeout')
        self.advance()
        self.control.box_x=.9  # Retained consequence of the previous physical request.
        self.advance()
        first_end=self.session.rounds[0]['end_observation']['time']
        self.now=800.
        self.session.submit('revision')
        self.await_request()
        self.now=1101.  # Beyond the old deadline, inside this revision's budget.
        self.session.thread.join(timeout=.12)
        self.assertTrue(self.session.thread.is_alive())
        self.complete_request()
        self.assertEqual(self.await_round()['status'],'completed')
        second=self.session.rounds[1]
        self.assertEqual(second['start_observation']['box_pos'][0],.9)
        self.assertGreater(second['start_observation']['time'],first_end)
        self.assertEqual(second['session_id'],self.session.rounds[0]['session_id'])

    def test_stale_round_and_closed_session_do_not_admit_actions(self):
        self.session.submit('first')
        self.await_request()
        stale=self.session.dispatch('hold_box',[1.],{},'old-round')
        self.assertEqual(stale,dict(status='rejected',reason='stale_round'))
        self.session.finish('closed')
        self.session.closed.set()
        self.assertEqual(self.await_round()['reason'],'closed')
        with self.assertRaises(RuntimeError):self.session.submit('revision')
        self.assertIsNone(self.session.pending)
        self.assertTrue(self.session.queue.empty())

    def test_worker_startup_delay_consumes_round_budget(self):
        def executor(source,task,round_id,dispatch,**options):
            return dict(status='completed',remaining_budget=options['wall_timeout'])
        self.session.executor=executor
        with patch('g1cap.arena_session.threading.Thread') as worker:
            self.session.submit('delayed startup')
            self.now=250.
            worker.call_args.kwargs['target']()
        self.session.thread=None
        self.assertEqual(self.session.rounds[0]['execution']['remaining_budget'],450.)

    def test_expired_before_worker_start_does_not_execute_program(self):
        def executor(*args,**kwargs):
            return dict(status='unexpected_execution')
        self.session.executor=executor
        with patch('g1cap.arena_session.threading.Thread') as worker:
            self.session.submit('expired startup')
            self.now=700.
            worker.call_args.kwargs['target']()
        self.session.thread=None
        self.assertEqual(self.session.rounds[0]['execution']['status'],'wall_timeout')
        self.assertTrue(self.session.queue.empty())


if __name__=='__main__':unittest.main()
