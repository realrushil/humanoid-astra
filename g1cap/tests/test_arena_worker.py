import io
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from g1cap.arena_session import ArenaSession
from g1cap.execution import execute_policy
from test_arena_session import FakeControl


class ArenaWorkerTests(unittest.TestCase):
    def test_explicit_session_methods_dispatch(self):
        calls=[]
        def dispatch(method,args,kwargs,episode):
            calls.append((method,args,kwargs,episode))
            return {'status':'completed'}
        result=execute_policy('def run(robot, task):\n robot.retreat_with_box(distance_m=0.5)\n',
                              {},'arena-test',dispatch,tools={'observe','retreat_with_box'})
        self.assertEqual(result['status'],'completed',result)
        self.assertEqual(calls,[('retreat_with_box',[],{'distance_m':.5},'arena-test')])

    def test_unavailable_sonic_method_never_dispatches(self):
        calls=[]
        result=execute_policy('def run(robot, task):\n robot.walk_to([1., 0.])\n',
                              {},'arena-test',lambda *args:calls.append(args),tools={'observe'})
        self.assertEqual(result['status'],'policy_error',result)
        self.assertEqual(calls,[])

    def test_invalid_tool_declaration_rejected_before_worker(self):
        for methods in ['observe',[],['_private'],['two words']]:
            with self.subTest(methods=methods),self.assertRaises(ValueError):
                execute_policy('',{},'test',None,tools=methods)


class ArenaExecutorDeadlineTests(unittest.TestCase):
    def run_delayed_setup(self,setup_at,completed_at):
        """Real session/executor watchdog; replace only clock and operating-system I/O."""
        now=[100.]
        clock=SimpleNamespace(monotonic=lambda:now[0])
        spawned=[]
        data=iter([b'{"type":"done"}\n',b'',b''])
        class Pipe(io.BytesIO):
            def fileno(self):return 0
        class Process:
            pid=123456789
            returncode=0
            def __init__(self,*args,**kwargs):
                spawned.append(now[0])
                self.stdin,self.stdout,self.stderr=Pipe(),Pipe(),Pipe()
            def poll(self):return 0
            def wait(self,**kwargs):return 0
        class Selector:
            def __init__(self):self.mapping={}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def register(self,stream,event,label):
                self.mapping[stream]=SimpleNamespace(fileobj=stream,data=label)
            def unregister(self,stream):del self.mapping[stream]
            def get_map(self):return self.mapping
            def select(self,timeout):
                now[0]=min(completed_at,now[0]+timeout)
                if now[0]<completed_at:return []
                return [(next(iter(self.mapping.values())),None)]
        def sandbox_setup():
            now[0]=setup_at
            return ['fake-sandbox'],'fake-only'
        with tempfile.TemporaryDirectory() as folder:
            with (patch('g1cap.arena_session.time',clock),
                  patch('g1cap.execution.time',clock),
                  patch('g1cap.execution.sys.platform','linux'),
                  patch('g1cap.execution.sandbox_command',sandbox_setup),
                  patch('g1cap.execution.subprocess.Popen',Process),
                  patch('g1cap.execution.selectors.DefaultSelector',Selector),
                  patch('g1cap.execution.os.read',lambda *args:next(data)),
                  patch('g1cap.execution.os.killpg',lambda *args:None),
                  patch('g1cap.arena_session.threading.Thread') as worker):
                session=ArenaSession(FakeControl(),{},folder,worker_timeout=600.)
                try:
                    session.submit('def run(robot, task):\n return None\n')
                    worker.call_args.kwargs['target']()
                    return session.rounds[0]['execution'],now[0],spawned
                finally:
                    session.thread=None
                    session.close()

    def test_executor_setup_cannot_extend_computation_past_submit_deadline(self):
        result,finished_at,spawned=self.run_delayed_setup(setup_at=500.,completed_at=750.)
        self.assertEqual(result['status'],'wall_timeout',result)
        self.assertEqual(result['requests'],0)
        self.assertLessEqual(finished_at,700.000001)
        self.assertEqual(spawned,[500.])

    def test_executor_setup_exhausting_deadline_does_not_spawn(self):
        result,finished_at,spawned=self.run_delayed_setup(setup_at=700.,completed_at=750.)
        self.assertEqual(result['status'],'wall_timeout',result)
        self.assertEqual(spawned,[])
        self.assertEqual(finished_at,700.)

    def test_executor_can_complete_within_remaining_budget_after_setup(self):
        result,finished_at,spawned=self.run_delayed_setup(setup_at=500.,completed_at=650.)
        self.assertEqual(result['status'],'completed',result)
        self.assertEqual(finished_at,650.)
        self.assertEqual(spawned,[500.])


if __name__=='__main__':unittest.main()
