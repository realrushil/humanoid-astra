"""Own one headless simulator and simulation-only SONIC controller episode."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .toolkit.sonic_backend import SimulatorClient, SonicBackend, SonicPublisher
from .runner import run_episode, write_json
from .tasks import make_task, task_from_dict
from .observations import ObservationConfig
from .sonic_sim import support_release_ready, unsupported_standing


def check_startup_contact(state):
    """Reject a scene disturbed before admission, including transient contacts."""
    if 'first_environment_contact' not in state:
        raise ValueError('startup contact history unavailable')
    contact = state['first_environment_contact']
    if contact is not None:
        raise RuntimeError('startup_environment_contact: '+json.dumps(contact, allow_nan=False))


class SonicRuntime:
    def __init__(self, root, output_dir, gpu, port=15556, startup_timeout=180., *, scene=None,
                 hand_posture='upstream_default',arm_posture='upstream_default'):
        self.root = Path(root).resolve()
        self.repo = self.root/'deps/GR00T-WholeBodyControl'
        self.output = Path(output_dir).resolve()
        self.gpu, self.port, self.startup_timeout = gpu, port, startup_timeout
        self.socket = f'/tmp/g1cap-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock'
        self.client = SimulatorClient(self.socket)
        self.processes, self.streams = [], []
        self.publisher = None
        self.backend = None
        self.scene = scene
        from .toolkit.sonic_backend import hand_reference,arm_reference
        self.hand_fields = hand_reference(hand_posture)
        self.hand_posture = hand_posture
        self.arm_fields = arm_reference(arm_posture)
        self.arm_posture = arm_posture

    def _spawn(self, command, log):
        stream = (self.output/log).open('w')
        self.streams.append(stream)
        process = subprocess.Popen(command, cwd=self.root, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        self.processes.append(process)
        return process

    def _healthy(self):
        for process in self.processes:
            if process.poll() is not None:
                raise RuntimeError(f'runtime process {process.pid} exited: {process.returncode}')

    def _wait(self, predicate, timeout):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            self._healthy()
            if predicate():
                return
            time.sleep(.1)
        raise TimeoutError('SONIC startup acknowledgement timed out')

    def start(self):
        self.output.mkdir(parents=True, exist_ok=False)
        try:
            model_args=[]
            if self.scene is not None:
                from .scene import write_scene_model
                import hashlib
                original=self.repo/'gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'
                model=write_scene_model(original,self.output/'scene.xml',self.scene)
                write_json(self.output/'scene.json',dict(self.scene.to_dict(),
                    model_sha256=hashlib.sha256(model.read_bytes()).hexdigest()))
                model_args=['--model',str(model)]
            self._spawn([str(self.root/'.venv-sonic/bin/python'), '-u', '-m', 'g1cap.sonic_sim',
                         '--repo', str(self.repo), '--socket', self.socket,
                         '--record', str(self.output/'poses.jsonl'),
                         '--arm-posture', self.arm_posture, '--hand-posture', self.hand_posture,
                         *model_args], 'simulator.log')
            self._wait(lambda: Path(self.socket).exists(), 30.)
            check_startup_contact(self.client.state())
            self.publisher = SonicPublisher(self.repo, self.socket, port=self.port,
                hand_posture=self.hand_posture,arm_posture=self.arm_posture)
            self.publisher.start()
            self._spawn(['bash', str(self.root/'launch-sonic-controller.sh'), str(self.root),
                         str(self.gpu), str(self.port)], 'controller.log')
            self.publisher.health_check = self._healthy
            log = self.output/'controller.log'
            self._wait(lambda: 'Init Done' in log.read_text(errors='replace'), self.startup_timeout)
            self.publisher.send_start(duration=1.)
            self._wait(lambda: 'transitioning to CONTROL state' in log.read_text(errors='replace'), 15.)
            self._wait(lambda: 'Reference motion name: planner_motion' in log.read_text(errors='replace'), 15.)
            # Transfer load once the active controller has both feet grounded.
            # The support spring is not a balance test: qualify standing below,
            # with all external assistance removed and no pose/time reset.
            stable_since = None
            deadline = time.monotonic()+15.
            with (self.output/'supported.jsonl').open('w') as trace:
                while time.monotonic() < deadline:
                    self._healthy()
                    state = self.client.state()
                    trace.write(json.dumps(state, allow_nan=False)+'\n')
                    trace.flush()
                    check_startup_contact(state)
                    settled = support_release_ready(state)
                    stable_since = (state['sim_time'] if stable_since is None else stable_since) if settled else None
                    if stable_since is not None and state['sim_time']-stable_since >= .1:
                        break
                    time.sleep(.05)
                else:
                    raise RuntimeError('supported robot did not establish load-bearing two-foot contact')
            self.client.request('release_support')
            self._wait(lambda: self.client.state()['no_support'], 2.)
            samples = []
            first = self.client.state()['sim_time']
            deadline = time.monotonic()+10.
            stable_since = None
            with (self.output/'standing.jsonl').open('w') as trace:
                while time.monotonic() < deadline:
                    self._healthy()
                    state = self.client.state()
                    trace.write(json.dumps(state, allow_nan=False)+'\n')
                    trace.flush()
                    check_startup_contact(state)
                    samples.append(state)
                    if not state['no_support'] or state['fallen'] or state['tilt'] > math.pi/4:
                        raise RuntimeError('unsupported standing failed')
                    settled = unsupported_standing(state)
                    stable_since = (state['sim_time'] if stable_since is None else stable_since) if settled else None
                    if stable_since is not None and state['sim_time']-stable_since >= 1. and state['sim_time']-first >= 3.:
                        break
                    time.sleep(.05)
                else:
                    raise RuntimeError('unsupported robot did not settle within startup budget')
            self.backend = SonicBackend(self.publisher)
            self.backend.metadata.update(
                startup_physics='declared initial posture; dynamics start at first body command and never pause afterward',
                startup_scene_check='first forbidden external contact is latched at every physics step and rejects admission',
                hand_posture=self.hand_posture,hand_targets=self.hand_fields,
                arm_posture=self.arm_posture,nominal_upper_body_targets=self.arm_fields,
                episode_frame={'origin_xy': self.backend.frame.origin_xy,
                               'origin_yaw': self.backend.frame.origin_yaw,
                               'origin_time': self.backend.frame.origin_time},
                robot_model='MuJoCo G1: 29 body joints plus two 7-joint Dex3 hands',
                checkpoint=json.loads((self.root/'setup-logs/models.json').read_text()),
                simulation_patch=(self.root/'setup-logs/sonic-simulation.patch').read_text(),
                gpu_index=self.gpu, startup_evidence=str(self.output/'standing.jsonl'))
            write_json(self.output/'startup.json', dict(
                status='unsupported_standing', samples=len(samples),
                standing_sim_seconds=samples[-1]['sim_time']-first,
                max_tilt=max(s['tilt'] for s in samples), final_state=samples[-1]))
            return self.backend
        except Exception as exc:
            write_json(self.output/'startup.json', {'status':'initialization_failed',
                       'error':f'{type(exc).__name__}: {exc}'})
            self.close()
            raise

    def close(self):
        if self.publisher is not None:
            self.publisher.idle()
            self.publisher.close()
        for process in reversed(self.processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=3.)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass
        for stream in self.streams:
            stream.close()
        # This unique socket belongs to this runtime, including crash leftovers.
        Path(self.socket).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--port', type=int, default=15556)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task-file', type=Path, help='explicit task JSON, instead of seed sampling')
    parser.add_argument('--observation-file', type=Path, help='assumed high-level observation errors; absent means nominal truth')
    args = parser.parse_args()
    source = args.policy.read_text()
    task = task_from_dict(json.loads(args.task_file.read_text())) if args.task_file else make_task('waypoint', args.seed)
    observation = ObservationConfig.from_dict(json.loads(args.observation_file.read_text())) if args.observation_file else None
    def interrupted(signum, frame):
        raise RuntimeError(f'runtime interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    runtime = SonicRuntime(args.root, args.out/'runtime', args.gpu, args.port)
    try:
        backend = runtime.start()
        result = run_episode(source, task, args.out/'episode', backend=backend,
                             wall_timeout=max(45., task.deadline*2), observation_config=observation)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result['execution_status'] == 'completed' else 1
    except Exception as exc:
        print(f'sonic_runtime: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    finally:
        runtime.close()


if __name__ == '__main__':
    raise SystemExit(main())
