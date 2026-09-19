"""Live, simulation-only SONIC planner transport and measured-state backend.

The publisher's independent wall-clock lease expires to an idle planner command.
No policy, dynamics, pose interpolation or Cartesian reach is implemented here.
Heavy dependencies are loaded only when a real publisher is started.
"""
import copy
from collections import deque
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import socket
import subprocess
import threading
import time

from ..models import State, finite_number, wrap_yaw

PINNED_REVISION = '087f9ac01d46f6d8e4d0b73c01ae64799f292a38'


def hand_reference(posture):
    """Fixed, opt-in Dex3 posture, in upstream seven-joint wire order/radians.

    Omitting targets makes SONIC supply fully closed fingers. Tucked thumb keeps
    thumb joints 0/1 at zero and distal thumb at +/-1.7; other fingers extend.
    This is a development travel posture, not a grasp or hardware qualification.
    """
    if posture=='upstream_default':
        return {}
    if posture=='tucked_thumb':
        return dict(left_hand_position=[0.,0.,1.7,0.,0.,0.,0.],
                    right_hand_position=[0.,0.,-1.7,0.,0.,0.,0.])
    raise ValueError('hand_posture must be upstream_default or tucked_thumb')


def arm_reference(posture):
    """Optional nominal travel posture, 17 named waist/arm radians in wire order.

    Shoulder roll +/-0.6 moves arms away from hips; elbows flex 0.65. Waist and
    other arm joints target zero. This is a controller reference, not a pose
    rewrite. Explicit stationary reach references take precedence over it.
    """
    if posture=='upstream_default':
        return {}
    if posture!='travel':
        raise ValueError('arm_posture must be upstream_default or travel')
    from .arm_planner import UPPER_BODY
    joints=dict.fromkeys(UPPER_BODY,0.)
    joints.update(left_shoulder_roll_joint=.6,right_shoulder_roll_joint=-.6,
                  left_elbow_joint=.65,right_elbow_joint=.65)
    return dict(upper_body_position=[joints[name] for name in UPPER_BODY],
                upper_body_velocity=[0.]*17)


def _vector(values, size):
    if not isinstance(values, (list, tuple)) or len(values) != size or not all(
            finite_number(value) for value in values):
        raise ValueError(f'expected finite {size}-vector')
    return tuple(float(value) for value in values)


def _checked_sample(raw, require_unsupported=True):
    if not isinstance(raw, dict) or raw.get('dds_domain') != 1 or raw.get('dds_interface') != 'lo':
        raise ValueError('state must come from the isolated lo/domain 1 simulator')
    if raw.get('ready') is not True or raw.get('backend_ok') is not True:
        raise ValueError('simulator state is not ready or healthy')
    if require_unsupported and raw.get('no_support') is not True:
        raise ValueError('episode requires measured support removal')
    for name in ('sim_time', 'pelvis_yaw', 'yaw_rate', 'tilt', 'state_age_s'):
        if not finite_number(raw.get(name)):
            raise ValueError(f'invalid measured {name}')
    if raw['sim_time'] < 0 or not 0 <= raw['state_age_s'] <= .25:
        raise ValueError('simulator state is stale or has invalid time')
    if not isinstance(raw.get('sequence'), int) or isinstance(raw['sequence'], bool) or raw['sequence'] < 0:
        raise ValueError('invalid measured sequence')
    for name, size in (('pelvis_position', 3), ('planar_velocity', 2),
                       ('left_wrist_position', 3), ('right_wrist_position', 3),
                       ('right_wrist_quaternion_wxyz', 4)):
        _vector(raw.get(name), size)
    if math.hypot(*raw['right_wrist_quaternion_wxyz']) < 1e-12:
        raise ValueError('invalid measured wrist quaternion')
    return raw


class EpisodeFrame:
    """Freeze a horizontal world origin; retain absolute simulator heights."""

    def __init__(self, sample):
        raw = _checked_sample(sample)
        self.origin_xy = tuple(raw['pelvis_position'][:2])
        self.origin_yaw = float(raw['pelvis_yaw'])
        self.origin_time = float(raw['sim_time'])
        self.c, self.s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)

    def _xy(self, x, y):
        return (self.c*x + self.s*y, -self.s*x + self.c*y)

    def _position(self, position):
        x, y = self._xy(position[0]-self.origin_xy[0], position[1]-self.origin_xy[1])
        return (x, y, position[2])

    def convert(self, sample):
        raw = _checked_sample(sample)
        elapsed = raw['sim_time'] - self.origin_time
        if elapsed < 0:
            raise ValueError('simulator time reset during episode')
        q = raw['right_wrist_quaternion_wxyz']
        norm = math.hypot(*q)
        w, x, y, z = (value/norm for value in q)
        c, s = math.cos(self.origin_yaw/2), math.sin(self.origin_yaw/2)
        # q_episode_world * q_world_wrist, both WXYZ.
        orientation = (c*w+s*z, c*x+s*y, c*y-s*x, c*z-s*w)
        return State(
            sim_time=elapsed, sequence=raw['sequence'],
            pelvis_position=self._position(raw['pelvis_position']),
            pelvis_yaw=wrap_yaw(raw['pelvis_yaw']-self.origin_yaw),
            planar_velocity=self._xy(*raw['planar_velocity']),
            yaw_rate=raw['yaw_rate'], tilt=raw['tilt'],
            right_wrist_position=self._position(raw['right_wrist_position']),
            left_wrist_position=self._position(raw['left_wrist_position']),
            wrist_orientation=orientation, state_age_s=raw['state_age_s'],
            ready=True, backend_ok=True,
        )


class LeasedPlanner:
    """Pure command lease state, driven by an external monotonic clock."""

    def __init__(self, heading, now, lease_s=.25):
        if not all(finite_number(v) for v in (heading, now, lease_s)) or not 0 < lease_s <= .25:
            raise ValueError('finite heading/time and lease in (0, .25] required')
        self.heading = wrap_yaw(heading)
        self.last_time = now
        self.expires = now
        self.lease_s = lease_s
        self.velocity = (0., 0., 0.)
        self.stationary_fields = None

    def _advance(self, now):
        if not finite_number(now) or now < self.last_time:
            raise ValueError('lease clock must be finite and monotonic')
        active_dt = max(0., min(now, self.expires)-self.last_time)
        self.heading = wrap_yaw(self.heading + self.velocity[2]*active_dt)
        self.last_time = now
        if now >= self.expires:
            self.velocity = (0., 0., 0.)
            self.stationary_fields = None

    def command(self, vx, vy, yaw_rate, now):
        if not all(finite_number(v) for v in (vx, vy, yaw_rate)) or (
                math.hypot(vx, vy) > .30+1e-12 or abs(vy) > .20+1e-12 or abs(yaw_rate) > .40):
            raise ValueError('command exceeds bounded body velocity limits')
        self._advance(now)
        # Scaling a vector can produce 0.20000000000000004 at the 0.20 limit.
        self.velocity = (vx, max(-.20, min(.20, vy)), yaw_rate)
        self.stationary_fields = None
        self.expires = now + self.lease_s

    def stationary(self, mode, height, positions, velocities, now):
        """One owner supplies posture and optional 17-joint waist/arm references.

        Mode 4 height is a clip request in metres, not measured height control.
        [0.60, 0.85] is a conservative development input bound, not calibration.
        Expiry deliberately retains the existing idle fallback.
        """
        if (type(mode) is not int or mode not in (0, 4) or not finite_number(height)
                or (mode == 0 and height != -1.) or (mode == 4 and not .60 <= height <= .85)):
            raise ValueError('stationary mode must be idle/-1 or squat/height in [0.60, 0.85] m')
        fields = dict(mode=mode, height=float(height))
        if positions is not None:
            fields['upper_body_position'] = list(_vector(positions, 17))
        if velocities is not None:
            if positions is None:
                raise ValueError('joint velocities require joint positions')
            fields['upper_body_velocity'] = list(_vector(velocities, 17))
        self._advance(now)
        self.velocity = (0., 0., 0.)
        self.stationary_fields = fields
        self.expires = now + self.lease_s

    def motion(self, mode, velocity_world, facing_world, height, positions, now):
        """Leased reference fields; world movement/facing, no goal controller."""
        velocity=_vector(velocity_world,2)
        speed=math.hypot(*velocity)
        if type(mode) is not int or mode not in (0,1,4) or not finite_number(facing_world):
            raise ValueError('invalid motion mode or facing')
        if speed>.20+1e-12 or (mode!=1 and speed!=0.):
            raise ValueError('world velocity <=0.20 m/s and only in walk mode')
        if not finite_number(height) or (mode==4 and not .60<=height<=.85) or (mode!=4 and height!=-1.):
            raise ValueError('height [.60,.85] only in squat mode; otherwise -1')
        fields=dict(mode=mode,movement=[velocity[0]/speed,velocity[1]/speed,0.] if speed else [0.,0.,0.],
                    speed=speed if mode==1 else -1.,height=height,
                    facing=[math.cos(facing_world),math.sin(facing_world),0.])
        if positions is not None:
            fields.update(upper_body_position=list(_vector(positions,17)),upper_body_velocity=[0.]*17)
        self._advance(now)
        self.velocity=(0.,0.,0.)
        self.heading=wrap_yaw(facing_world)
        self.stationary_fields=fields
        self.expires=now+self.lease_s

    def idle(self, now):
        self._advance(now)
        self.velocity = (0., 0., 0.)
        self.stationary_fields = None
        self.expires = now

    def fields(self, measured_yaw, now):
        if not finite_number(measured_yaw):
            raise ValueError('finite measured heading required')
        self._advance(now)
        vx, vy, _ = self.velocity
        speed = math.hypot(vx, vy)
        movement = [0., 0., 0.]
        if speed:
            c, s = math.cos(measured_yaw), math.sin(measured_yaw)
            movement = [(c*vx-s*vy)/speed, (s*vx+c*vy)/speed, 0.]
        fields = dict(mode=1 if speed else 0, movement=movement,
                    facing=[math.cos(self.heading), math.sin(self.heading), 0.],
                    speed=speed if speed else -1., height=-1.)
        if self.stationary_fields is not None:
            fields.update(self.stationary_fields)
        return fields


class SimulatorClient:
    """Bounded requests to the simulation's local Unix socket."""

    def __init__(self, socket_path, timeout=.2):
        if not finite_number(timeout) or not 0 < timeout <= 2.:
            raise ValueError('socket timeout must be in (0, 2] seconds')
        self.socket_path = str(socket_path)
        self.timeout = timeout

    def request(self, command='state', **fields):
        payload = json.dumps(dict(command=command, **fields), allow_nan=False).encode()+b'\n'
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.socket_path)
            connection.sendall(payload)
            with connection.makefile('rb') as reader:
                line = reader.readline(262145)
        if len(line) > 262144 or not line.endswith(b'\n'):
            raise RuntimeError('invalid or oversized simulator reply')
        response = json.loads(line)
        if not isinstance(response, dict) or response.get('ok') is not True:
            message = response.get('error', 'invalid reply') if isinstance(response, dict) else 'invalid reply'
            raise RuntimeError(f'simulator: {message}')
        if not isinstance(response.get('state'), dict):
            raise RuntimeError('simulator reply has no measured state')
        return response['state']

    def state(self):
        return self.request('state')


def _load_builders(repo):
    repo = Path(repo).resolve()
    revision = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if revision != PINNED_REVISION:
        raise RuntimeError(f'SONIC revision must be {PINNED_REVISION}, found {revision}')
    path = repo/'gear_sonic/utils/teleop/zmq/zmq_planner_sender.py'
    spec = importlib.util.spec_from_file_location('g1cap_upstream_sonic_wire', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_command_message, module.build_planner_message


class SampleHistory:
    """Ordered captured samples; callers own locking and independent cursors."""

    def __init__(self, limit=12000):
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 12000:
            raise ValueError('history limit must be an integer in [1, 12000]')
        self.limit = limit
        self.samples = deque()
        self.dropped_through = -1

    def append(self, raw):
        if self.samples:
            previous = self.samples[-1]
            if raw['sequence'] < previous['sequence'] or raw['sim_time'] < previous['sim_time']:
                raise RuntimeError('simulator reset while collecting history')
            if raw['sequence'] == previous['sequence']:
                if raw['sim_time'] != previous['sim_time']:
                    raise RuntimeError('same simulator sequence has inconsistent time')
                return
            if raw['sim_time'] <= previous['sim_time']:
                raise RuntimeError('new simulator sequence must advance measured time')
        if len(self.samples) == self.limit:
            self.dropped_through = self.samples.popleft()['sequence']
        self.samples.append(copy.deepcopy(raw))

    def after(self, sequence):
        if sequence < self.dropped_through:
            raise RuntimeError('unconsumed simulator history overflowed; evidence was dropped')
        return [copy.deepcopy(raw) for raw in self.samples if raw['sequence'] > sequence]


class SonicPublisher:
    """Independent 50 Hz publisher with 250 ms maximum command leases.

    Start this before the C++ subscriber. The same publisher can idle throughout
    supported initialization, then serve a backend after support is released.
    All ZMQ operations run on its owner thread, including bind and shutdown.
    """

    def __init__(self, repo, socket_path, port=5556, lease_s=.25, *, hand_posture='upstream_default',
                 arm_posture='upstream_default'):
        if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535:
            raise ValueError('loopback port must be an integer in [1024, 65535]')
        # Validate lease without loading transport dependencies.
        LeasedPlanner(0., 0., lease_s)
        self.fixed_hand_fields=hand_reference(hand_posture)
        self.hand_posture=hand_posture
        self.fixed_arm_fields=arm_reference(arm_posture)
        self.arm_posture=arm_posture
        self.repo = str(repo)
        self.client = SimulatorClient(socket_path)
        self.endpoint = f'tcp://127.0.0.1:{port}'
        self.lease_s = lease_s
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.started = threading.Event()
        self.thread = None
        self.error = None
        self._raw = None
        self._received_at = None
        self._planner = None
        self._start_until = 0.
        self.published_messages = 0
        self.history = SampleHistory()
        self.health_check = None

    def start(self, timeout=5.):
        if self.thread is not None:
            raise RuntimeError('publisher already started')
        self.thread = threading.Thread(target=self._run, name='sonic-planner-publisher', daemon=True)
        self.thread.start()
        if not self.started.wait(timeout):
            self.close()
            raise TimeoutError('SONIC publisher did not start')
        if self.error:
            raise RuntimeError(self.error)
        return self

    def latest_state(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if self._raw is None:
                raise RuntimeError('publisher has no simulator sample; call start first')
            raw = copy.deepcopy(self._raw)
            raw['state_age_s'] += max(0., time.monotonic()-self._received_at)
            return raw

    def read_state(self):
        return self.latest_state()

    def samples_after(self, sequence):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            return self.history.after(sequence)

    def _capture_sample(self):
        """Check controller health before accepting even a fresh physics sample."""
        try:
            if self.health_check is not None:
                if self.health_check() is False:
                    raise RuntimeError('controller health check failed')
            raw = _checked_sample(self.client.state(), require_unsupported=False)
            raw['hand_posture']=self.hand_posture  # Command profile; measured joints remain separate.
            raw['arm_posture']=self.arm_posture
            with self.lock:
                if self.error:
                    raise RuntimeError(self.error)
                self.history.append(raw)
                self._raw, self._received_at = copy.deepcopy(raw), time.monotonic()
            return raw
        except Exception as exc:
            with self.lock:
                self.error = self.error or f'{type(exc).__name__}: {exc}'
            self.stopped.set()
            raise

    def set_velocity(self, vx, vy, yaw_rate):
        with self.lock:
            if self.error or self.stopped.is_set() or self._planner is None:
                raise RuntimeError(self.error or 'publisher is not running')
            self._planner.command(vx, vy, yaw_rate, time.monotonic())

    def idle(self):
        with self.lock:
            if self._planner is not None:
                self._planner.idle(time.monotonic())

    def set_stationary(self, mode=0, height=-1., positions=None, velocities=None):
        with self.lock:
            if self.error or self.stopped.is_set() or self._planner is None:
                raise RuntimeError(self.error or 'publisher is not running')
            self._planner.stationary(mode, height, positions, velocities, time.monotonic())

    def set_motion(self, **fields):
        with self.lock:
            if self.error or self.stopped.is_set() or self._planner is None:
                raise RuntimeError(self.error or 'publisher is not running')
            self._planner.motion(**fields,now=time.monotonic())

    def send_start(self, duration=.5):
        """Repeat upstream start messages through PUB/SUB startup; not an ack."""
        if not finite_number(duration) or not 0 < duration <= 2.:
            raise ValueError('start publication duration must be in (0, 2] seconds')
        with self.lock:
            if self.error or self.stopped.is_set() or self._planner is None:
                raise RuntimeError(self.error or 'publisher is not running')
            self._start_until = time.monotonic()+duration

    def _run(self):
        context = transport = None
        fields = None
        build_planner = None
        try:
            import zmq
            build_command, build_planner = _load_builders(self.repo)
            context = zmq.Context()
            transport = context.socket(zmq.PUB)
            transport.setsockopt(zmq.LINGER, 0)
            transport.setsockopt(zmq.SNDHWM, 3)
            transport.bind(self.endpoint)
            while not self.stopped.is_set():
                tick_start = time.monotonic()
                raw = self._capture_sample()
                with self.lock:
                    now = time.monotonic()
                    if self._planner is None:
                        self._planner = LeasedPlanner(raw['pelvis_yaw'], now, self.lease_s)
                    fields = self._planner.fields(raw['pelvis_yaw'], now)
                    send_start = now < self._start_until
                transport.send(build_planner(**self._message_fields(fields)))
                if send_start:
                    transport.send(build_command(start=True, stop=False, planner=True))
                self.published_messages += 1
                self.started.set()
                self.stopped.wait(max(0., .02-(time.monotonic()-tick_start)))
        except Exception as exc:
            with self.lock:
                self.error = f'{type(exc).__name__}: {exc}'
            self.stopped.set()
        finally:
            # Best effort immediate idle; upstream's own planner timeout is an
            # additional fallback if the socket or this process has failed.
            if transport is not None and build_planner is not None and fields is not None:
                try:
                    idle = dict(mode=0, movement=[0., 0., 0.], speed=-1.,
                                height=-1., facing=fields['facing'])
                    transport.send(build_planner(**self._message_fields(idle)))
                except Exception:
                    pass
            if transport is not None:
                transport.close(linger=0)
            if context is not None:
                context.term()
            self.started.set()

    def _message_fields(self,fields):
        # A task's local arm trajectory overrides the nominal travel reference.
        # Idle/lease expiry restores travel posture; it does not close the hands.
        return {**self.fixed_arm_fields,**fields,**self.fixed_hand_fields}

    def close(self):
        self.idle()
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=3.)


class SonicBackend:
    """Robot facade backend using only measured MuJoCo observations."""
    name = 'sonic'
    supported_operations = frozenset(('observe', 'move_base', 'stop', 'hold', 'walk_to', 'turn_to'))
    metadata = dict(physics=True, controller='SONIC', version='experimental',
                    qualification='experimental_unqualified', upstream_revision=PINNED_REVISION,
                    wrist_frames={'right': 'right_wrist_yaw_link', 'left': 'left_wrist_yaw_link'})

    def __init__(self, publisher, step_timeout=1.):
        if not finite_number(step_timeout) or not 0 < step_timeout <= 5.:
            raise ValueError('step timeout must be in (0, 5] seconds')
        self.publisher = publisher
        self.metadata = copy.deepcopy(type(self).metadata)
        self.step_timeout = step_timeout
        raw = publisher.latest_state()
        self.frame = EpisodeFrame(raw)
        self._state = self.frame.convert(raw)
        self.initial_state = self._state
        self._received_at = time.monotonic()
        self._failure = None
        self._scored_sequence = raw['sequence']
        self._scored_time = raw['sim_time']

    @property
    def state(self):
        if not self._failure:
            try:
                raw = self.publisher.latest_state()
                candidate = self.frame.convert(raw)
                if candidate.sim_time < self._state.sim_time or candidate.sequence < self._state.sequence:
                    raise RuntimeError('simulator state reset during episode')
                self._state = candidate
                self._received_at = time.monotonic()
            except Exception as exc:
                self._failure = str(exc)
                self.idle()
        age = self._state.state_age_s + max(0., time.monotonic()-self._received_at)
        healthy = not self._failure and not self.publisher.error
        return replace(self._state, state_age_s=age, backend_ok=bool(healthy))

    def read_state(self):
        return self.state

    def consume_samples(self):
        """Drain every captured sample after the independent evaluator cursor.

        Historical state_age_s records freshness when captured, so delayed
        consumption does not erase evidence. A gap cannot earn dwell credit.
        Reading .state or calling step() never advances this scoring cursor.
        """
        try:
            if self._failure:
                raise RuntimeError(self._failure)
            raw_samples = self.publisher.samples_after(self._scored_sequence)
            result = []
            sequence, timestamp = self._scored_sequence, self._scored_time
            for raw in raw_samples:
                if raw['sequence'] <= sequence or raw['sim_time'] <= timestamp:
                    raise RuntimeError('unordered simulator history')
                if raw['sim_time']-timestamp > .1+1e-9:
                    raise RuntimeError('simulator sample gap exceeds 0.1 s; continuity lost')
                result.append(self.frame.convert(raw))
                sequence, timestamp = raw['sequence'], raw['sim_time']
            self._scored_sequence, self._scored_time = sequence, timestamp
            return result
        except Exception as exc:
            self._failure = str(exc)
            self.idle()
            raise

    def command_base(self, vx, vy, yaw_rate):
        if self._failure:
            raise RuntimeError(self._failure)
        self.publisher.set_velocity(vx, vy, yaw_rate)

    def idle(self):
        self.publisher.idle()

    def reach(self, position):
        raise NotImplementedError('SONIC Cartesian wrist reach is not implemented or qualified')

    def hold_wrist(self):
        # No wrist command exists in this locomotion-only bridge. In particular,
        # this must not create a fictitious Cartesian hold by editing physics.
        pass

    def step(self, dt):
        if not finite_number(dt) or not 0 < dt <= 2.:
            raise ValueError('measured step duration must be in (0, 2] seconds')
        target = self._state.sim_time + dt
        deadline = time.monotonic() + self.step_timeout
        try:
            while time.monotonic() < deadline:
                raw = self.publisher.latest_state()
                candidate = self.frame.convert(raw)
                if candidate.sim_time < self._state.sim_time or candidate.sequence < self._state.sequence:
                    raise RuntimeError('simulator state reset during episode')
                if candidate.sequence > self._state.sequence and candidate.sim_time >= target-1e-9:
                    self._state = candidate
                    self._received_at = time.monotonic()
                    return self.state
                time.sleep(min(.002, max(0., deadline-time.monotonic())))
            raise TimeoutError('simulator did not provide increasing measured state before deadline')
        except Exception as exc:
            self._failure = str(exc)
            self.idle()
            raise
