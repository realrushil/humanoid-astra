"""Headless upstream SONIC MuJoCo simulator with local measured-state RPC.

Run in the Linux SONIC environment. Importing this module requires only stdlib.
The declared initial arm/hand posture is assigned before the first recording or
physics step. Subsequent motion comes from actuator forces; poses never reset.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time


def _vector(values, length):
    result = [float(value) for value in values]
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f'expected {length} finite physics values')
    return result


def _quaternion(values):
    q = _vector(values, 4)
    norm = math.hypot(*q)
    if norm < 1e-12:
        raise ValueError('zero quaternion cannot describe orientation')
    return [value / norm for value in q]


def quaternion_state(quaternion_wxyz):
    """Return XYZ roll/pitch/yaw and angle of body +Z from world +Z."""
    w, x, y, z = _quaternion(quaternion_wxyz)
    return dict(
        roll=math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
        pitch=math.asin(max(-1., min(1., 2 * (w*y - z*x)))),
        yaw=math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)),
        tilt=math.acos(max(-1., min(1., 1 - 2 * (x*x + y*y)))),
    )


def world_to_body(vector_world, quaternion_wxyz):
    """Apply the inverse body-to-world quaternion to a 3-vector."""
    vx, vy, vz = _vector(vector_world, 3)
    w, x, y, z = _quaternion(quaternion_wxyz)
    return [
        (1-2*(y*y+z*z))*vx + 2*(x*y+w*z)*vy + 2*(x*z-w*y)*vz,
        2*(x*y-w*z)*vx + (1-2*(x*x+z*z))*vy + 2*(y*z+w*x)*vz,
        2*(x*z+w*y)*vx + 2*(y*z-w*x)*vy + (1-2*(x*x+y*y))*vz,
    ]


def _ground_contact(state, tilt_limit):
    """Healthy, load-bearing feet; simulator observations, not hardware readiness."""
    feet = state.get('foot_contacts', {})
    forces = state.get('foot_normal_forces', {})
    age = state.get('state_age_s')
    if age is None:
        age = max(0., time.monotonic()-state['measured_monotonic'])
    return (state['ready'] and state['backend_ok'] and age <= .25
            and not state['fallen']
            and feet.get('left') is True and feet.get('right') is True
            and all(forces.get(side,0.) >= 5. for side in ('left','right'))
            and state['pelvis_position'][2] > .5 and state['tilt'] < tilt_limit)


def support_release_ready(state):
    """Transfer load after controller activation and ground contact.

    The stiff startup spring constrains balance; velocity convergence is checked
    after removing it. The runtime separately requires 0.1 s of contact first.
    """
    return state['support_enabled'] and _ground_contact(state, math.radians(10))


def unsupported_standing(state):
    """Task admission: no external support and stable two-foot standing."""
    return (not state['support_enabled'] and state['no_support']
            and _ground_contact(state, math.radians(15))
            and math.hypot(*state['planar_velocity']) < .05 and abs(state['yaw_rate']) < .10)


def summarize_contacts(rows):
    """Physics contact evidence for stationary qualification, in metres/newtons.

    Floor/ankle contact is permitted. Other external contacts with >2 mm penetration
    or >5 N normal force are forbidden. Shallow self-touch is recorded separately;
    self-penetration >2 mm is forbidden. These are development tolerances,
    not calibrated collision or balance limits; force alone cannot prove balance.
    """
    forces, forbidden, self_contacts = {'left': 0., 'right': 0.}, [], []
    for row in rows:
        if row.get('robot_contact') is False:
            continue  # Object/table/floor support is not a robot contact fault.
        bodies = {row['body1'], row['body2']}
        side = next((s for s in forces if bodies == {'world', f'{s}_ankle_roll_link'}), None)
        if side:
            forces[side] += max(0., row['normal_force'])
        else:
            if row.get('self_contact'):
                self_contacts.append(row)
            if row['distance'] < -.002 or (not row.get('self_contact') and row['normal_force'] > 5.):
                forbidden.append(row)
    return dict(foot_normal_forces=forces, forbidden_contacts=forbidden, self_contacts=self_contacts)


def initial_joint_targets(arm_posture, hand_posture):
    """Named initial joint radians, using the same profiles as the publisher."""
    from .toolkit.sonic_backend import arm_reference, hand_reference
    from .toolkit.arm_planner import UPPER_BODY
    arms, hands = arm_reference(arm_posture), hand_reference(hand_posture)
    targets = dict(zip(UPPER_BODY, arms.get('upper_body_position', ())))
    for side in ('left', 'right'):
        # Upstream DDS hand order differs between sides; both start with thumb.
        fingers = ('middle', 'index') if side == 'left' else ('index', 'middle')
        suffixes = ('thumb_0', 'thumb_1', 'thumb_2') + tuple(f'{f}_{i}' for f in fingers for i in (0, 1))
        names = [f'{side}_hand_{suffix}_joint' for suffix in suffixes]
        targets.update(zip(names, hands.get(side+'_hand_position', ())))
    return targets


def latch_environment_contact(previous, state):
    """Keep the first forbidden external contact, even if it lasts one step.

    Startup uses this history to reject a disturbed scene. After admission it is
    diagnostic: intentional-contact tools must interpret individual contacts.
    """
    if previous is not None:
        return previous
    for contact in state['forbidden_contacts']:
        if not contact.get('self_contact', False):
            return dict(sim_time=state['sim_time'], contact=contact)
    return None


class SonicSimulation:
    """One physics owner; RPC reads immutable snapshots under the same lock."""

    def __init__(self, repo, record_path=None, model_path=None, *,
                 arm_posture='upstream_default', hand_posture='upstream_default'):
        self.repo = Path(repo).resolve()
        yaml_path = self.repo / 'gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml'
        if not yaml_path.is_file():
            raise RuntimeError(f'not an upstream SONIC repository: {self.repo}')
        sys.path.insert(0, str(self.repo))
        try:
            import yaml
            import mujoco
            import numpy
            from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
        except ImportError as exc:
            raise RuntimeError(
                'SONIC simulator dependencies missing; use the Linux environment with '
                'mujoco, numpy, scipy, PyYAML and unitree_sdk2py: ' + str(exc)
            ) from exc
        config = yaml.safe_load(yaml_path.read_text())
        config.update(DOMAIN_ID=1, INTERFACE='lo', USE_JOYSTICK=0,
                      ENABLE_ONSCREEN=False, ENABLE_OFFSCREEN=False,
                      ENABLE_ELASTIC_BAND=True, enable_waist=True)
        self.model_path = Path(model_path).resolve() if model_path else (self.repo / config['ROBOT_SCENE']).resolve()
        if not self.model_path.is_file():
            raise RuntimeError(f'upstream G1 robot assets missing: {self.model_path}')
        # Absolute generated scene path retains the robot but adds static geometry.
        # The exact same model_path is published for planning and playback.
        config['ROBOT_SCENE'] = str(self.model_path)
        self.mujoco, self.np = mujoco, numpy
        self.simulator = BaseSimulator(config, onscreen=False, offscreen=False)
        self.env = self.simulator.sim_env
        self.initial_targets = initial_joint_targets(arm_posture, hand_posture)
        for name, value in self.initial_targets.items():
            joint = self.env.mj_model.joint(name)
            if joint.limited[0] and not joint.range[0] <= value <= joint.range[1]:
                raise ValueError(f'initial posture exceeds joint limits: {name}')
            self.env.mj_data.qpos[joint.qposadr] = value
        self.physics_started = False
        self.first_environment_contact = None
        # Upstream anchors the pelvis at z=1 m, suspending both feet. Support
        # the model's initial pose instead; never release into a drop test.
        self.env.elastic_band.point = self.env.mj_data.qpos[:3].copy()
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.sequence = 0
        self.error = None
        self.fallen = False
        self.release_at = None
        self.released_at = None
        self.thread = None
        self.record_path = Path(record_path) if record_path else None
        self._record_file = None
        self._record_next_time = None
        self.wrist_ids = {
            side: self.env.mj_model.body(f'{side}_wrist_yaw_link').id
            for side in ('left', 'right')
        }
        from .sim_state import robot_joint_ids
        self.robot_joints=robot_joint_ids(self.env.mj_model)
        # Upstream resets on falls, hiding failures and rewinding simulation time.
        # Keep upstream dynamics and latch the evidence instead of resetting.
        self.env.check_fall = self._check_fall
        mujoco.mj_forward(self.env.mj_model, self.env.mj_data)
        self.snapshot = self._capture()
        if self.record_path:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self._record_file = self.record_path.open('w', encoding='utf-8')
            self._record_file.write(json.dumps({
                'record_type': 'metadata', 'model_path': str(self.model_path),
                'model_name': self.snapshot['model_name'], 'qpos_size': len(self.env.mj_data.qpos),
                'sim_dt': float(self.simulator.sim_dt), 'record_hz': 30.0,
                'initial_joint_targets': self.initial_targets,
                'waits_for_first_body_command': True,
            }, allow_nan=False) + '\n')
            self._record_frame(self.snapshot, force=True)

    def _check_fall(self):
        self.fallen = self.fallen or float(self.env.mj_data.qpos[2]) < .2
        self.env.fall = self.fallen

    def _capture(self):
        env, mj = self.env, self.mujoco
        model, data = env.mj_model, env.mj_data
        q = data.qpos[3:7].tolist()
        attitude = quaternion_state(q)
        velocity = self.np.zeros(6)
        mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY,
                             env.root_body_id, velocity, 0)
        linear = velocity[3:6].tolist()
        state = dict(
            sim_time=float(data.time), sequence=self.sequence,
            measured_monotonic=time.monotonic(), measured_unix=time.time(),
            pelvis_position=data.qpos[:3].tolist(), pelvis_quaternion_wxyz=q,
            pelvis_yaw=attitude['yaw'], roll=attitude['roll'], pitch=attitude['pitch'],
            tilt=attitude['tilt'], planar_velocity=linear[:2],
            linear_velocity_world=linear, linear_velocity_body=world_to_body(linear, q),
            angular_velocity_world=velocity[:3].tolist(), yaw_rate=float(velocity[2]),
            support_enabled=bool(env.elastic_band.enable),
            support_force=data.xfrc_applied[env.band_attached_link].tolist(),
            support_released_at=self.released_at, support_release_scheduled_at=self.release_at,
            no_support=not bool(env.elastic_band.enable) and not bool(
                self.np.any(data.xfrc_applied[env.band_attached_link])),
            fallen=self.fallen, ready=self.error is None,
            backend_ok=self.error is None, error=self.error,
            model_name=model.names.split(b'\0', 1)[0].decode(),
            model_path=str(self.model_path), dds_domain=1, dds_interface='lo',
            joint_names=[model.joint(i).name for i in self.robot_joints],
            joint_positions=data.qpos[model.jnt_qposadr[self.robot_joints]].tolist(),
            joint_velocities=data.qvel[model.jnt_dofadr[self.robot_joints]].tolist(),
            qpos=data.qpos.tolist(),
            body_joint_names=[model.joint(int(i)).name for i in env.body_joint_index],
            body_joint_positions=data.qpos[env.body_joint_index + 6].tolist(),
            body_joint_velocities=data.qvel[env.body_joint_index + 5].tolist(),
        )
        from .sim_state import object_states,geom_metadata
        state['objects']=object_states(model,data)
        feet = {'left': False, 'right': False}
        floor = model.geom('floor').id
        contacts = []
        for index, contact in enumerate(data.contact):
            force = self.np.zeros(6)
            mj.mj_contactForce(model, data, index, force)
            contacts.append(dict(body1=model.body(int(model.geom_bodyid[contact.geom[0]])).name,
                                 body2=model.body(int(model.geom_bodyid[contact.geom[1]])).name,
                                 robot_contact=any(model.body_rootid[model.geom_bodyid[g]]==env.root_body_id for g in contact.geom),
                                 self_contact=bool(model.body_rootid[model.geom_bodyid[contact.geom[0]]] ==
                                                   model.body_rootid[model.geom_bodyid[contact.geom[1]]]),
                                 geoms=[geom_metadata(model,g) for g in contact.geom],
                                 position_world=contact.pos.tolist(), normal_world=contact.frame[:3].tolist(),
                                 distance=float(contact.dist), normal_force=float(force[0])))
            if floor in contact.geom and contact.dist <= .002:
                other = contact.geom[1] if contact.geom[0] == floor else contact.geom[0]
                body = model.body(int(model.geom_bodyid[other])).name
                for side in feet:
                    if body == f'{side}_ankle_roll_link':
                        feet[side] = True
        state['foot_contacts'] = feet
        state['contacts']=contacts
        state.update(summarize_contacts(contacts))
        self.first_environment_contact = latch_environment_contact(self.first_environment_contact, state)
        state['first_environment_contact'] = self.first_environment_contact
        state['physics_started'] = self.physics_started
        state['support_anchor'] = env.elastic_band.point.tolist()
        for side, body_id in self.wrist_ids.items():
            state[f'{side}_wrist_position'] = data.xpos[body_id].tolist()
            state[f'{side}_wrist_quaternion_wxyz'] = data.xquat[body_id].tolist()
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
            state[f'{side}_wrist_velocity_world'] = velocity[3:6].tolist()
        json.dumps(state, allow_nan=False)
        return state

    def _record_frame(self, state, force=False):
        if self._record_file is None:
            return
        sim_time = float(state['sim_time'])
        if not force and self._record_next_time is not None and sim_time < self._record_next_time:
            return
        self._record_file.write(json.dumps({
            'record_type': 'frame', 'sim_time': sim_time,
            'qpos': self.env.mj_data.qpos.tolist(), 'no_support': bool(state['no_support']),
        }, allow_nan=False) + '\n')
        self._record_file.flush()
        if self._record_next_time is None:
            self._record_next_time = sim_time + 1.0 / 30.0
        else:
            self._record_next_time += 1.0 / 30.0

    def start(self):
        if self.thread is not None:
            raise RuntimeError('simulation already started')
        self.thread = threading.Thread(target=self._run, name='sonic-physics', daemon=True)
        self.thread.start()

    def _advance(self):
        """Publish initial DDS state while loading; never pause after first command."""
        bridge = self.env.unitree_bridge
        with bridge.low_cmd_lock:
            self.physics_started = self.physics_started or bridge.low_cmd_received
        if self.physics_started:
            self.env.sim_step()
        else:
            bridge.PublishLowState(self.env.prepare_obs())
        return self.physics_started

    def _run(self):
        try:
            while not self.stopped.is_set():
                started = time.monotonic()
                with self.lock:
                    if self.release_at is not None and self.env.mj_data.time >= self.release_at:
                        if not support_release_ready(self.snapshot):
                            raise RuntimeError('support release lost load-bearing two-foot contact')
                        self.env.elastic_band.enable = False
                        self.env.mj_data.xfrc_applied[self.env.band_attached_link] = 0.
                        self.released_at = float(self.env.mj_data.time)
                        self.release_at = None
                    advanced = self._advance()
                    # mj_step leaves position-dependent derived data from before
                    # integration. Forward refreshes wrist positions to this qpos.
                    self.mujoco.mj_forward(self.env.mj_model, self.env.mj_data)
                    self.sequence += int(advanced)
                    self.snapshot = self._capture()
                    self._record_frame(self.snapshot)
                self.stopped.wait(max(0., self.simulator.sim_dt - (time.monotonic()-started)))
        except Exception as exc:
            with self.lock:
                self.error = f'{type(exc).__name__}: {exc}'
                self.snapshot.update(backend_ok=False, ready=False, error=self.error)
            self.stopped.set()
        finally:
            if self._record_file is not None:
                self._record_file.close()
                self._record_file = None
            self.simulator.close()

    def request(self, request):
        if not isinstance(request, dict):
            raise ValueError('request must be a JSON object')
        command = request.get('command')
        with self.lock:
            if command == 'release_support':
                at = float(request.get('at_sim_time', self.snapshot['sim_time']))
                if not math.isfinite(at) or at < 0:
                    raise ValueError('at_sim_time must be finite and nonnegative')
                if self.env.elastic_band.enable:
                    if not support_release_ready(self.snapshot):
                        raise ValueError('support release requires load-bearing two-foot contact')
                    self.release_at = at
            elif command == 'shutdown':
                self.stopped.set()
            elif command != 'state':
                raise ValueError('supported commands: state, release_support, shutdown')
            result = dict(self.snapshot)
            result['support_release_scheduled_at'] = self.release_at
            result['state_age_s'] = max(0., time.monotonic()-result['measured_monotonic'])
            return result

    def close(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.simulator.close()


def serve(simulation, socket_path):
    """One newline-delimited JSON request/reply per local connection."""
    path = Path(socket_path)
    # Fail on any existing socket rather than disrupting another simulator.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        os.chmod(path, 0o600)
        server.listen(8)
        server.settimeout(.2)
        try:
            simulation.start()
            while not simulation.stopped.is_set():
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(2.)
                    try:
                        with connection.makefile('rb') as reader:
                            data = reader.readline(65537)
                        if len(data) > 65536 or not data.endswith(b'\n'):
                            raise ValueError('expected one JSON line, maximum 64 KiB')
                        response = {'ok': True, 'state': simulation.request(json.loads(data))}
                    except (ValueError, TypeError, OSError) as exc:
                        response = {'ok': False, 'error': str(exc)}
                    try:
                        connection.sendall(json.dumps(response, allow_nan=False).encode()+b'\n')
                    except OSError:
                        pass
        finally:
            simulation.close()
            path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--socket', required=True)
    parser.add_argument('--record', help='JSONL measured qpos recording path')
    parser.add_argument('--model', help='Trusted prebuilt scene XML; simulation startup only')
    parser.add_argument('--arm-posture', default='upstream_default', choices=('upstream_default','travel'))
    parser.add_argument('--hand-posture', default='upstream_default', choices=('upstream_default','tucked_thumb'))
    args = parser.parse_args(argv)
    simulation = None
    try:
        simulation = SonicSimulation(args.repo, args.record, args.model,
                                     arm_posture=args.arm_posture, hand_posture=args.hand_posture)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: simulation.stopped.set())
        serve(simulation, args.socket)
        if simulation.error:
            raise RuntimeError(simulation.error)
    except Exception as exc:
        if simulation is not None:
            simulation.close()
        print(f'sonic_sim: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
