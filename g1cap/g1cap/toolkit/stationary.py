"""Bounded stationary capabilities for the simulation-only trusted parent.

This executor uses MuJoCo world coordinates, unlike the legacy navigation API's
episode frame. SessionTools exposes it through the persistent coding worker.
"""
import bisect
import copy
from concurrent.futures import ThreadPoolExecutor
import math
import time
from ..models import finite_number
from ..stationary_task import state_fault
from .arm_planner import RIGHT_ARM, UPPER_BODY, upper_reference
from .bimanual import ARMS, paired_reference, validate_goals, hand_errors, hands_at_goal, quaternion_error


class ExecutionFailure(Exception):
    """A checked execution failure whose reason should reach the caller."""


class StationaryTools:
    def __init__(self, publisher, planner, on_event=None, paired_planner=None):
        self.publisher, self.planner = publisher, planner
        self.on_event = on_event or (lambda event: None)
        self.paired_planner = paired_planner
        self.pose_goals = None
        self.reference_joints = RIGHT_ARM
        self.initial = self.observe()
        self.mode, self.height, self.positions = 0, -1., None
        self.last_plan = None
        self.reference_deadline = None
        self.reference_configuration = None

    def observe(self):
        """Privileged measured state; frame=mujoco_world, lengths=m, angles=rad."""
        return self.publisher.latest_state()

    def _publish(self):
        self.publisher.set_stationary(mode=self.mode, height=self.height,
                                      positions=self.positions,
                                      velocities=[0.]*17 if self.positions is not None else None)
        self.reference_deadline = time.monotonic()+getattr(self.publisher,'lease_s',.25)

    def _release(self):
        self.publisher.idle()
        self.mode,self.height,self.positions=0,-1.,None
        self.reference_deadline=None
        self.pose_goals=self.reference_configuration=None

    def _expired(self):
        if (self.mode != 0 or self.positions is not None) and self.reference_deadline is not None:
            if time.monotonic() >= self.reference_deadline:
                self._release()
                return True
        return False

    def _run(self, operation, timeout, good, trajectory=None, plan_start=None):
        if not finite_number(timeout) or not 0 < timeout <= 20.:
            raise ValueError('timeout must be in (0,20] seconds')
        start = self.observe()
        wall_end = time.monotonic()+timeout*2+1.
        since = unsupported_since = None
        previous = None
        reason, status = 'command_timeout', 'timed_out'
        raw = start
        try:
            while time.monotonic() < wall_end:
                raw = self.observe()
                if raw.get('episode_status','running')!='running':
                    status,reason='cancelled',raw['episode_status']
                    break
                reason = state_fault(raw)
                if reason:
                    status = 'failed'
                    break
                if math.dist(raw['pelvis_position'][:2],self.initial['pelvis_position'][:2]) > .15:
                    status,reason = 'failed','base_displacement'
                    break
                if previous is not None:
                    if raw['sim_time'] < previous['sim_time'] or raw['sequence'] < previous['sequence']:
                        status,reason = 'failed','state_order'
                        break
                    if raw['sequence'] == previous['sequence']:
                        time.sleep(.005)
                        continue
                    if not 0 < raw['sim_time']-previous['sim_time'] <= .1+1e-9:
                        status,reason = 'failed','state_gap'
                        break
                elapsed = raw['sim_time']-start['sim_time']
                if elapsed > timeout+1e-9:
                    reason = 'command_timeout'
                    break
                supported = all(raw['foot_normal_forces'][s] >= 5. for s in ('left','right'))
                unsupported_since = (None if supported else raw['sim_time'] if unsupported_since is None else unsupported_since)
                if unsupported_since is not None and raw['sim_time']-unsupported_since >= .25:
                    status,reason = 'failed','support_lost'
                    break
                if self.pose_goals is not None:
                    try:hand_errors(raw,self.pose_goals)
                    except ValueError:raise ExecutionFailure('invalid_hand_state')
                if plan_start is not None:
                    # Replans replace this snapshot; held references retain it.
                    plan_start = self.reference_configuration or plan_start
                    # The collision path assumes measured nonmoving joints/root.
                    current = dict(zip(raw['body_joint_names'],raw['body_joint_positions']))
                    before = dict(zip(plan_start['body_joint_names'],plan_start['body_joint_positions']))
                    if (math.dist(raw['pelvis_position'],plan_start['pelvis_position']) > .05
                            or abs(raw['tilt']-plan_start['tilt']) > .10
                            or abs(math.atan2(math.sin(raw['pelvis_yaw']-plan_start['pelvis_yaw']),
                                              math.cos(raw['pelvis_yaw']-plan_start['pelvis_yaw']))) > .15
                            or any(abs(current[n]-before[n]) > .20 for n in UPPER_BODY if n not in self.reference_joints)):
                        status,reason = 'failed','plan_configuration_changed'
                        break
                    # Free objects remain obstacles even if they move without
                    # contacting the robot. Never keep using a stale scene path.
                    for name,obj in plan_start.get('objects',{}).items():
                        current_obj=raw.get('objects',{}).get(name)
                        if (current_obj is None or math.dist(obj['position_world'],current_obj['position_world'])>.01
                                or quaternion_error(obj['quaternion_wxyz'],current_obj['quaternion_wxyz'])>.10):
                            raise ExecutionFailure('plan_object_changed')
                    if self.reference_joints == ARMS:
                        measured=dict(zip(raw['joint_names'],raw['joint_positions']))
                        frozen=dict(zip(plan_start['joint_names'],plan_start['joint_positions']))
                        if any(abs(measured[n]-q)>.20 for n,q in frozen.items() if 'hand_' in n):
                            raise ExecutionFailure('plan_fingers_changed')
                if trajectory is not None:
                    self.positions = trajectory(raw,elapsed)
                self._publish()
                self.on_event(dict(type='command', operation=operation, sim_time=raw['sim_time'],
                                   mode=self.mode, height=self.height, positions=self.positions, goals=self.pose_goals, observation=raw))
                settled = (supported and abs(raw['tilt']) <= math.radians(15)
                           and math.hypot(*raw['planar_velocity']) <= .05 and abs(raw['yaw_rate']) <= .10)
                qualifying = settled and good(raw,elapsed)
                since = (raw['sim_time'] if since is None else since) if qualifying else None
                if since is not None and raw['sim_time']-since >= .5-1e-9:
                    status,reason = 'completed','measured_goal'
                    break
                if elapsed >= timeout-1e-9:
                    reason = 'command_timeout'
                    break
                previous = raw
                time.sleep(.02)
            else:
                status,reason = 'failed','wall_timeout'
        except ExecutionFailure as exc:
            status,reason = 'failed',str(exc)
        except Exception as exc:
            status,reason = 'failed',f'backend_error:{type(exc).__name__}:{exc}'
        if status != 'completed':
            self._release()
        # A successful call leaves a leased reference, not an indefinite motor
        # promise. Call hold() to maintain it; a stalled caller expires to idle.
        result = dict(operation=operation,status=status,reason=reason,
                      elapsed=raw['sim_time']-start['sim_time'], observation=raw)
        self.on_event(dict(type='result', **result))
        return result

    def set_posture(self, height=None, timeout=8.):
        """Request idle standing (None), or squat clip height [0.60,0.85] m.

        Report completion only within 3 cm of requested *measured* pelvis height.
        Height availability and accuracy depend on the released controller.
        """
        if height is not None and (not finite_number(height) or not .60 <= height <= .85):
            raise ValueError('height must be None or in [0.60,0.85] m')
        if not finite_number(timeout) or not 0 < timeout <= 20.:
            raise ValueError('timeout must be in (0,20] seconds')
        self.mode,self.height,self.positions = (0,-1.,None) if height is None else (4,height,None)
        self.reference_configuration = None
        self.pose_goals = None
        target = self.initial['pelvis_position'][2] if height is None else height
        return self._run('set_posture',timeout,lambda raw,elapsed: abs(raw['pelvis_position'][2]-target) <= .03)

    def reach_right(self, target_world, timeout=10.):
        """Position-only wrist-body-origin reach, world metres, current stance.

        Local planner supplies collision-sampled joint references. The motor
        controller still has to track them; no kinematic result earns success.
        """
        if len(target_world) != 3 or not all(finite_number(v) for v in target_world):
            raise ValueError('three finite world coordinates required')
        if not finite_number(timeout) or not 0 < timeout <= 20.:
            raise ValueError('timeout must be in (0,20] seconds')
        if self.planner is None:
            return dict(status='rejected',reason='planner_unavailable')
        if self._expired():
            return dict(status='rejected',reason='reference_expired')
        sample = self.observe()
        fault = state_fault(sample)
        if fault:
            self._release()
            return dict(status='failed',reason=fault)
        plan = self.planner.plan(sample,target_world)
        self.last_plan = plan
        self.on_event(dict(type='plan', **plan))
        if plan['status'] != 'planned':
            return dict(status='rejected',reason=plan['reason'],planning=plan)
        if self._expired():
            return dict(status='rejected',reason='reference_expired',planning=plan)
        path = plan['joint_path']
        self.pose_goals = None
        self.reference_joints = RIGHT_ARM
        self.reference_configuration = sample
        times = [0.]
        for a,b in zip(path,path[1:]):
            times.append(times[-1]+max(.02,max(abs(x-y) for x,y in zip(a,b))/.35))
        def trajectory(raw,elapsed):
            index = min(bisect.bisect_right(times,elapsed),len(path)-1)
            if index == 0 or elapsed >= times[-1]:
                q = path[index]
            else:
                alpha = (elapsed-times[index-1])/(times[index]-times[index-1])
                q = [a+(b-a)*alpha for a,b in zip(path[index-1],path[index])]
            return upper_reference(sample,q)
        result = self._run('reach_right',timeout,
            lambda raw,elapsed: elapsed >= times[-1] and
                math.dist(raw['right_wrist_position'],target_world) <= .05 and
                math.hypot(*raw['right_wrist_velocity_world']) <= .05,
            trajectory=trajectory,plan_start=sample)
        result.update(planning=dict(status=plan['status'],reason=plan['reason'],
                                    position_error=plan['position_error']),
                      wrist_error=math.dist(result['observation']['right_wrist_position'],target_world))
        return result

    def reach_hands(self, goals, timeout=20.):
        """Collision-checked paired wrist poses, followed with at most two replans.

        Goal error is measured in world metres/radians. Both wrists must remain
        within 2 cm / 0.15 rad and move below 0.05 m/s during the stance dwell.
        No intentional contact. Successful references persist under supervision.
        """
        validate_goals(goals)
        if not finite_number(timeout) or not 0<timeout<=20.:
            raise ValueError('timeout must be in (0,20] seconds')
        if self.paired_planner is None:return dict(status='rejected',reason='planner_unavailable')
        if self._expired():return dict(status='rejected',reason='reference_expired')
        goals=copy.deepcopy(goals)
        sample=self.observe()
        fault=state_fault(sample)
        try:hand_errors(sample,goals)
        except ValueError:fault=fault or 'invalid_hand_state'
        if fault:
            self._release()
            return dict(status='failed',reason=fault)
        plan=self.paired_planner.plan(sample,goals)
        self.last_plan=plan
        self.on_event(dict(type='plan',operation='reach_hands',**plan))
        if plan['status']!='planned':return dict(status='rejected',reason=plan['reason'],planning=plan)
        if self._expired():return dict(status='rejected',reason='reference_expired',planning=plan)
        self.pose_goals=goals
        self.reference_joints=ARMS
        self.reference_configuration=sample
        path=plan['joint_path'];times=path_times(path,.25)
        path_start=0.;replans=0;pending=None;pending_sample=None
        # Planning touches a separate MuJoCo state. One worker permits the
        # observation/publication loop to keep checking faults during IK.
        with ThreadPoolExecutor(max_workers=1) as pool:
            def trajectory(raw,elapsed):
                nonlocal path,times,path_start,replans,pending,pending_sample
                if pending is not None and pending.done():
                    replacement=pending.result();pending=None
                    self.on_event(dict(type='plan',operation='reach_hands',replan=replans,**replacement))
                    if replacement['status']!='planned':
                        raise ExecutionFailure('replan_'+replacement['reason'])
                    path=replacement['joint_path'];times=path_times(path,.25);path_start=elapsed
                    self.reference_configuration=pending_sample
                local_time=elapsed-path_start
                if local_time>=times[-1]+3. and replans<2 and pending is None and not hands_at_goal(raw,goals):
                    replans+=1;pending_sample=copy.deepcopy(raw)
                    pending=pool.submit(self.paired_planner.plan,pending_sample,goals)
                return paired_reference(self.reference_configuration,interpolate_path(path,times,local_time))
            result=self._run('reach_hands',timeout,
                lambda raw,elapsed: elapsed-path_start>=times[-1] and hands_at_goal(raw,goals),
                trajectory=trajectory,plan_start=sample)
        result['replans']=replans
        result['holding']='paired_pose_reference' if result['status']=='completed' else 'nominal_idle'
        try:result['hand_errors']=hand_errors(result['observation'],goals)
        except ValueError:result['hand_errors']=None
        return result

    def hold(self, duration=1.):
        """Refresh current stationary references for duration; verify final settling."""
        if not finite_number(duration) or not .5 <= duration <= 5.:
            raise ValueError('hold duration must be in [0.5,5] seconds')
        if self._expired():
            return dict(status='rejected',reason='reference_expired')
        return self._run('hold',duration+.6,
                         lambda raw,elapsed: elapsed >= duration-.5 and
                         math.hypot(*raw['right_wrist_velocity_world']) <= .05 and
                         (self.pose_goals is None or hands_at_goal(raw,self.pose_goals)),
                         plan_start=self.reference_configuration if self.positions is not None else None)

    def stop(self):
        """Bounded stationary hold, preserving current posture/arm while alive.

        Failure and subsequent lease expiry use idle. This is unsuitable under
        overhead obstacles and does not establish a hardware emergency stop.
        """
        return self.hold(.5)


def path_times(path,speed):
    """Joint radians -> simulation seconds, bounding each joint reference speed."""
    times=[0.]
    for a,b in zip(path,path[1:]):
        times.append(times[-1]+max(.02,max(abs(x-y) for x,y in zip(a,b))/speed))
    return times


def interpolate_path(path,times,elapsed):
    index=min(bisect.bisect_right(times,elapsed),len(path)-1)
    if index==0 or elapsed>=times[-1]:return path[index]
    alpha=(elapsed-times[index-1])/(times[index]-times[index-1])
    return [a+(b-a)*alpha for a,b in zip(path[index-1],path[index])]
