"""Small world-frame tools for a supervised approach/reach session."""
import math
import time
from .models import finite_number, wrap_yaw
from .execution import TOOLS
from .toolkit.stationary import StationaryTools


class SupervisedPublisher:
    lease_s=.25
    def __init__(self,session): self.session=session
    def latest_state(self): return self.session.observe()
    def set_stationary(self,**fields): self.session.command_stationary(**fields)
    def idle(self): self.session.idle()


class SupervisedStationaryTools(StationaryTools):
    def _expired(self):
        # The supervisor, rather than the Python caller, refreshes valid holds.
        # A supervisor fallback invalidates the old tool's reference explicitly.
        with self.publisher.session.lock:
            active=self.publisher.session.desired[0]=='stationary'
        if (self.mode!=0 or self.positions is not None) and not active:
            self.mode,self.height,self.positions=0,-1.,None
            self.reference_configuration=None
            self.pose_goals=None
            return True
        return False


class SessionTools:
    METHODS=TOOLS
    def __init__(self,session,planner):
        self.session,self.planner=session,planner
        self.stationary=None
        self.paired_planner=None

    def observe(self): return self.session.observe()

    def observe_scene(self):
        scene=getattr(self.session.task,'scene',None)
        if scene is None:
            return dict(status='rejected',reason='scene_unavailable')
        raw=self.observe()
        return dict(scene.to_dict(),sim_time=raw['sim_time'],sequence=raw['sequence'],
                    objects=raw.get('objects',{}),contacts=raw.get('contacts',[]),
                    object_observation='privileged_current_simulator_state',
                    state_age_s=raw['state_age_s'],session_id=raw['session_id'])

    def check_reach(self,target_world):
        """Read-only local IK from one measured stance; never installs a command."""
        from .stationary_task import state_fault
        raw=self.observe()
        fault=state_fault(raw)
        if fault:
            return dict(status='rejected',reason=fault)
        if self.planner is None:
            return dict(status='rejected',reason='planner_unavailable')
        result=self.planner.plan(raw,target_world)
        return dict(result,sequence=raw['sequence'],sim_time=raw['sim_time'],
                    assumption='fixed measured base, waist, left arm and fingers; no balance or tracking guarantee')

    def _paired(self,raw):
        if self.paired_planner is None:
            from .toolkit.bimanual import BimanualPlanner
            self.paired_planner=BimanualPlanner(raw['model_path'])
        return self.paired_planner

    def check_hands(self,goals):
        """Read-only local paired wrist IK; no reference or physical-state change."""
        from .stationary_task import state_fault
        from .toolkit.bimanual import validate_goals,hand_errors
        validate_goals(goals)
        raw=self.observe()
        fault=state_fault(raw)
        try:hand_errors(raw,goals)
        except ValueError:fault=fault or 'invalid_hand_state'
        if fault:return dict(status='rejected',reason=fault)
        result=self._paired(raw).plan(raw,goals)
        return dict(result,sequence=raw['sequence'],sim_time=raw['sim_time'],
                    assumption='fixed measured base/waist/fingers/objects; kinematics only, no contact or grasp')

    def reach_hands(self,goals,timeout=20.):
        from .toolkit.bimanual import validate_goals
        validate_goals(goals)
        if not finite_number(timeout) or not 0<timeout<=20.:
            raise ValueError('timeout must be in (0,20] seconds')
        raw=self.observe()
        if raw['settled_for']<.5:
            return dict(status='rejected',reason='base_not_settled',observation=raw)
        planner=self._paired(raw)
        tools=self._stationary()
        tools.paired_planner=planner
        return tools.reach_hands(goals,timeout)

    def _stationary(self):
        if self.stationary is None:
            self.session.idle()
            self.stationary=SupervisedStationaryTools(SupervisedPublisher(self.session),self.planner,
                on_event=lambda event:self.session._event('stationary_tool',event=event))
        return self.stationary

    def set_posture(self,height=None,timeout=8.):
        if self.observe()['settled_for']<.5:
            return dict(status='rejected',reason='base_not_settled',observation=self.observe())
        return self._stationary().set_posture(height,timeout)

    def reach_right(self,target_world,timeout=10.):
        if self.observe()['settled_for']<.5:
            return dict(status='rejected',reason='base_not_settled',observation=self.observe())
        return self._stationary().reach_right(target_world,timeout)

    def hold(self,duration=1.):
        return self._stationary().hold(duration)

    def stop(self,timeout=3.):
        if self.stationary is not None:
            return self.stationary.stop()
        return self._navigate('stop',None,timeout)

    def walk_to(self,position_xy,timeout=15.):
        if len(position_xy)!=2 or not all(finite_number(v) for v in position_xy):
            raise ValueError('two finite world coordinates in metres required')
        from .mobility_task import MobilityTask
        task=self.session.task
        radius=task.workspace_radius if isinstance(task,MobilityTask) else 1.2
        if math.dist(position_xy,self.session.initial['pelvis_position'][:2])>radius:
            raise ValueError('target exceeds task navigation envelope')
        if isinstance(task,MobilityTask) and task.corridor_distance(position_xy)>task.corridor_half_width:
            raise ValueError('target lies outside the declared route corridor')
        return self._navigate('walk_to',position_xy,timeout)

    def turn_to(self,yaw,timeout=8.):
        if not finite_number(yaw) or abs(wrap_yaw(yaw-self.observe()['pelvis_yaw']))>math.pi/2:
            raise ValueError('world yaw must be finite and within pi/2 of current heading')
        return self._navigate('turn_to',yaw,timeout)

    def move_base(self,vx,vy,yaw_rate,duration):
        if not finite_number(duration) or not 0<duration<=2:
            raise ValueError('duration must be in (0,2] seconds')
        from .toolkit.sonic_backend import LeasedPlanner
        LeasedPlanner(0.,0.).command(vx,vy,yaw_rate,0.)
        return self._navigate('move_base',(vx,vy,yaw_rate),duration)

    def sonic_motion(self,segments):
        """Run a bounded command sequence; completion means duration, not attainment."""
        from .toolkit.motion_sequence import prepare_motion,motion_fields
        from .stationary_task import state_fault
        start=self.observe()
        prepared=prepare_motion(segments,start,self.planner)
        fault=state_fault(start)
        if fault or start['episode_status']!='running':
            return dict(status='rejected',reason=fault or start['episode_status'],observation=start)
        if start['settled_for']<.5:
            return dict(status='rejected',reason='base_not_settled',observation=start)
        # Only accepted requests release old task references. No reset or replay.
        self.stationary=None
        self.session.idle()
        total=sum(s['duration'] for s in prepared)
        wall_end=time.monotonic()+total*2+1.
        index=0;offset=0.;announced=-1
        status,reason='timed_out','wall_timeout'
        try:
            while time.monotonic()<wall_end:
                raw=self.observe();elapsed=raw['sim_time']-start['sim_time']
                if raw['episode_status']!='running':
                    status,reason='cancelled',raw['episode_status'];break
                # Check continuity before time completion: a stalled caller may
                # wake after both the motion lease and entire sequence expired.
                if announced>=0:
                    with self.session.lock:
                        active=(self.session.desired[0]=='motion' and
                                time.monotonic()<self.session.velocity_until)
                    if not active:
                        status,reason='failed','motion_reference_lost';break
                if elapsed>=total:
                    status,reason=('completed','duration_elapsed') if announced>=0 else ('timed_out','sequence_not_started')
                    break
                while elapsed>=offset+prepared[index]['duration']:
                    offset+=prepared[index]['duration'];index+=1
                fields=motion_fields(prepared[index],elapsed-offset)
                if announced!=index:
                    self.session._event('motion_segment',index=index,segment=prepared[index])
                    announced=index
                self.session.command_motion(**fields)
                self.session._event('motion_reference',index=index,fields=fields)
                time.sleep(.02)
        finally:
            self.session.idle()
        end=self.observe()
        return dict(status=status,reason=reason,operation='sonic_motion',
                    elapsed=end['sim_time']-start['sim_time'],requested_duration=total,
                    completion='command_duration_only',observation=end)

    def _navigate(self,operation,target,timeout):
        if not finite_number(timeout) or not 0<timeout<=20:
            raise ValueError('timeout must be in (0,20] seconds')
        # This first demo intentionally releases arm references before walking.
        # It is unsuitable for carrying; this transition is part of the trace.
        self.stationary=None
        self.session.idle()
        self.session._event('navigation_start',operation=operation,target=target)
        start=self.observe()
        wall_end=time.monotonic()+timeout*2+1
        since=None
        status,reason='timed_out','command_timeout'
        try:
            while time.monotonic()<wall_end:
                raw=self.observe()
                elapsed=raw['sim_time']-start['sim_time']
                if raw['episode_status']!='running':
                    status,reason='cancelled',raw['episode_status']
                    break
                if elapsed>=timeout:
                    if operation=='move_base': status,reason='completed','duration_elapsed'
                    break
                vx=vy=w=0.
                good=False
                if operation=='walk_to':
                    dx,dy=(target[i]-raw['pelvis_position'][i] for i in range(2))
                    distance=math.hypot(dx,dy)
                    c,s=math.cos(raw['pelvis_yaw']),math.sin(raw['pelvis_yaw'])
                    # Match the existing navigation wrapper's gain and bounds.
                    vx,vy=1.5*(c*dx+s*dy),1.5*(-s*dx+c*dy)
                    scale=min(1.,.25/max(math.hypot(vx,vy),1e-12),.20/max(abs(vy),1e-12))
                    vx,vy=vx*scale,vy*scale
                    good=distance<=.06
                elif operation=='turn_to':
                    error=wrap_yaw(target-raw['pelvis_yaw'])
                    w=max(-.35,min(.35,.8*error))
                    good=abs(error)<=.08
                elif operation=='stop': good=True
                else: vx,vy,w=target
                if good:
                    self.session.idle()
                    settled=math.hypot(*raw['planar_velocity'])<=.05 and abs(raw['yaw_rate'])<=.10
                    since=(raw['sim_time'] if since is None else since) if settled else None
                    if since is not None and raw['sim_time']-since>=.5:
                        status,reason='completed','measured_goal'
                        break
                else:
                    since=None
                    self.session.command_velocity(vx,vy,w)
                time.sleep(.02)
            else: reason='wall_timeout'
        finally:
            self.session.idle()
        return dict(status=status,reason=reason,operation=operation,
                    elapsed=self.observe()['sim_time']-start['sim_time'],observation=self.observe())


def session_api(task=None):
    workstation=task is not None and task.get('name')=='workstation_reach'
    goal=('''Task: use observe_scene() to find task['target_id'] and reach that marker.
Choose your own base position and pelvis height; neither is prescribed.
The scene contains static collision boxes and nonphysical wrist markers.
The legacy observation field approached has no meaning for this task. Use
episode_status for completion, and measured wrist/stance state for planning.
Final success requires wrist within 5 cm, both feet supported, slow wrist/base
and small tilt together for .5s. Contact with workstations is forbidden.
''' if workstation else '''Task: approach task['approach_world_xy'], then reach task['wrist_target_world']
at pelvis height task['height']. Approach requires base within 8 cm and settled
for .5s; final success requires prior approach, base within 15 cm, wrist within
5 cm, height within 3 cm, both feet supported, slow wrist/base and small tilt
together for .5s.
''')
    if task is not None and task.get('name')=='mobility':
        goal='''Task: visit task['waypoints'] in order. These are world XY metres.
Read observe()['task_progress']['waypoint_index'] to resume at the first unfinished
region. Stop within 12 cm at each region, with both feet loaded and slow base/yaw,
continuously for .5 s. Final region additionally requires task['final_yaw'] within
.12 rad when specified. Stay inside the fixed route corridor. Larger task workspace
does not increase motor speed or per-call timeout. Retry from current state after
a walk timeout; do not assume failure erased distance already walked.
'''
    return '''Write one self-contained Python file defining run(robot, task).
This program runs on the CURRENT robot in a persistent simulation session. Prior
actions and elapsed time remain; do not replay earlier work blindly. Read observe()
first, including episode_status, elapsed, remaining, pelvis_position,
pelvis_yaw, right_wrist_position and right_wrist_velocity_world. Units are metres,
radians and seconds. ALL poses/targets use MuJoCo WORLD coordinates, Z up; these
are privileged simulator observations, no camera input. This differs from legacy APIs.
''' + goal + '''Fixed targets persist across rounds. Reaching means
the right_wrist_yaw_link origin, not a fingertip. No grasping is implemented.
Methods: observe(); observe_scene(); check_reach(target_world);
check_hands(goals); reach_hands(goals,timeout=20);
walk_to(world_xy,timeout=15); turn_to(world_yaw,timeout=8);
move_base(vx,vy,yaw_rate,duration); stop(timeout=3); hold(duration=1);
set_posture(height=None,timeout=8); reach_right(target_world,timeout=10); sonic_motion(segments).
sonic_motion executes a short timed sequence of SONIC movement/facing/posture and
bounded named right-arm references. Read tool_docs/sonic_motion.md before use.
It requires settled entry; completed/duration_elapsed means only the interval
finished. All accepted exits release motion references toward nominal idle.
It has no IK, obstacle avoidance, automatic grasp or arbitrary task-goal solver.
Velocity is BODY forward/left in m/s; norm<=.30, |vy|<=.20, |yaw_rate|<=.40 rad/s,
duration in (0,2]. Turn change<=pi/2. Motion timeouts in (0,20]. hold duration [.5,5].
set_posture accepts None for neutral or a requested height condition [.60,.85] metres;
this accepted range is not a demonstrated measured-height capability.
Posture/reach entry requires .5s measured two-foot settling; otherwise returns
rejected/base_not_settled without applying a new posture/arm command. Use stop()
or hold() and fresh observe()['settled_for'] before retrying.
reach_right does local collision-aware position-only IK from the CURRENT stance;
target must be within 25 cm of the current wrist. The controller may not track it.
observe_scene returns named boxes (center_world, half_size) and markers
(name, workstation, position_world, label), with sim_time/sequence/state_age_s.
Box centres describe initial scene geometry. Positive mass (kg) denotes a free
object; objects[name] gives its CURRENT world position, WXYZ quaternion and world
linear/angular velocity. contacts contains measured contact bodies/forces.
All are privileged simulator data, not camera perception or a grasp detector.
Legacy open-floor tasks return rejected/scene_unavailable.
check_reach performs the same bounded local IK without moving the robot. It returns
planned or rejected, reason, joint_path (seven right-arm radians per point) and
snapshot sequence/time. It fixes the measured base/waist/left arm/fingers and
does not establish balance or tracking feasibility. Compute only; no command
changes. reach_right ALWAYS replans from a fresh state before execution.
check_hands/reach_hands accept a dict with exactly left and right, each with
position (world metres) and quaternion (unit WXYZ body-to-world). Targets refer
to wrist_yaw_link origins, not palm centres. Each target must be within 40 cm
of its current wrist. Planning freezes root, waist, fingers and free objects.
reach_hands uses bounded feedback replanning; completion requires BOTH wrists
within 2 cm/0.15 rad and below 0.05 m/s, with stable stance, for 0.5 s.
No intentional hand/object contact is allowed. Read tool_docs/reach_hands.md.
Navigation releases arm/posture targets. Posture changes release old arm targets.
Successful posture/reach leaves a supervised hold during editing, subject to
configuration/fault checks. Physics and task time continue while you compute.
Results use status completed, timed_out, rejected, failed or cancelled, with reason.
Tool completion is not whole-task success. Planning rejection is recoverable;
reobserve and change your approach if needed. Falls/global timeout end the session.
A Python error preserves earlier actions. Returning ends only this program round;
the parent may ask you for a revision using feedback and the same physical world.
Use Python control flow and math. No simulator mutation, filesystem/network access
in the submitted worker. Source is one file; development helper files are not bundled.
'''
