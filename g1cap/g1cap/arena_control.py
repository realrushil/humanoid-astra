"""Bounded box operations; the Arena owner supplies physics and native controllers.

Actions are 43 sim-order joint targets [rad], local vx/vy [m/s], yaw rate
[rad/s], pelvis height [m], and torso RPY [rad]. Observations use world metres,
WXYZ quaternions and simulation seconds. Only the simulator thread calls here.
This development controller assumes the witnessed 20 cm, 0.1 kg cube.
"""
from collections import deque
from copy import deepcopy
import math
import struct

from .arena_observation import assess_hold
from .arena_retreat import LoadedRetreat, preparation_contact_fault
from .arena_lift import SupportedLift, supported_grasp_ready
from .arena_grasp import GripPreparation
from .arena_motion import LoadedTranslation, LoadedTurn, space_fault
from .sensor_lift import ClearanceLift

METHODS=frozenset({'observe','wait','pickup_box','lift_supported_box','raise_held_box','hold_box','return_box_to_source','retreat_with_box','move_with_box','turn_with_box','place_box'})
LOADED_METHODS=frozenset({'retreat_with_box','move_with_box','turn_with_box'})
DT=.02


def observation_fault(obs):
    """Require explicit observation age and substep peak; never default missing data."""
    try:
        numbers=[obs[k] for k in ('time','state_age_s','clearance','tilt','robot_source_peak_N')]
        for key,size in [('box_pos',3),('root_pos',3),('box_quat',4),('root_quat',4),('joint_pos',43)]:
            if len(obs[key])!=size:return 'invalid_observation'
            numbers.extend(obs[key])
        for side in ('left','right'):
            pose=obs['wrists_world'][side+'_wrist_yaw_link']
            if len(pose['pos'])!=3 or len(pose['xyzw'])!=4:return 'invalid_observation'
            numbers.extend(pose['pos']);numbers.extend(pose['xyzw'])
        numbers.extend(obs['hand_forces_N'][s] for s in ('left','right'))
        if not isinstance(obs.get('surfaces',{}),dict):return 'invalid_observation'
        numbers.extend(surface['robot_contact_peak_N'] for surface in obs.get('surfaces',{}).values())
        if 'box_floor_contact_peak_N' in obs:numbers.append(obs['box_floor_contact_peak_N'])
        if not all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in numbers):
            return 'invalid_observation'
        if any(type(obs[k]) is not bool for k in ('bilateral','stance_clear','supported')):
            return 'invalid_observation'
        if sum(v*v for v in obs['box_quat'])<1e-8:return 'invalid_observation'
        if not 0<=obs['state_age_s']<=.25:return 'stale_observation'
    except (KeyError,TypeError,ValueError):return 'invalid_observation'
    return None


class BoxControl:
    def __init__(self, initial_action, acquisition, wrist_motion, *, placement_factory=None, visual_grasp=None,scene_hold=None,approach_feedback=None,hand_clearance_feedback=None,sensor_fault=None,sensor_retreat_factory=None,sensor_lift_ready=None,sensor_turn_factory=None,loaded_stop=None):
        self.last_action=self._action(initial_action)
        self.last_action[43:46]=[0.,0.,0.]
        self.acquisition=acquisition
        # Factory: (observation, common_translation_or_None) -> measured wrist
        # controller exposing command(joints, root, xyzw, previous, elapsed).
        # None requests the witnessed outward withdrawal; a vector requests lift.
        self.wrist_motion=wrist_motion
        self.placement_factory=placement_factory
        self.visual_grasp=visual_grasp  # Optional sensor-derived pickup/hold feedback; called with time only.
        self.loaded_stop=loaded_stop  # Separate sensor-screened stop, reusing the paired owner.
        self.stopping_failure=None
        self.scene_hold=scene_hold  # (phase, time, own reference); receives no privileged observation.
        self.approach_feedback=approach_feedback  # Sensor availability, time-only interface.
        self.hand_clearance_feedback=hand_clearance_feedback
        self.sensor_fault=sensor_fault
        self.sensor_retreat_factory=sensor_retreat_factory
        self.sensor_turn_factory=sensor_turn_factory
        self.sensor_lift_ready=sensor_lift_ready  # Shared floor-frame wrist owner is synchronized.
        self.clearance_lift=None
        self.lift_used_m=0.
        self.lift_failed=False
        if sensor_fault is not None and any(callback is None for callback in
                (visual_grasp,scene_hold,approach_feedback,hand_clearance_feedback)):
            raise ValueError('sensor control requires all pickup/hold measurement callbacks')
        self.history=deque(maxlen=51)
        self.method=None
        self.phase='idle'
        self.phase_started=0.
        self.result=None
        self.terminal_reason=None
        self.motion=None
        self.loaded_motion=None
        self.preparation=None
        self.grasp_prepared=False
        self.supported_lift=None
        self.placement=None
        self.last_observation_time=None
        self.acquired=0
        self.acquired_at=None

    @staticmethod
    def _action(value):
        if len(value)!=50 or not all(math.isfinite(v) for v in value):
            raise ValueError('50 finite native action values required')
        # Native action tensors are float32; retain the actual sent reference.
        return list(struct.unpack('50f',struct.pack('50f',*value)))

    def _reply(self,status,reason,obs):
        keys=('time','root_pos','root_quat','box_pos','box_quat','clearance','bilateral','supported','hand_forces_N','stance_clear')
        return dict(status=status,reason=reason,observation={k:deepcopy(obs[k]) for k in keys if k in obs})

    def _finish(self,status,reason,obs):
        if self.phase=='sensor_lift' and status!='completed':self.lift_failed=True
        if status=='completed' and self.method in ('return_box_to_source','place_box'):
            self.grasp_prepared=False;self.preparation=None
        self.result=self._reply(status,reason,obs)
        if self.method in LOADED_METHODS and self.loaded_motion:
            try:self.result['measurements']=self.loaded_motion.measurements(obs['time'] if self.sensor_fault else obs)
            except (KeyError,TypeError,IndexError):pass  # Invalid observations still cancel navigation.
        self.method=None
        self.phase='idle'
        self.last_action[43:46]=[0.,0.,0.]
        return self.result

    def cancel(self,reason,obs):
        """Owner cancellation retains loaded references and every episode fault."""
        return self._finish('cancelled',reason,obs)

    def _phase(self,phase,obs):
        self.phase=phase
        self.phase_started=obs['time']
        self.history.clear()

    def _hold_metric(self,now):
        return self.visual_grasp(now) if self.visual_grasp is not None else assess_hold(list(self.history))

    def _settled_grasp(self):
        if self.visual_grasp is not None:
            return bool(self.history and self._hold_metric(self.history[-1]['time']).get('ready',False))
        metric=assess_hold(list(self.history))
        return (len(self.history)==51 and all(s['bilateral'] and s['stance_clear'] and s['clearance']>.01 for s in self.history)
                and all(metric.get(k,float('inf'))<=limit for k,limit in
                        [('max_box_speed_m_s',.05),('max_box_angular_speed_rad_s',.2),('max_root_planar_speed_m_s',.05)]))

    def _observation_fault(self,obs):
        if self.sensor_fault is None:return observation_fault(obs)
        # No true pose/contact fields may enter the sensor control track, even
        # if a future branch accidentally tries to read them. Measurements live
        # behind the explicit time-only callbacks above.
        if (set(obs)!={'time'} or isinstance(obs['time'],bool) or
                not isinstance(obs['time'],(int,float)) or not math.isfinite(obs['time']) or obs['time']<0):
            return 'invalid_sensor_control_packet'
        return self.sensor_fault(obs['time'])

    def _geometry_fault(self,now):
        for callback,reason in ((self.approach_feedback,'sensor_approach_unavailable'),
                (self.hand_clearance_feedback,'sensor_hand_clearance_unavailable')):
            if callback is not None and callback(now)['status']!='available':return reason
        return None

    def _stopped_action(self,now_s):
        """A failed task can still need active load support with zero navigation.

        Only an explicitly installed sensor-screened callback may update arms
        after ordinary geometry or task-height feedback is lost. It never turns
        the operation's failure into success. Hard sensor faults bypass this path.
        """
        previous=list(self.last_action);previous[43:46]=[0.,0.,0.]
        action=previous
        if self.loaded_stop is not None:
            try:
                action=self._action(self.loaded_stop(now_s,list(previous)))
                action[43:46]=[0.,0.,0.]
                self.stopping_failure=None
            except ValueError as error:
                self.stopping_failure=str(error)
        self.last_action=self._action(action)
        return list(self.last_action)

    def _begin_sensor_lift(self,now_s,clearance_m):
        if self.lift_failed:raise ValueError('clearance_lift_invalidated')
        lift=ClearanceLift(now_s,clearance_m,self.lift_used_m)
        lift.validate(now_s,self.visual_grasp(now_s))
        self.clearance_lift=lift

    def start(self,method,args,obs):
        if self.terminal_reason:return self._reply('rejected','episode_failed',obs)
        if self.method:return self._reply('rejected','operation_active',obs)
        fault=self._observation_fault(obs)
        if fault:return self._reply('rejected',fault,obs)
        fields={'wait':{'duration'},'pickup_box':{'object_id'},'lift_supported_box':set(),'raise_held_box':{'clearance_m'},'hold_box':{'duration'},'return_box_to_source':set(),'retreat_with_box':{'distance_m'},'move_with_box':{'distance_m'},'turn_with_box':{'yaw_rad'},'place_box':{'surface_id'}}
        if method not in fields or not isinstance(args,dict) or set(args)-fields[method]:
            return self._reply('rejected','invalid_request',obs)
        sensor_retreat=(method=='retreat_with_box' and self.sensor_fault is not None and self.sensor_retreat_factory is not None)
        sensor_turn=(method=='turn_with_box' and self.sensor_fault is not None and self.sensor_turn_factory is not None)
        sensor_raise=(method=='raise_held_box' and self.sensor_fault is not None and self.sensor_lift_ready is not None)
        if self.visual_grasp is not None and method not in ('wait','pickup_box','hold_box') and not (sensor_retreat or sensor_raise or sensor_turn):
            return self._reply('rejected','sensor_tool_unavailable',obs)
        if self.visual_grasp is not None and (method in ('pickup_box','hold_box') or sensor_retreat or sensor_raise or sensor_turn):
            if self.visual_grasp(obs['time'])['status']!='available':
                return self._reply('rejected','visual_state_unavailable',obs)
            if self.sensor_fault is not None:
                fault=self._geometry_fault(obs['time'])
                if fault:return self._reply('rejected',fault,obs)
        if method=='wait':
            duration=args.get('duration',1.)
            if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or not 0<duration<=5:
                return self._reply('rejected','duration_outside_envelope',obs)
            self.duration=duration
            phase='wait'
        elif method=='pickup_box':
            if args.get('object_id')!='brown_box':return self._reply('rejected','unsupported_object',obs)
            # Repeating an acquisition request is not evidence of a new grasp.
            # A retained visual grasp must use raise/hold, keeping its lift budget.
            if (self.sensor_lift_ready is not None and
                    self.visual_grasp(obs['time']).get('opposing_near_wrists',False)):
                return self._reply('rejected','retained_grasp_requires_raise_or_hold',obs)
            fault=self._geometry_fault(obs['time'])
            if fault:return self._reply('rejected',fault,obs)
            self.preparation=None;self.grasp_prepared=False
            self.clearance_lift=None;self.lift_used_m=0.;self.lift_failed=False
            phase='acquire'
        elif method=='lift_supported_box':
            if not supported_grasp_ready(list(self.history)):
                return self._reply('rejected','stable_supported_grasp_required',obs)
            self.supported_lift=SupportedLift(list(self.history),self.last_action,self.wrist_motion)
            phase=self.supported_lift.phase
        else:
            value=(args.get('distance_m') if method in ('retreat_with_box','move_with_box') else
                   args.get('yaw_rad') if method=='turn_with_box' else
                   args.get('clearance_m',.08) if method=='raise_held_box' else args.get('duration',1.))
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                return self._reply('rejected','invalid_parameter',obs)
            if method in ('retreat_with_box','move_with_box') and not .20<=value<=.50:
                return self._reply('rejected','distance_outside_envelope',obs)
            if method=='turn_with_box' and not 0<abs(value)<=math.pi/2:
                return self._reply('rejected','angle_outside_envelope',obs)
            if method=='raise_held_box' and not .05<=value<=.10:
                return self._reply('rejected','height_outside_envelope',obs)
            if method=='hold_box' and not 0<value<=3:
                return self._reply('rejected','duration_outside_envelope',obs)
            if not sensor_raise and not self._settled_grasp():return self._reply('rejected','settled_bilateral_grasp_required',obs)
            if sensor_raise:
                try:self._begin_sensor_lift(obs['time'],value)
                except ValueError as error:return self._reply('rejected',str(error),obs)
                phase='sensor_lift'
            elif method=='raise_held_box':
                distance=value-obs['clearance']
                if distance>.06:return self._reply('rejected','translation_outside_envelope',obs)
                if distance<=0:return self._reply('completed','height_already_satisfied',obs)
                self.motion=self.wrist_motion(obs,[0.,0.,distance])
                self.motion_duration=distance/.01+1.
                self.clearance_goal=value
                phase='lift'
            elif sensor_retreat or sensor_turn:
                factory=self.sensor_turn_factory if sensor_turn else self.sensor_retreat_factory
                try:self.loaded_motion=factory(obs['time'],list(self.last_action),value)
                except ValueError as error:return self._reply('rejected',str(error),obs)
                phase=self.loaded_motion.phase
            elif method in LOADED_METHODS:
                if not self._hold_metric(obs['time'])['ready']:return self._reply('rejected','clearance_below_goal',obs)
                if method=='retreat_with_box':
                    w,x,y,z=obs['root_quat'];yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
                    if abs(yaw)>math.radians(5):return self._reply('rejected','unsupported_heading',obs)
                    clearance=obs.get('approach_clearance_m')
                    if not isinstance(clearance,(int,float)) or not math.isfinite(clearance) or clearance<.03:
                        return self._reply('rejected','insufficient_starting_table_clearance',obs)
                else:
                    fault=space_fault(obs,turning=method=='turn_with_box',admission=True)
                    if fault:return self._reply('rejected',fault,obs)
                fault=preparation_contact_fault(obs)
                if fault:return self._reply('rejected',fault,obs)
                if self.grasp_prepared:phase=self._begin_loaded_motion(method,value,obs)
                else:
                    if self.preparation is not None:return self._reply('rejected','grasp_preparation_not_verified',obs)
                    self.preparation=GripPreparation(obs,self.last_action,self.wrist_motion)
                    self.loaded_motion=None
                    self.motion_value=value
                    phase='prepare_carry'
            elif method=='hold_box':
                if not self._hold_metric(obs['time'])['ready']:return self._reply('rejected','clearance_below_goal',obs)
                self.duration=value
                phase='hold'
            else:
                if not self._hold_metric(obs['time'])['ready']:return self._reply('rejected','clearance_below_goal',obs)
                surface_id='source' if method=='return_box_to_source' else args.get('surface_id')
                if self.placement_factory is None:return self._reply('rejected','placement_controller_unavailable',obs)
                try:self.placement=self.placement_factory(obs,self.last_action,surface_id)
                except ValueError as error:return self._reply('rejected',str(error),obs)
                phase=self.placement.phase
        self.method=method
        self.result=None
        self.acquired=0
        self.acquired_at=None
        self.last_action[43:46]=[0.,0.,0.]
        self._phase(phase,obs)
        return self._reply('running','accepted',obs)

    def _begin_loaded_motion(self,method,value,obs):
        if method=='retreat_with_box':self.loaded_motion=LoadedRetreat(obs,self.last_action,value)
        elif method=='move_with_box':self.loaded_motion=LoadedTranslation(obs,self.last_action,value)
        else:self.loaded_motion=LoadedTurn(obs,self.last_action,value)
        return self.loaded_motion.phase

    def command(self,obs):
        fault=self._observation_fault(obs)
        if fault:
            self.terminal_reason=fault
            self._finish('failed',fault,obs)
            return list(self.last_action)
        if self.sensor_fault is not None:
            fault=self._geometry_fault(obs['time'])
            if not fault and self.visual_grasp(obs['time'])['status']!='available':
                fault='visual_state_unavailable'
            if fault:
                if self.method in ('pickup_box','hold_box','retreat_with_box','raise_held_box','turn_with_box'):self._finish('failed',fault,obs)
                # Task-height precision can fail with collision edges intact.
                # Screen stopped support before ordinary hold can invalidate the
                # existing owner, including subsequent idle/wait ticks. Only the
                # separate callback may admit correction; task failure remains.
                return self._stopped_action(obs['time'])
        action=list(self.last_action)
        elapsed=(round((obs['time']-self.phase_started)/DT)+1)*DT
        if self.visual_grasp is not None and self.method in ('pickup_box','hold_box'):
            if self.visual_grasp(obs['time'])['status']!='available':
                self._finish('failed','visual_state_unavailable',obs)
                action=list(self.last_action)
        if self.phase=='acquire' or self.sensor_fault is not None and self.method in ('pickup_box','hold_box'):
            fault=self._geometry_fault(obs['time'])
            if fault:
                self._finish('failed',fault,obs)
                action=list(self.last_action)
        if self.phase=='acquire':action=self._action(self.acquisition(obs))
        elif self.phase=='sensor_lift':
            try:
                now=obs['time'];grasp=self.visual_grasp(now)
                self.clearance_lift.validate(now,grasp)
                if not self.sensor_lift_ready():
                    if now-self.phase_started>.150001:raise ValueError('lift_wrist_handoff_timeout')
                elif abs(now-grasp['time_s'])<1e-8:
                    self.clearance_lift.update(now,grasp)
                    self.lift_used_m=self.clearance_lift.offset
                action[43:46]=[0.,0.,0.]
            except ValueError as error:
                self._finish('failed',str(error),obs)
                return list(self.last_action)
        elif self.phase in ('supported_lift','verify_supported_lift'):action=self.supported_lift.command(obs)
        elif self.method in LOADED_METHODS and self.phase!='prepare_carry':
            if self.sensor_fault is not None:
                try:
                    action=self.loaded_motion.command(obs['time'],action)
                    self.phase=self.loaded_motion.phase
                except ValueError as error:
                    self._finish('failed',str(error),obs)
                    return list(self.last_action)
            else:action=self.loaded_motion.command(obs)
        elif self.method in ('place_box','return_box_to_source'):action=self.placement.command(obs)
        elif self.phase=='prepare_carry':action=self.preparation.command(obs)
        elif self.phase=='lift':
            root=obs['root_quat'];xyzw=[*root[1:],root[0]]
            action[:43]=self.motion.command(obs['joint_pos'],obs['root_pos'],xyzw,action[:43],elapsed)
        if self.scene_hold is not None and self.terminal_reason is None:
            try:action=self.scene_hold(self.phase,obs['time'],action)
            except ValueError as error:
                self._finish('failed',str(error),obs)
                action=list(self.last_action)
        self.last_action=self._action(action)
        return list(self.last_action)

    def update(self,obs):
        fault=self._observation_fault(obs)
        if self.sensor_fault is None and not fault and (obs['robot_source_peak_N']>5 or obs['tilt']>math.radians(15)):
            fault='forbidden_source_contact' if obs['robot_source_peak_N']>5 else 'body_tilt_limit'
        if self.sensor_fault is None and not fault and obs.get('box_floor_contact_peak_N',0.)>.5:fault='box_floor_contact'
        if self.sensor_fault is None and not fault:
            for name,surface in obs.get('surfaces',{}).items():
                if surface['robot_contact_peak_N']>5:fault='forbidden_'+name+'_contact';break
        if fault:
            self.terminal_reason=fault
            return self._finish('failed',fault,obs)
        if self.last_observation_time is not None:
            gap=obs['time']-self.last_observation_time
            if abs(gap)<1e-9:return self.result
            if gap<0 or abs(gap-DT)>1e-6:
                self.terminal_reason='missing_or_reordered_observation'
                return self._finish('failed',self.terminal_reason,obs)
        self.last_observation_time=obs['time']
        self.history.append(deepcopy(obs))
        if self.method is None:return self.result
        elapsed=round((obs['time']-self.phase_started)/DT)*DT
        visual=self.visual_grasp(obs['time']) if self.visual_grasp is not None else None
        if visual is not None and self.method in ('pickup_box','hold_box') and visual['status']!='available':
            return self._finish('failed','visual_state_unavailable',obs)
        if self.sensor_fault is not None and self.method in ('pickup_box','hold_box','retreat_with_box','raise_held_box','turn_with_box'):
            fault=self._geometry_fault(obs['time'])
            if fault:return self._finish('failed',fault,obs)
        if self.phase=='wait':
            if elapsed>=self.duration:return self._finish('completed','dwell_elapsed',obs)
        elif self.phase=='sensor_lift':
            try:self.clearance_lift.validate(obs['time'],visual)
            except ValueError as error:return self._finish('failed',str(error),obs)
            if self.clearance_lift.outcome:
                return self._finish(self.clearance_lift.outcome,self.clearance_lift.reason,obs)
            if elapsed>=8.:return self._finish('failed','clearance_lift_timeout',obs)
        elif self.phase in ('supported_lift','verify_supported_lift'):
            result=self.supported_lift.update(obs)
            self.phase=self.supported_lift.phase
            if result:return self._finish(*result,obs)
        elif self.phase=='prepare_carry':
            result=self.preparation.update(obs)
            if result:
                if result[0]!='completed':return self._finish(*result,obs)
                self.grasp_prepared=True
                if self.method!='retreat_with_box':
                    fault=space_fault(obs,turning=self.method=='turn_with_box',admission=True)
                    if fault:return self._finish('failed',fault,obs)
                phase=self._begin_loaded_motion(self.method,self.motion_value,obs)
                self._phase(phase,obs)
        elif self.method in LOADED_METHODS:
            if self.sensor_fault is None and self.method!='retreat_with_box':
                fault=space_fault(obs,turning=self.method=='turn_with_box')
                if fault:return self._finish('failed',fault,obs)
            if self.sensor_fault is not None:
                try:result=self.loaded_motion.update(obs['time'])
                except ValueError as error:return self._finish('failed',str(error),obs)
            else:result=self.loaded_motion.update(obs)
            self.phase=self.loaded_motion.phase
            if result:return self._finish(*result,obs)
        elif self.phase=='acquire':
            if self.sensor_fault is not None:
                # Two DISTINCT synchronized images admit stabilization, not
                # completion. An unavailable stance pose does not invalidate a
                # visible local grasp; the owner separately requires floor/IMU.
                observed_at=visual.get('time_s')
                fresh=(type(observed_at) in (int,float) and math.isfinite(observed_at)
                    and abs(observed_at-obs['time'])<=1e-8)
                if fresh and observed_at!=self.acquired_at:
                    continuous=(self.acquired_at is not None and
                        0<observed_at-self.acquired_at<=.150001)
                    self.acquired=(self.acquired+1 if continuous else 1) if visual['raised'] else 0
                    self.acquired_at=observed_at
                admitted=fresh and self.acquired>=2
            else:
                raised=(visual['raised'] and visual.get('scene_status','available')=='available') if visual is not None else obs['bilateral'] and obs['clearance']>=.05
                self.acquired=self.acquired+1 if raised else 0
                admitted=self.acquired>=15
            if admitted:
                self.last_action[43:46]=[0.,0.,0.]
                self._phase('verify_pickup',obs)
            elif elapsed>=15:return self._finish('failed','acquisition_timeout',obs)
        elif self.phase in ('verify_pickup','verify_lift'):
            metric=self._hold_metric(obs['time'])
            stance_ok=visual['attitude_ok'] if visual is not None else obs['stance_clear']
            if metric['ready'] and stance_ok and (visual is None or elapsed>=1.):
                if self.phase=='verify_lift' and obs['clearance']<self.clearance_goal-.005:
                    return self._finish('failed','height_goal_not_reached',obs)
                return self._finish('completed','visually_raised_and_stable' if visual is not None else 'stable_raised_grasp',obs)
            if (self.phase=='verify_pickup' and self.sensor_lift_ready is not None
                    and elapsed>=1. and visual is not None and visual.get('settled_retention',False)
                    and visual.get('gap_m',0.)<.055):
                try:self._begin_sensor_lift(obs['time'],.08)
                except ValueError as error:return self._finish('failed',str(error),obs)
                self._phase('sensor_lift',obs)
                return None
            if visual is None and self.phase=='verify_pickup' and elapsed>=2-1e-9 and self._settled_grasp() and obs['clearance']<.05:
                return self._finish('failed','clearance_below_goal',obs)
            if elapsed>=5:return self._finish('failed','pose_hold_timeout',obs)
        elif self.phase=='hold':
            retained=visual['raised'] if visual is not None else obs['bilateral'] and obs['stance_clear'] and obs['clearance']>=.05
            if not retained:
                return self._finish('failed','hold_contact_or_clearance_lost',obs)
            # Require one observed second even for a shorter requested dwell.
            if elapsed>=max(1.,self.duration) and self._hold_metric(obs['time'])['ready']:
                return self._finish('completed','hold_verified',obs)
            if elapsed>max(1.,self.duration)+2:return self._finish('failed','hold_did_not_settle',obs)
        elif self.phase=='lift':
            if not obs['stance_clear'] or not obs['bilateral'] or obs['clearance']<=.01:
                return self._finish('failed','lift_contact_or_stance_failure',obs)
            if elapsed>=self.motion_duration:self._phase('verify_lift',obs)
        elif self.method in ('place_box','return_box_to_source'):
            result=self.placement.update(obs)
            self.phase=self.placement.phase
            if result:return self._finish(*result,obs)
        return self.result
