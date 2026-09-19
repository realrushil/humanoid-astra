"""One native Arena world, observations and complete physical evidence.

Import after SimulationAppContext starts. This owner alone accesses Isaac state;
it has no program generation, worker management, recorded setup or reset API.
"""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time
import types

import imageio.v2 as imageio
import numpy as np
import torch
import isaaclab.sim as sim
from isaaclab.sensors import CameraCfg,ContactSensorCfg,ImuCfg

from .arena_control import BoxControl
from .arena_observation import assess_hold, loaded_contact_summary
from .arena_scene import simplify_scene, configure_task, validate_transfer_scene, spawn_transfer_box
from .arena_surfaces import box_bounds, support_observation, lower_body_clearance, turn_clearance, read_support_parts
from .toolkit.arena_approach import front_clearance,limit_navigation,GUARDED_BODIES
from .toolkit.arena_geometry import collision_geometry
from .toolkit.convex_clearance import ConvexClearance
from .toolkit.loaded_wrists import PairedWristController
from .toolkit.acquisition_wrists import AcquisitionHandClearance
from .arena_alignment import HeldAlignment
from .arena_retraction import PlacementRetraction
from .arena_placement import PreparedPlacement


def values(tensor):return tensor.detach().cpu().tolist()
def wxyz(tensor):
    x,y,z,w=values(tensor)[0]
    return [w,x,y,z]


class NativeTerminal(Exception):pass


class ArenaWorld:
    def __init__(self,builder,output,recipe,policy_port):
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        fixture=recipe.get('fixture','native_bin')
        scene=validate_transfer_scene(recipe.get('scene')) if fixture=='box_transfer_v1' else None
        if scene is None and 'scene' in recipe:raise ValueError('scene requires box_transfer_v1')
        if scene is not None and recipe.get('task_kind')!='table_transfer':
            raise ValueError('box_transfer_v1 requires table_transfer scoring')
        if scene is not None and (recipe.get('box_offset_y_m',0.) or any(recipe.get('destination_offset_xy_m',(0.,0.)))):
            raise ValueError('explicit scene cannot use legacy offsets')
        self.box_size=list(scene['box']['size_m']) if scene else [.2,.2,.2]
        self.box_mass=scene['box']['mass_kg'] if scene else .1
        configure_task(builder,fixture)
        cfg,kwargs=builder.compose_manager_cfg()
        record_sensors=recipe.get('record_sensors',False)
        sensor_balance=recipe.get('sensor_balance',False)
        visual_grasp_checks=recipe.get('visual_grasp_checks',False)
        arm_gravity=recipe.get('arm_gravity_compensation',False)
        rgbd_period_steps=recipe.get('rgbd_period_steps',5)
        if type(rgbd_period_steps) is not int or rgbd_period_steps<=0:
            raise ValueError('rgbd_period_steps must be a positive integer')
        if any(type(v) is not bool for v in (record_sensors,sensor_balance,visual_grasp_checks,arm_gravity)):
            raise ValueError('sensor mode flags must be boolean')
        if arm_gravity and not visual_grasp_checks:
            raise ValueError('arm gravity compensation requires the body/Dex3 visual sensor track')
        sensor_balance=sensor_balance or visual_grasp_checks
        record_sensors=record_sensors or sensor_balance
        if sensor_balance:
            from .arena_sensor_action import SensorJointAction
            terms=[term for term in vars(cfg.actions).values()
                   if getattr(getattr(term,'class_type',None),'__name__','')=='G1DecoupledWBCJointAction']
            if len(terms)!=1:raise ValueError('expected one native joint action term')
            terms[0].class_type=SensorJointAction
        self.sensor_recorder=None
        self.box_perception=None
        self.approach=None
        self.observed_hand=None
        self.sensor_guard=None
        self.loaded_stop_clearance=None
        self.loaded_stop_evidence_error=None
        self.loaded_stop_feedback=dict(status='inactive',time_s=0.)
        self.scene_wrist_hold=None
        self.scene_wrist_failure=None
        self.floor_hold_pending=False
        self.floor_hold_active=False
        self.lift_anchor_m=0.
        if record_sensors:
            # Permitted measurements for the opt-in sensor control track.
            cfg.scene.robot_head_cam.data_types=['rgb','distance_to_image_plane']
            native_camera_period_s=cfg.scene.robot_head_cam.update_period
            capture_period_s=rgbd_period_steps*.02
            if (not math.isfinite(native_camera_period_s) or native_camera_period_s<0 or
                    native_camera_period_s>capture_period_s+1e-9 or
                    (native_camera_period_s>0 and abs(capture_period_s/native_camera_period_s-
                                                    round(capture_period_s/native_camera_period_s))>1e-8)):
                raise ValueError('native camera update period does not support requested RGB-D capture period')
            cfg.scene.body_imu=ImuCfg(prim_path='{ENV_REGEX_NS}/Robot/pelvis',update_period=0.)
            # Reset moves the box after renderer startup. Render before the
            # native observation manager reads/caches the initial head image;
            # otherwise the step-zero RGB-D can still show the pre-reset scene.
            # Rendering advances no physics and introduces no settling actions.
            cfg.num_rerenders_on_reset=max(1,cfg.num_rerenders_on_reset)
        deadline=recipe.get('deadline',60.)
        if not math.isfinite(deadline) or not 0<deadline<=180:raise ValueError('episode deadline must be in (0,180] simulation seconds')
        # Native Galileo defaults to 30 s. Generation and revisions consume the
        # same episode, so propagate our declared budget before world creation.
        cfg.episode_length_s=deadline
        cfg.seed=recipe.get('seed',42)
        offset=recipe.get('box_offset_y_m',0.)
        if not math.isfinite(offset) or abs(offset)>.03:raise ValueError('box Y offset must be within 3 cm')
        changes=[]
        for name,event in vars(cfg.events).items():
            params=getattr(event,'params',{})
            if 'pose_range' in params and any(asset.name=='brown_box' for asset in params.get('asset_cfgs',[])):
                before=list(params['pose_range']['y'])
                if scene:
                    spec=scene['box'];jitter=spec['xy_jitter_m']
                    old=dict(params['pose_range'])
                    x,y=spec['center_xy'];z=scene['tables']['source']['top_z']+self.box_size[2]/2+.0007
                    yaw=math.pi+spec['yaw_offset_rad']
                    params['pose_range'].update(x=(x-jitter,x+jitter),y=(y-jitter,y+jitter),z=(z,z),
                                               roll=(math.pi,math.pi),pitch=(0.,0.),yaw=(yaw,yaw))
                    changes.append(dict(event=name,before=old,after=dict(params['pose_range'])))
                else:
                    params['pose_range']['y']=tuple(y+offset for y in before)
                    changes.append(dict(event=name,before_y_m=before,after_y_m=params['pose_range']['y']))
        if len(changes)!=1:raise ValueError('expected one native brown-box reset event')
        self.write('fixture-variation.json',dict(world_y_offset_m=offset,seed=cfg.seed,reset_changes=changes))
        if scene:
            # Scale the actual native USD collision/render asset before spawning.
            # The runtime checks measured collision dimensions and PhysX mass.
            cfg.scene.brown_box.spawn.scale=tuple(v/.2 for v in self.box_size)
            cfg.scene.brown_box.spawn.func=spawn_transfer_box
            cfg.scene.brown_box.spawn.mass_props=sim.MassPropertiesCfg(mass=self.box_mass,density=0.)
        self.scene_description=simplify_scene(cfg,fixture,recipe.get('destination_offset_xy_m',(0.,0.)),scene=scene)
        self.scene_description['box']=dict(size_m=self.box_size,mass_kg=self.box_mass)
        self.write('scene.json',self.scene_description)
        # Native acquisition proposals retain the released policy's conditioning.
        # The application task is exposed separately to the coding agent.
        cfg.task_description='Pick up the brown box from the source table and place it into the blue bin on the destination table.'
        cfg.scene.overview=CameraCfg(prim_path='{ENV_REGEX_NS}/OverviewCamera',update_period=0.,width=960,height=720,
            data_types=['rgb'],spawn=sim.PinholeCameraCfg(focal_length=15.,clipping_range=(.1,30.)))
        box=cfg.scene.contact_sensor_brown_box.prim_path
        for side in ('left','right'):
            setattr(cfg.scene,side+'_box_contact',ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+side+'_hand_.*',update_period=0.,filter_prim_paths_expr=[box]))
            setattr(cfg.scene,side+'_foot_contact',ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+side+'_ankle_roll_link',update_period=0.,filter_prim_paths_expr=['/World/SimpleFloor/geometry/mesh']))
        self.source_parts=['source_top',*[f'source_leg_{i}' for i in range(4)]]
        cfg.scene.box_source_contact=ContactSensorCfg(prim_path=box,update_period=0.,filter_prim_paths_expr=['{ENV_REGEX_NS}/source_top/geometry/mesh'])
        cfg.scene.robot_source_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',update_period=0.,filter_prim_paths_expr=['{ENV_REGEX_NS}/'+n+'/geometry/mesh' for n in self.source_parts])
        destination_parts=['destination_top',*[f'destination_leg_{i}' for i in range(4)]]
        cfg.scene.box_destination_contact=ContactSensorCfg(prim_path=box,update_period=0.,filter_prim_paths_expr=['{ENV_REGEX_NS}/destination_top/geometry/mesh'])
        cfg.scene.robot_destination_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',update_period=0.,filter_prim_paths_expr=['{ENV_REGEX_NS}/'+n+'/geometry/mesh' for n in destination_parts])
        cfg.scene.box_floor_contact=ContactSensorCfg(prim_path=box,update_period=0.,filter_prim_paths_expr=['/World/SimpleFloor/geometry/mesh'])
        if fixture!='native_bin':
            assert not hasattr(cfg.scene,'blue_sorting_bin') or cfg.scene.blue_sorting_bin is None
            assert 'blue_sorting_bin' not in repr(cfg.events) and cfg.terminations.success is None
        if visual_grasp_checks:
            from .arena_scene import sensor_terminations
            sensor_terminations(cfg.terminations)
        self.env=builder.make_registered(cfg,kwargs)
        self.native=self.env.unwrapped
        self.native.episode_recorder.set_output_path(str(self.output/'native-results.jsonl'))
        self.step_index=0;self.phase='idle';self.native_terminal=False
        self.contacts=[];self.history=deque(maxlen=51);self.clearances=deque(maxlen=6)
        self.writers={name:imageio.get_writer(str(self.output/(name+'.mp4')),fps=50) for name in ('head','overview')}
        self.files={name:(self.output/(name+'.jsonl')).open('w') for name in ('states','actions','physics-contacts')}
        self.obs,_=self.env.reset()
        native=self.native
        native.scene['overview'].set_world_poses_from_view(torch.tensor([[-1.5,-2.8,1.3]],device=native.device),torch.tensor([[.1,-.65,.05]],device=native.device))
        if scene:
            tables=scene['tables'];target=[.1,sum(t['center_xy'][1] for t in tables.values())/2,.0]
            span=max(2.,abs(tables['source']['center_xy'][1]-tables['destination']['center_xy'][1])+1.)
            eye=[target[0]-span,target[1]-span,target[2]+.9*span]
            native.scene['overview'].set_world_poses_from_view(torch.tensor([eye],device=native.device),torch.tensor([target],device=native.device))
        native.sim.render();native.scene['overview'].update(dt=0.,force_recompute=True)
        import omni.usd
        from pxr import UsdPhysics
        stage=omni.usd.get_context().get_stage()
        paths=['/World/SimpleFloor/geometry/mesh',*['/World/envs/env_0/'+n+'/geometry/mesh' for n in self.source_parts]]
        assert all(stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI) for path in paths)
        self.geometry=collision_geometry(stage,native.scene['robot'])
        self.write('collision-geometry.json',self.geometry)
        self.support_parts=read_support_parts(stage)
        self.placement_clearance=ConvexClearance(self.geometry,self.support_parts)
        for name,parts in self.support_parts.items():
            if abs(parts[0]['max'][2]-self.scene_description['tables'][name]['top_z'])>1e-6:
                raise ValueError('built support height differs from fixture: '+name)
        self.write('support-geometry.json',self.support_parts)
        if scene:
            for camera in ('head','overview'):
                sensor='robot_head_cam' if camera=='head' else 'overview'
                rgb=native.scene[sensor].data.output['rgb'][0].detach().cpu().numpy()
                imageio.imwrite(self.output/('initial-'+camera+'.png'),rgb)
            self.write('built-box.json',self.verify_box(stage))

        self.names=native.scene['robot'].joint_names
        assert len(self.names)==43 and not native._physics_handles_decimation
        self.original_update=native.scene.update
        self.last_physics_step=native.sim.get_physics_step_count()
        native.scene.update=types.MethodType(lambda scene,dt:self._physics_update(dt),native.scene)
        def reject_reset(env,ids):
            if len(ids):
                self.write('native-terminal.json',dict(step=self.step_index,
                    timeouts=values(env.reset_time_outs),terminated=values(env.reset_terminated),
                    terms={name:values(env.termination_manager.get_term(name)) for name in env.termination_manager.active_terms}))
                raise NativeTerminal('native reset blocked; episode consequences retained')
        native._reset_idx=types.MethodType(reject_reset,native)
        self.term=next(native.action_manager.get_term(n) for n in native.action_manager.active_terms if hasattr(native.action_manager.get_term(n),'robot_model'))
        if record_sensors:
            from .arena_sensor_recording import SensorRecorder
            self.sensor_recorder=SensorRecorder(self.output/'sensors',native,self.term.robot_model,
                                                sensor_balance=sensor_balance,hand_positions=visual_grasp_checks,
                                                arm_gravity=arm_gravity,rgbd_period_steps=rgbd_period_steps)
        if visual_grasp_checks:
            from .arena_perception import ArenaBoxPerception
            from .observed_approach import ApproachEstimate
            from .observed_hand import HandClearanceEstimate
            from .sensor_robot_geometry import RobotGeometry
            self.box_perception=ArenaBoxPerception(self.term.sensor_model)
            assets=Path(__file__).parent/'assets'
            self.approach_geometry=RobotGeometry(assets/'arena_g1_rev1_0_kinematics.urdf',
                assets/'arena_g1_rev1_0_bounds.json',self.sensor_recorder.names)
            from .loaded_stop import LoadedStopClearance
            self.loaded_stop_clearance=LoadedStopClearance(assets/'arena_g1_rev1_0_kinematics.urdf',
                assets/'arena_g1_rev1_0_bounds.json',self.names)
            self.approach=ApproachEstimate()
            self.observed_hand=HandClearanceEstimate()
            from .sensor_guard import SensorGuard
            self.sensor_guard=SensorGuard(self.names,self.sensor_recorder.effort_limits)
            self.files['sensor-guard']=(self.output/'sensor-guard.jsonl').open('w')
            self.files['sensor-hand-clearance']=(self.output/'sensor-hand-clearance.jsonl').open('w')
            self.files['sensor-approach']=(self.output/'sensor-approach.jsonl').open('w')
            self.files['gr00t-inputs']=(self.output/'gr00t-inputs.jsonl').open('w')
            self.files['visual-grasp']=(self.output/'visual-grasp.jsonl').open('w')
            self.files['visual-timing']=(self.output/'visual-timing.jsonl').open('w')
            self.files['scene-hold']=(self.output/'scene-hold.jsonl').open('w')
            self.files['loaded-stop']=(self.output/'loaded-stop.jsonl').open('w')
        self.raw=self.capture()
        if sensor_balance:
            self.term.sensor_source=self.sensor_recorder.measurements
            self.files['wbc-inputs']=(self.output/'wbc-inputs.jsonl').open('w')
        if arm_gravity:
            from .arm_gravity import ArmGravity
            masses=json.loads((Path(__file__).parent/'assets/arena_g1_arm_mass.json').read_text())['links']
            stiffness={n:k for n,k in self.sensor_recorder.body_stiffness.items()
                       if any(part in n for part in ('shoulder','elbow','wrist'))}
            self.term.arm_gravity=ArmGravity(self.term.sensor_model,self.names,stiffness,masses)
        from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import Gr00tRemoteClosedloopPolicy,Gr00tRemoteClosedloopPolicyCfg
        self.policy=Gr00tRemoteClosedloopPolicy(Gr00tRemoteClosedloopPolicyCfg(policy_config_yaml_path='isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml',policy_device='cpu',remote_host='127.0.0.1',remote_port=policy_port,num_envs=1))
        self.policy.reset();self.policy.set_task_description(native.get_language_instruction())
        if visual_grasp_checks:
            from .hand_sensors import measured_joint_positions
            initial_q=measured_joint_positions(self.sensor_recorder.latest_packet,self.sensor_recorder.latest_hands,self.names)
        else:initial_q=values(native.scene['robot'].data.joint_pos)[0]
        initial=initial_q+[0.,0.,0.,.75,0.,0.,0.]
        self.control=BoxControl(initial,self.acquire,self.wrist_motion,placement_factory=self.placement_controller,
            visual_grasp=self.box_perception.feedback if self.box_perception is not None else None,
            scene_hold=self.scene_hold_command if self.box_perception is not None else None,
            approach_feedback=self.approach.feedback if self.approach is not None else None,
            hand_clearance_feedback=self.observed_hand.feedback if self.observed_hand is not None else None,
            sensor_fault=self.sensor_guard.fault if self.sensor_guard is not None else None,
            sensor_retreat_factory=self.sensor_retreat_controller if self.sensor_guard is not None else None,
            sensor_turn_factory=self.sensor_turn_controller if self.sensor_guard is not None else None,
            loaded_stop=self.loaded_stop_command if self.sensor_guard is not None else None,
            sensor_lift_ready=(lambda:self.floor_hold_active and not self.floor_hold_pending
                and self.scene_wrist_failure is None) if self.sensor_guard is not None else None)
        self.raw_policy=None;self.guard=None;self.input_hash=None
        self.hand_clearance=None;self.hand_guard=None
        self.write('metadata.json',dict(backend='arena_homie_gr00t',physics_dt=native.physics_dt,step_dt=native.step_dt,
            contact_observation_rate_Hz=200,recorded_setup_actions=0,gr00t_seed=recipe.get('gr00t_seed',0),
            joint_names=self.names,guarded_bodies=list(GUARDED_BODIES),box_mass_kg=self.box_mass,box_size_m=self.box_size,
            policy_instruction=native.get_language_instruction(),observation_source='simulator_ground_truth',
            sensor_balance=sensor_balance,visual_grasp_checks=visual_grasp_checks,
            arm_gravity_compensation=arm_gravity,
            control_track='sensor_loaded_turn_v5' if visual_grasp_checks else 'privileged_outer_controller',
            initial_action=initial,source_parts=self.source_parts))
        self.write('timing.json',dict(declared_deadline_s=deadline,native_episode_length_s=cfg.episode_length_s,
            native_max_episode_steps=native.max_episode_length,neural_inference='step_synchronous',
            waiting_between_programs='physics_advances'))

    def write(self,name,value):
        (self.output/name).write_text(json.dumps(value,indent=2,allow_nan=False))

    def verify_box(self,stage):
        """Check spawned collision dimensions and physical mass, before actions.

        Root USD scaling is outside a relative bounding box. Include each root
        transform basis length; do not mistake visual-only scale for physics.
        """
        from pxr import Usd,UsdGeom,UsdPhysics,Gf
        asset=self.native.scene['brown_box']
        prim=stage.GetPrimAtPath(self.native.scene['contact_sensor_brown_box'].body_physx_view.prim_paths[0])
        cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'],
                               useExtentsHint=False,ignoreVisibility=True)
        shapes=[]
        for child in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
            if child.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get() is not False:
                bound=cache.ComputeRelativeBound(child,prim).ComputeAlignedBox()
                if not bound.IsEmpty():shapes.append(bound)
        if not shapes:raise ValueError('box has no collision geometry')
        matrix=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        scales=[matrix.TransformDir(Gf.Vec3d(*(1. if i==j else 0. for i in range(3)))).GetLength() for j in range(3)]
        low=[min(b.GetMin()[i] for b in shapes)*scales[i] for i in range(3)]
        high=[max(b.GetMax()[i] for b in shapes)*scales[i] for i in range(3)]
        size=[high[i]-low[i] for i in range(3)]
        # Arena's PhysX view uses Warp arrays; convert before indexing/serializing.
        mass=float(asset.root_physx_view.get_masses().numpy().reshape(-1)[0])
        evidence=dict(requested_size_m=self.box_size,collision_size_m=size,collision_min_local_m=low,
                      collision_max_local_m=high,requested_mass_kg=self.box_mass,physx_mass_kg=mass,
                      physx_inertia=asset.root_physx_view.get_inertias().numpy().tolist(),checked_before_actions=True,
                      matches_recipe=all(abs(a-b)<=.001 for a,b in zip(size,self.box_size))
                          and all(abs(a+b)<=.001 for a,b in zip(low,high)) and abs(mass-self.box_mass)<=1e-5)
        self.write('built-box.json',evidence)
        if any(abs(a-b)>.001 for a,b in zip(size,self.box_size)) or any(abs(a+b)>.001 for a,b in zip(low,high)):
            raise ValueError('spawned box collision dimensions/center differ from recipe')
        if abs(mass-self.box_mass)>1e-5:raise ValueError('spawned box mass differs from recipe')
        return evidence

    def force(self,name):return self.native.scene[name].data.force_matrix_w
    def force_sum(self,name):return float(self.force(name).norm(dim=-1).sum())
    def upward(self,name):return float(self.force(name)[...,2].sum())

    def _physics_update(self,dt):
        self.original_update(dt)
        counter=self.native.sim.get_physics_step_count()
        if counter==self.last_physics_step:return
        assert counter==self.last_physics_step+1 and abs(dt-.005)<1e-9
        self.last_physics_step=counter
        norms=self.force('robot_source_contact').norm(dim=-1)[0]
        index=int(norms.reshape(-1).argmax())
        destination_norms=self.force('robot_destination_contact').norm(dim=-1)[0]
        destination_index=int(destination_norms.reshape(-1).argmax())
        destination_peak=float(destination_norms.reshape(-1)[destination_index])
        # Contact filters are ordered top, leg_0..leg_3. These per-pair physics
        # diagnostics identify collisions; they are not extra agent observations.
        row=dict(physics_step=counter,action_step=self.step_index,phase=self.phase,
            robot_source_contact_N=float(norms.sum()),peak_body=self.native.scene['robot_source_contact'].body_names[index//5],
            peak_source_part_index=index%5,peak_pair_N=float(norms.reshape(-1)[index]),
            hand_forces_N={s:self.force_sum(s+'_box_contact') for s in ('left','right')},
            box_source_upward_N=self.upward('box_source_contact'),
            box_destination_upward_N=self.upward('box_destination_contact'),
            robot_destination_contact_N=float(destination_norms.sum()),
            peak_destination_body=self.native.scene['robot_destination_contact'].body_names[destination_index//5] if destination_peak>0 else None,
            peak_destination_part_index=destination_index%5 if destination_peak>0 else None,
            peak_destination_pair_N=destination_peak,
            box_floor_contact_N=self.force_sum('box_floor_contact'),foot_upward_N={s:self.upward(s+'_foot_contact') for s in ('left','right')})
        self.contacts.append(row)
        self.files['physics-contacts'].write(json.dumps(row)+'\n')

    def capture(self):
        self.camera_captured_at_unix_s=time.time()
        if self.sensor_recorder is not None:
            self.sensor_recorder.capture(self.step_index,self.step_index*.02)
        if self.box_perception is not None:
            measured=self.sensor_recorder.measurements()
            packet,camera=measured['proprioception'],measured['rgbd']
            self.box_perception.advance_imu(packet['time_s'],packet['gyro_rad_s'])
            self.approach.advance_imu(packet['time_s'],packet['gyro_rad_s'],
                self.approach_geometry.lower_body_points(packet))
            from .observed_hand import hand_shapes,observe_hand_planes
            self.observed_hand.advance(packet['time_s'],packet['gyro_rad_s'],
                hand_shapes(self.approach_geometry.model,self.approach_geometry.bounds,
                            packet,self.sensor_recorder.latest_hands))
            if camera['step']==packet['step']:
                started=time.perf_counter()
                visual=self.box_perception.update(packet,camera,self.term.up_at_measurement(packet,camera))
                from .observed_approach import observe_front
                front=observe_front(self.approach_geometry,self.approach,packet,camera,visual['height_reference'],visual['up_body'])
                observe_hand_planes(self.observed_hand,self.approach_geometry.model,packet,camera,visual['height_reference'],front)
                self.files['sensor-approach'].write(json.dumps(dict(step=self.step_index,
                    time_s=packet['time_s'],measurement=front,feedback=self.approach.feedback(packet['time_s'])),allow_nan=False)+'\n')
                self.files['sensor-approach'].flush()
                self.files['visual-timing'].write(json.dumps(dict(step=self.step_index,
                    time_s=packet['time_s'],processing_wall_s=time.perf_counter()-started))+'\n')
                self.files['visual-timing'].flush()
                self.files['visual-grasp'].write(json.dumps(visual,allow_nan=False)+'\n')
                self.files['visual-grasp'].flush()
            self.loaded_stop_evidence_error=None
            try:
                source=self.box_perception.source_plane
                self.loaded_stop_clearance.measure(packet,self.box_perception.carry_seed,
                    self.box_perception.motion.rotation,self.term.up_at_measurement(packet,camera),
                    source.geometry if source is not None else None)
            except ValueError as error:
                self.loaded_stop_evidence_error=str(error)
                self.loaded_stop_clearance.plane=None
                self.loaded_stop_clearance.retention=None
            self.files['sensor-hand-clearance'].write(json.dumps(dict(step=self.step_index,
                feedback=self.observed_hand.feedback(packet['time_s'])),allow_nan=False)+'\n')
            self.files['sensor-hand-clearance'].flush()
            guard=self.sensor_guard.update(packet,self.sensor_recorder.latest_hands,
                self.term.up_at_measurement(packet,camera).tolist(),
                self.approach.feedback(packet['time_s']),self.observed_hand.feedback(packet['time_s']))
            self.files['sensor-guard'].write(json.dumps(dict(step=self.step_index,**guard),allow_nan=False)+'\n')
            self.files['sensor-guard'].flush()
        robot,box=self.native.scene['robot'],self.native.scene['brown_box']
        row=dict(step=self.step_index,time=self.step_index*.02,physics_step=self.native.sim.get_physics_step_count(),phase=self.phase,
            state_age_s=0.,root_pos=values(robot.data.root_pos_w)[0],root_quat=wxyz(robot.data.root_quat_w),
            joint_pos=values(robot.data.joint_pos)[0],box_pos=values(box.data.root_pos_w)[0],box_quat=wxyz(box.data.root_quat_w),
            box_size_m=self.box_size,box_mass_kg=self.box_mass,
            box_solver_velocity=values(box.data.root_vel_w)[0],root_solver_velocity=values(robot.data.root_vel_w)[0])
        positions,rotations=values(robot.data.body_pos_w)[0],values(robot.data.body_quat_w)[0]
        row['body_poses']={n:dict(pos=positions[i],xyzw=rotations[i]) for i,n in enumerate(robot.body_names)}
        row['wrists_world']={n:row['body_poses'][n] for n in ('left_wrist_yaw_link','right_wrist_yaw_link')}
        clearance=front_clearance(self.geometry,row['body_poses'])
        self.clearances.append((row['time'],clearance))
        old_time,old_clearance=self.clearances[0]
        row['approach_clearance_m']=clearance
        row['approach_closing_speed_m_s']=(old_clearance-clearance)/(row['time']-old_time) if row['time']>old_time else 0.
        row['box_bounds']=box_bounds(row)
        row['clearance']=row['box_bounds']['min'][2]-self.scene_description['tables']['source']['top_z']
        w,x,y,z=row['root_quat'];row['tilt']=math.acos(max(-1.,min(1.,1-2*(x*x+y*y))))
        row['hand_forces_N']={s:self.force_sum(s+'_box_contact') for s in ('left','right')}
        row['bilateral']=all(v>.5 for v in row['hand_forces_N'].values())
        row['foot_upward_N']={s:self.upward(s+'_foot_contact') for s in ('left','right')}
        row['box_source_upward_N']=self.upward('box_source_contact')
        row['robot_source_contact_N']=self.force_sum('robot_source_contact')
        row['robot_source_peak_N']=max((s['robot_source_contact_N'] for s in self.contacts),default=row['robot_source_contact_N'])
        row['loaded_contacts']=loaded_contact_summary(self.contacts)
        row['supported']=row['box_source_upward_N']>=.5*self.box_mass*9.81 and abs(row['clearance'])<=.01
        row['stance_clear']=all(v>5 for v in row['foot_upward_N'].values()) and row['robot_source_peak_N']<=5 and row['tilt']<=math.radians(15)
        # One atomic state/camera record contains both selected-support observations.
        # Source compatibility fields above retain their source-relative meaning.
        row['surfaces']={}
        for name,parts in self.support_parts.items():
            current=self.force_sum('robot_'+name+'_contact')
            peak=max([current]+[sample['robot_'+name+'_contact_N'] for sample in self.contacts])
            row['surfaces'][name]=support_observation(row['box_bounds'],parts[0],
                self.upward('box_'+name+'_contact'),peak,lower_body_clearance(self.geometry,row,parts),mass_kg=self.box_mass)
        row['box_floor_contact_peak_N']=max([self.force_sum('box_floor_contact')]+
                                          [sample['box_floor_contact_N'] for sample in self.contacts])
        row['turn_clearance_m']=turn_clearance(self.geometry,row,[part for parts in self.support_parts.values() for part in parts])
        row['stance_clear']=row['stance_clear'] and all(s['robot_contact_peak_N']<=5 for s in row['surfaces'].values())
        self.history.append(row);row['hold_observation_v2']=assess_hold(list(self.history))
        self.camera_rgb={}
        for name,sensor in [('head','robot_head_cam'),('overview','overview')]:
            rgb=self.native.scene[sensor].data.output['rgb'][0].detach().cpu().numpy()
            self.camera_rgb[name]=rgb[:,:,:3].copy()
            row[name+'_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest()
            self.writers[name].append_data(rgb)
        self.files['states'].write(json.dumps(row)+'\n');self.files['states'].flush()
        return row

    def controller_observation(self):
        """Strict sensor controller boundary; raw state is reserved for evidence."""
        if self.sensor_guard is None:return self.raw
        return {'time':self.sensor_recorder.latest_packet['time_s']}

    def sensor_observation(self,now_s):
        """Physics-owner callback; never expose the raw evaluator observation."""
        from .arena_public import sensor_observation
        from .destination_surface import (destination_direction_segment, green_destination_mask,
                                           observe_destination_surface)
        from .sensor_kinematics import camera_in_body
        packet=self.sensor_recorder.latest_packet
        if abs(packet['time_s']-now_s)>1e-8:raise ValueError('public sensor timestamp mismatch')
        result=sensor_observation(packet,self.box_perception.track.observe(now_s),
            self.box_perception.feedback(now_s),self.box_perception.latest_motion,
            loaded_stop=self.loaded_stop_feedback)
        camera=self.sensor_recorder.latest_rgbd
        try:
            if (not math.isfinite(camera['time_s']) or
                    not 0 <= packet['time_s']-camera['time_s'] <= .150001):
                raise ValueError('destination_rgbd_stale')
            destination=observe_destination_surface(camera['time_s'],camera['rgb'],
                green_destination_mask(camera['rgb']),camera['depth_m'],camera['calibration']['intrinsic'])
            if destination.get('status') == 'observed_destination_candidate':
                joints=dict(zip(packet['joint_names'],packet['q_rad'],strict=True))
                camera_pose=camera_in_body(self.approach_geometry.model,joints,camera['calibration'])
                motion=self.box_perception.latest_motion
                if motion is None or motion.get('status') != 'tracked_local_segment':
                    raise ValueError('destination_motion_unavailable')
                direction=destination_direction_segment(destination,camera_pose[:3,:3],
                    np.asarray(motion['body_in_segment'])[:3,:3])
                destination['direction_segment_xy']=direction
                destination['direction_frame']='local_stance_segment'
        except ValueError as error:
            destination=dict(status='unavailable',reason='destination_sensor_invalid')
        result['destination']=destination
        return result

    def export_camera_frames(self,directory):
        """Owner thread only: export the last recorded step, without stepping physics."""
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        frames=[]
        for camera,rgb in self.camera_rgb.items():
            if self.box_perception is not None and camera!='head':continue
            path=directory/(camera+'.png')
            imageio.imwrite(path,rgb)
            frames.append(dict(camera=camera,file=path.name,width=rgb.shape[1],height=rgb.shape[0],
                encoded_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),source='live_camera'))
        return frames

    def acquire(self,observation):
        if self.box_perception is not None:
            from .hand_sensors import acquisition_joint_sample
            names=[name for name,_ in sorted(self.policy.robot_state_joints_config.items(),key=lambda item:item[1])]
            camera=self.sensor_recorder.latest_rgbd
            sample=acquisition_joint_sample(self.sensor_recorder.latest_packet,
                self.sensor_recorder.latest_hands,camera,names)
            rgb=camera['rgb']
            self.input_hash=hashlib.sha256(rgb.tobytes()).hexdigest()
            # The installed native adapter reads only these two fields; do not
            # give it the full simulator observation or an environment handle.
            policy_input=dict(policy=dict(robot_joint_pos=torch.tensor([sample['q_rad']],dtype=torch.float32)),
                camera_obs=dict(robot_head_cam_rgb=torch.as_tensor(rgb.copy()).unsqueeze(0)))
            record=dict(sample,rgb_sha256=self.input_hash,call_kind='local_policy_step_not_necessarily_network_fetch')
            self.files['gr00t-inputs'].write(json.dumps(record,allow_nan=False)+'\n');self.files['gr00t-inputs'].flush()
            self.raw_policy=values(self.policy.get_action(None,policy_input))[0]
        else:
            self.input_hash=hashlib.sha256(self.obs['camera_obs']['robot_head_cam_rgb'][0].cpu().numpy().tobytes()).hexdigest()
            self.raw_policy=values(self.policy.get_action(self.env,self.obs))[0]
        action=list(self.raw_policy)
        if self.approach is not None:
            action[43:46],self.guard=self.approach.limit(action[43:46],observation['time'])
        else:
            action[43:46],self.guard=limit_navigation(action[43:46],observation['root_quat'],observation['approach_clearance_m'],observation['approach_closing_speed_m_s'])
        if self.hand_clearance is None:
            self.hand_clearance=AcquisitionHandClearance(self.term.robot_model,self.names,
                self.geometry if self.observed_hand is None else None)
        if self.observed_hand is None:
            action=self.hand_clearance.command(observation,action)
        else:
            from .hand_sensors import measured_joint_positions
            packet=self.sensor_recorder.latest_packet
            joints=measured_joint_positions(packet,self.sensor_recorder.latest_hands,self.names)
            action=self.hand_clearance.command_measured(joints,action,self.observed_hand.feedback(packet['time_s']))
        self.hand_guard=self.hand_clearance.last
        return action

    def loaded_stop_command(self,now_s,previous):
        """Zero navigation, same loaded wrist owner, separately measured clearance."""
        result=list(previous);result[43:46]=[0.,0.,0.]
        try:
            fault=self.sensor_guard.fault(now_s)
            if fault:raise ValueError(fault)
            if self.scene_wrist_hold is None:raise ValueError('loaded_stop_wrist_owner_unavailable')
            if self.loaded_stop_evidence_error:raise ValueError(self.loaded_stop_evidence_error)
            return self.scene_hold_command('stopped',now_s,result)
        except ValueError as error:
            self.loaded_stop_feedback=dict(status='failed',time_s=now_s,reason=str(error))
            raise
        finally:
            self.files['loaded-stop'].write(json.dumps(self.loaded_stop_feedback,allow_nan=False)+'\n')
            self.files['loaded-stop'].flush()

    def scene_hold_command(self,phase,now_s,previous):
        """Sensor-only paired hold; actuator ownership remains in native HOMIE.

        Anchor at a synchronized camera/encoder frame after acquisition. In
        sensor control, update at IMU/encoder rate with aged camera height.
        Continue the same hold across idle/wait/program boundaries. A failed
        correction retains references and requires a new acquisition to reanchor.
        """
        from .scene_wrist_hold import SceneWristHold
        stopping=phase=='stopped'
        self.loaded_stop_feedback=dict(status='inactive',time_s=now_s)
        if phase=='acquire':
            self.scene_wrist_hold=None;self.scene_wrist_failure=None
            self.floor_hold_pending=False;self.floor_hold_active=False
            self.box_perception.begin_acquisition()
            return previous
        if self.scene_wrist_failure is not None:
            if phase in ('verify_pickup','hold','sensor_retreat','sensor_settle','sensor_lift','sensor_turn','sensor_turn_settle','stopped'):raise ValueError(self.scene_wrist_failure)
            return previous
        if self.scene_wrist_hold is None and phase not in ('verify_pickup','hold','sensor_retreat','sensor_settle','sensor_lift','sensor_turn','sensor_turn_settle'):return previous
        if phase=='sensor_lift' and not self.floor_hold_active:self.floor_hold_pending=True
        packet=self.sensor_recorder.latest_packet
        sensor_control=self.control.sensor_fault is not None
        carrying=sensor_control or self.floor_hold_pending or self.floor_hold_active
        motion=self.box_perception.latest_carry_frame if carrying else self.box_perception.latest_motion
        frame_key='body_in_control_frame' if carrying else 'body_in_segment'
        record=dict(step=packet['step'],time_s=now_s,phase=phase,frame_key=frame_key,
                    lift_used_m=self.control.lift_used_m)
        try:
            if stopping:
                retained=self.loaded_stop_clearance.retention
                if (retained is None or not retained['available']
                        or not 0<=now_s-retained['time_s']<=.150001):
                    raise ValueError('stop_visual_retention_unavailable')
                if self.scene_wrist_hold is None or not self.floor_hold_active or self.floor_hold_pending:
                    raise ValueError('loaded_stop_wrist_owner_unavailable')
            else:
                grasp=self.box_perception.feedback(now_s)
                if grasp['status']!='available':
                    raise ValueError('scene_wrist_visual_state_unavailable')
                if sensor_control and (not grasp.get('opposing_near_wrists') or not grasp.get('attitude_ok')
                        or grasp.get('gap_m',0.)<.02):
                    raise ValueError('hold_visual_retention_lost')
            fresh=motion is not None and abs(motion['time_s']-packet['time_s'])<1e-8
            if sensor_control:
                # Current encoders pair with the existing 50 Hz IMU rotation.
                # Height remains the explicitly aged camera measurement. The
                # fresh flag above still refers to a REAL camera for anchoring.
                motion=self.box_perception.control_frame(now_s)
                record.update(observed_at_s=motion['observed_at_s'],height_age_s=motion['height_age_s'])
            if self.scene_wrist_hold is None or self.floor_hold_pending:
                if not fresh:
                    record['status']='waiting_synchronized_camera'
                    result=list(previous);result[43:46]=[0.,0.,0.]
                    return result
                self.scene_wrist_hold=SceneWristHold(self.term.sensor_model,self.names,packet,motion,previous,
                                                    frame_key=frame_key)
                self.floor_hold_active=carrying;self.floor_hold_pending=False
                self.lift_anchor_m=self.control.lift_used_m
            elif (sensor_control or fresh) and packet['time_s']>self.scene_wrist_hold.updated_at:
                if carrying and not self.control.lift_failed:
                    self.scene_wrist_hold.raise_targets(self.control.lift_used_m-self.lift_anchor_m)
                self.scene_wrist_hold.update(packet,motion)
            result=self.scene_wrist_hold.command(now_s,previous)
            if stopping:
                screen=self.loaded_stop_clearance.screen(packet,self.sensor_recorder.latest_hands,
                    self.box_perception.motion.rotation,result)
                record['stop_screen']=screen
                if screen['status']!='clear':raise ValueError('coupled_stop_path_margin_insufficient')
                self.loaded_stop_feedback=dict(status='active',time_s=now_s,
                    clearance_lower_m=screen['lower_m'],required_margin_m=screen['required_margin_m'])
            record.update(status='active',solve=self.scene_wrist_hold.last_solve)
            return result
        except ValueError as error:
            self.scene_wrist_failure=str(error)
            record.update(status='failed',reason=str(error),
                solve=self.scene_wrist_hold.last_solve if self.scene_wrist_hold else None)
            raise
        finally:
            self.files['scene-hold'].write(json.dumps(record,allow_nan=False)+'\n')
            self.files['scene-hold'].flush()

    def sensor_turn_observation(self,now_s):
        """Current sensor control frame and visual grasp; no simulator pose."""
        return dict(frame=self.box_perception.control_frame(now_s),
            grasp=self.box_perception.feedback(now_s),up=self.box_perception.carry_frame.up.copy())

    def sensor_turn_controller(self,now_s,previous,angle_rad):
        from .sensor_turn import SensorTurn
        if (self.scene_wrist_hold is None or self.scene_wrist_failure is not None
                or not self.floor_hold_active or self.floor_hold_pending):
            raise ValueError('turn_wrist_owner_unavailable')
        controller=SensorTurn(now_s,angle_rad,self.sensor_turn_observation)
        sample=self.sensor_turn_observation(now_s)
        self.box_perception.begin_carry(now_s)
        self.scene_wrist_hold.follow_heading(sample['frame'],sample['up'])
        return controller

    def sensor_retreat_observation(self,now_s):
        """Only measured front/grasp/control-frame values, never self.raw."""
        front=self.approach.feedback(now_s)
        if front['status']=='available':
            observed_at,_,offset=self.approach.previous
            front=dict(status='available',time_s=observed_at,offset_body_m=offset,
                normal_segment=(self.box_perception.motion.rotation@self.approach.normal).tolist(),
                normal_navigation_xy=front['normal_navigation_xy'])
        return dict(front=front,frame=self.box_perception.latest_carry_frame,
                    grasp=self.box_perception.feedback(now_s))

    def sensor_retreat_controller(self,now_s,previous,distance_m):
        from .sensor_retreat import SensorRetreat
        if self.scene_wrist_failure is not None:raise ValueError(self.scene_wrist_failure)
        controller=SensorRetreat(now_s,distance_m,self.sensor_retreat_observation,
            lambda:self.floor_hold_active and not self.floor_hold_pending and self.scene_wrist_failure is None)
        self.box_perception.begin_carry(now_s)
        if not self.floor_hold_active:self.floor_hold_pending=True
        return controller

    def wrist_motion(self,observation,translation,*,inward_m=0.,max_translation_m=.06):
        q=observation['root_quat']
        return PairedWristController(self.term.robot_model,self.names,observation['joint_pos'],observation['root_pos'],[*q[1:],q[0]],observation['wrists_world'],translation,inward_m=inward_m,max_translation_m=max_translation_m)

    def placement_controller(self,obs,action,surface_id):
        def retract_wrists(row,translation):
            return self.wrist_motion(row,translation,max_translation_m=.08)
        return PreparedPlacement(obs,action,surface_id,
            lambda row,reference:HeldAlignment(row,reference,self.wrist_motion),
            lambda row,reference,surface:PlacementRetraction(row,reference,surface,
                retract_wrists,self.geometry,self.support_parts,arm_clearance=self.placement_clearance),self.wrist_motion)

    def step(self):
        self.phase=self.control.phase
        self.raw_policy=self.guard=self.input_hash=None
        self.hand_guard=None
        # A new pickup starts fresh measured-motion history. Idle/loaded actions
        # retain their physical targets; no controller state resets the world.
        if self.phase!='acquire':self.hand_clearance=None
        action=self.control.command(self.controller_observation())
        self.step_index+=1
        self.files['actions'].write(json.dumps(dict(step=self.step_index,phase=self.phase,action=[action],
            raw_policy_action=[self.raw_policy] if self.raw_policy is not None else None,navigation_guard=self.guard,
            hand_clearance_guard=self.hand_guard,
            sensor_lift=(self.control.clearance_lift.result() if self.control.phase=='sensor_lift' else None),
            sensor_turn=(self.control.loaded_motion.measurements(self.raw['time'])
                if self.sensor_guard is not None and self.control.method=='turn_with_box' else None),
            sensor_retreat=(self.control.loaded_motion.measurements(self.raw['time'])
                if self.sensor_guard is not None and self.control.method=='retreat_with_box' else None),
            visual_grasp_feedback=(self.box_perception.feedback(self.raw['time'])
                                   if self.box_perception is not None else None),
            policy_get_action_called=self.raw_policy is not None,input_camera_sha256=self.input_hash))+'\n');self.files['actions'].flush()
        self.contacts.clear()
        try:self.obs,*_=self.env.step(torch.tensor([action],device=self.native.device,dtype=torch.float32))
        except NativeTerminal:self.native_terminal=True
        if 'wbc-inputs' in self.files:
            self.files['wbc-inputs'].write(json.dumps(self.term.last_sensor_input)+'\n')
            self.files['wbc-inputs'].flush()
        self.raw=self.capture()
        self.files['physics-contacts'].flush()
        assert len(self.contacts)==4
        return self.raw

    def close(self):
        if self.sensor_recorder is not None:self.sensor_recorder.close()
        for writer in self.writers.values():writer.close()
        for file in self.files.values():file.close()
        self.policy.close()
        self.env.close()
