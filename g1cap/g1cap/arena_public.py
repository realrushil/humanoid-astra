"""Worker-visible sensor estimates; no physics/scorer state is accepted here."""
from copy import deepcopy
import math

MODE='sensor_estimates_v1'
METHODS=frozenset({'observe','wait','pickup_box','hold_box','retreat_with_box','raise_held_box','turn_with_box'})


def selected(value,keys):
    return {key:deepcopy(value[key]) for key in keys if key in value}


def public_task(task,*,sensor_control=False):
    # Semantic targets and time budgets are instructions, not sensed geometry.
    value=selected(task,('backend','object_id','instruction','task_kind','surface_id',
        'hold_duration_s','deadline_s','retreat_distance_m','minimum_travel_m',
        'wall_timeout_s','worker_timeout_s'))
    value.update(observation_mode=MODE,control_track=('sensor_coupled_stop_v6' if sensor_control else 'mixed_controller_sensor_publication'),
        available_tools=sorted(METHODS))
    return value


def sensor_observation(packet,box,grasp,motion,*,loaded_stop=None):
    """Current body sample plus independently timestamped optical box estimate.

    Scene origin is local to a continuous stance segment, not global world pose.
    Motor torque is estimated actuator output, never a tactile/contact force.
    """
    body=selected(packet,('version','step','time_s','joint_names','q_rad','dq_rad_s',
        'tau_est_nm','gyro_rad_s','specific_force_m_s2'))
    measured_box=selected(box,('status','reason','observed_at_s','age_s','track_epoch'))
    if box['status']=='accepted':
        measured_box.update(selected(box,('center_camera_m','axes_camera','dimensions_m')))
        measured_box['frame']='camera_optical_at_observation_time'
    scene=dict(status='unavailable',body_in_segment=None)
    if motion is not None:
        scene.update(selected(motion,('time_s','segment')))
        age=packet['time_s']-motion['time_s']
        if motion['status']!='unavailable' and 0<=age<=.150001:
            scene.update(selected(motion,('status','body_in_segment')))
            scene.update(age_s=age,frame='local_stance_segment',assumption='near_floor_sole_points_do_not_slip')
    # The existing Arena recorder numbers four 200 Hz physics substeps per
    # 50 Hz action sample. This is an episode counter, not a pose measurement.
    result=dict(observation_mode=MODE,time=packet['time_s'],step=packet['step'],physics_step=packet['step']*4,
        proprioception=body,box=measured_box,scene_motion=scene,
        grasp=selected(grasp,('status','reason','time_s','age_s','raised','ready','gap_m',
            'opposing_near_wrists','attitude_ok','wrist_relative_ready','scene_status','gap_reference','settled_retention')),
        object_load=dict(status='unknown',mass_kg=None))
    if loaded_stop is not None:
        observed=loaded_stop.get('time_s')
        current=(type(observed) in (int,float) and math.isfinite(observed)
            and 0<=packet['time_s']-observed<=.020001)
        if not current:
            result['loaded_stop']=dict(status='unavailable',reason='loaded_stop_status_stale')
        else:
            result['loaded_stop']=selected(loaded_stop,('status','time_s','clearance_lower_m','required_margin_m'))
            if 'reason' in loaded_stop:result['loaded_stop']['reason']=public_reason(loaded_stop['reason'])
    return result


def public_reason(reason):
    # Unknown controller exceptions/privileged guard details must not become
    # oracle recovery hints. Exact causes remain in physics/evaluation artifacts.
    allowed={'accepted','stale_round','invalid_request','duplicate_argument','operation_active',
        'request_wall_timeout','worker_request_cancelled','duration_outside_envelope','unsupported_object',
        'visual_state_unavailable','visual_grasp_mode_supports_pickup_hold_only','sensor_tool_unavailable',
        'visually_raised_and_stable','hold_verified','dwell_elapsed','acquisition_timeout',
        'pose_hold_timeout','hold_did_not_settle','sensor_approach_unavailable','sensor_hand_clearance_unavailable','scene_wrist_measurement_unavailable',
        'scene_wrist_visual_state_unavailable','scene_wrist_target_outside_envelope',
        'scene_wrist_hold_invalidated','scene_wrist_update_time_gap','scene_wrist_command_time_gap',
        'invalid_sensor_control_packet','invalid_sensor_measurement','sensor_sample_gap',
        'sensor_guard_uninitialized','sensor_guard_stale','sensor_body_tilt_limit',
        'sensor_effort_saturation','sensor_front_clearance_limit','sensor_hand_clearance_limit',
        'distance_outside_envelope','settled_bilateral_grasp_required','carry_source_observation_unavailable',
        'carry_initial_grasp_not_ready','carry_visual_grasp_lost','carry_motion_unavailable',
        'carry_measurement_time_mismatch','carry_rotation_drift','carry_front_identity_changed',
        'carry_wrist_handoff_timeout','carry_retreat_timeout','carry_wrong_direction','carry_overshoot',
        'carry_refinement_limit','carry_settle_timeout','stale_retreat_measurement',
        'sensor_retreat_and_hold_completed','observed_clearance_and_hold','height_outside_envelope',
        'clearance_lift_invalidated','clearance_lift_budget_exhausted','clearance_lift_timeout',
        'lift_camera_stale','lift_camera_discontinuity','lift_visual_retention_lost',
        'lift_grasp_unsettled','lift_wrist_handoff_timeout','invalid_parameter',
        'retained_grasp_requires_raise_or_hold','hold_visual_retention_lost',
        'floor_control_frame_unavailable','floor_control_imu_time_mismatch',
        'angle_outside_envelope','turn_initial_grasp_not_ready','turn_control_frame_unavailable',
        'turn_wrist_owner_unavailable','turn_ready_camera_time_invalid','turn_measurement_time_gap',
        'measured_turn_timeout','turn_refinement_budget_exhausted','measured_heading_and_hold',
        'turn_completion_not_retained','loaded_stop_wrist_owner_unavailable',
        'stop_visual_retention_unavailable','stop_top_unavailable','stop_camera_time_unavailable',
        'coupled_stop_path_margin_insufficient','vertical_camera_stale','need_height_history',
        'height_history_gap','acceleration_endpoints_missing','acceleration_time_gap',
        'vertical_acceleration_envelope_exceeded'}
    return reason if reason in allowed else 'controller_stopped'
