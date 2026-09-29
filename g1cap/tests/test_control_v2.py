"""Unit tests for the v2 skills and hard stops, using synthetic estimated rows."""
import math

import numpy as np
import unittest

from g1cap import control_v2 as v2


def row(t, x=0.0, y=0.0, yaw=0.0, held=True, tilt=0.0, box=True, clearance=0.08, surfaces=None, tables=None):
    q = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    b = dict(pos=[x + 0.5, y, 0.1], quat=[1, 0, 0, 0], size_m=[.2, .2, .2], age_s=0.0,
             bounds=dict(min=[x + .4, y - .1, 0.0], max=[x + .6, y + .1, .2]))
    return dict(time=t, step=int(round(t / .02)), root_pos=[x, y, 0.0], root_quat=q, tilt=tilt,
                box=b if box else None, box_held=held, box_between_hands=held, clearance=clearance,
                surfaces=surfaces or {}, tables=tables or {}, body_poses={}, walked_m=0.0,
                joint_pos=[0.0] * 43, wrists_world={})


ACTION = [0.0] * 46 + [0.75, 0.0, 0.0, 0.0]


class WalkTests(unittest.TestCase):
    def test_walk_reaches_goal_and_completes(self):
        walk = v2.Walk(row(0.0), ACTION, forward_m=0.3)
        x, t, result = 0.0, 0.0, None
        while result is None and t < 20:
            cmd = walk.command(row(t, x=x))
            x += cmd[43] * 0.02
            t += 0.02
            result = walk.update(row(t, x=x))
        self.assertEqual(result, ("completed", "walked_and_stopped"))
        self.assertAlmostEqual(x, 0.3, delta=0.03)

    def test_speed_is_capped(self):
        walk = v2.Walk(row(0.0), ACTION, forward_m=1.0)
        speeds = []
        for i in range(200):
            speeds.append(walk.command(row(i * .02))[43])
        self.assertGreater(max(speeds), 0.19)
        self.assertLessEqual(max(speeds), 0.22 + 1e-9)

    def test_rejects_out_of_envelope(self):
        with self.assertRaises(ValueError):
            v2.Walk(row(0.0), ACTION, forward_m=2.0)

    def test_settled_braking_drift_within_expected_stop_distance_completes(self):
        walk = v2.Walk(row(0.0), ACTION, forward_m=1.0)
        walk.phase, walk.settle_start = 'settle', 0.0
        self.assertEqual(walk.update(row(1.1, x=0.945)), ('completed', 'walked_and_stopped'))

    def test_settled_walk_with_larger_residual_resumes_bounded_correction(self):
        walk = v2.Walk(row(0.0), ACTION, forward_m=1.0)
        walk.phase, walk.settle_start = 'settle', 0.0
        self.assertIsNone(walk.update(row(1.1, x=0.92)))
        self.assertEqual(walk.phase, 'walk')
        self.assertGreater(walk.command(row(1.12, x=0.92))[43], 0.0)

    def test_requested_heading_changes_during_translation_and_finishes_at_goal(self):
        control=v2.ControlV2(ACTION,acquire=None,wrist_motion=None,robot_bounds={})
        self.assertEqual(control.start('walk',{'forward_m':1.0,'yaw_rad':0.4},row(0.0))['status'],'running')
        walk=control.skill
        x=y=yaw=t=0.0
        overlapped=False
        result=None
        while result is None and t<30.0:
            cmd=walk.command(row(t,x=x,y=y,yaw=yaw))
            overlapped |= abs(cmd[43])>0.05 and abs(cmd[45])>0.02
            x += (math.cos(yaw)*cmd[43]-math.sin(yaw)*cmd[44])*0.02
            y += (math.sin(yaw)*cmd[43]+math.cos(yaw)*cmd[44])*0.02
            yaw += cmd[45]*0.02
            t += 0.02
            result=walk.update(row(t,x=x,y=y,yaw=yaw))
        self.assertTrue(overlapped)
        self.assertEqual(result,('completed','walked_and_stopped'))
        self.assertAlmostEqual(x,1.0,delta=0.04)
        self.assertAlmostEqual(y,0.0,delta=0.04)
        self.assertAlmostEqual(yaw,0.4,delta=math.radians(6))


class ApproachSpeedTests(unittest.TestCase):
    def test_loaded_approach_can_continue_past_old_deadline(self):
        table = {'destination': {'min_xy': [1.5, -0.4], 'max_xy': [2.1, 0.4],
                                 'position_uncertainty_m': 0.0}}

        def observed(t, held):
            state = row(t, held=held, tables=table)
            state['body_poses'] = {
                name: {'pos': [0.0, 0.0, 0.0], 'xyzw': [0.0, 0.0, 0.0, 1.0]}
                for name in ('pelvis', 'left_knee_link', 'right_knee_link')}
            return state

        loaded = v2.Approach(observed(0.0, True), ACTION, 'destination')
        unloaded = v2.Approach(observed(0.0, False), ACTION, 'destination')
        old_deadline = 30.0 + math.dist((0.0, 0.0), loaded.goal) / 0.05
        self.assertIsNone(loaded.update(observed(old_deadline + 0.1, True)))
        self.assertEqual(unloaded.update(observed(old_deadline + 0.1, False)),
                         ('failed', 'approach_timeout'))

    def test_loaded_alignment_turn_uses_carried_yaw_limit(self):
        table={'destination':{'min_xy':[1.5,-0.4],'max_xy':[2.1,0.4],
                              'position_uncertainty_m':0.0}}
        approach=v2.Approach(row(0.0,yaw=math.pi/2,tables=table),ACTION,'destination')
        rates=[approach.command(row(i*.02,yaw=math.pi/2,tables=table))[45] for i in range(100)]
        self.assertLessEqual(max(abs(rate) for rate in rates),0.15+1e-9)

    def test_distant_approach_walk_keeps_conservative_loaded_command(self):
        table={'destination':{'min_xy':[1.5,-0.4],'max_xy':[2.1,0.4],
                              'position_uncertainty_m':0.0}}
        start=row(0.0,tables=table)
        approach=v2.Approach(start,ACTION,'destination')
        approach.phase='walk_to_goal'
        speeds=[approach.command(row(i*.02,tables=table))[43] for i in range(100)]
        self.assertGreater(max(speeds),0.11)
        self.assertLessEqual(max(speeds),0.12+1e-9)

    def test_distant_diagonal_approach_caps_vector_speed(self):
        table={'destination':{'min_xy':[1.5,0.5],'max_xy':[2.1,1.3],
                              'position_uncertainty_m':0.0}}
        approach=v2.Approach(row(0.0,tables=table),ACTION,'destination')
        approach.phase='walk_to_goal'
        speeds=[]
        for i in range(100):
            cmd=approach.command(row(i*.02,tables=table))
            speeds.append(math.hypot(cmd[43],cmd[44]))
        self.assertLessEqual(max(speeds),0.12+1e-9)

    def test_approach_ramp_transition_still_caps_vector_speed(self):
        table={'destination':{'min_xy':[1.5,-0.4],'max_xy':[2.1,0.4],
                              'position_uncertainty_m':0.0}}
        approach=v2.Approach(row(0.0,tables=table),ACTION,'destination')
        approach.phase='walk_to_goal'
        approach.v,approach.vy=0.20,0.02
        cmd=approach.command(row(0.0,tables=table))
        self.assertLessEqual(math.hypot(cmd[43],cmd[44]),0.20+1e-9)

    def test_last_part_of_distant_alignment_overlaps_translation(self):
        table={'destination':{'min_xy':[1.5,-0.4],'max_xy':[2.1,0.4],
                              'position_uncertainty_m':0.0}}
        near_heading=v2.Approach(row(0.0,yaw=0.3,tables=table),ACTION,'destination')
        cmd=None
        for i in range(30):
            cmd=near_heading.command(row(i*.02,yaw=0.3,tables=table))
        self.assertGreater(cmd[43],0.05)
        self.assertLess(cmd[45],-0.02)
        far_heading=v2.Approach(row(0.0,yaw=math.pi/2,tables=table),ACTION,'destination')
        cmd=far_heading.command(row(0.0,yaw=math.pi/2,tables=table))
        self.assertEqual(cmd[43:45],[0.0,0.0])


class TurnTests(unittest.TestCase):
    def test_turn_converges(self):
        turn = v2.Turn(row(0.0), ACTION, -math.pi / 2)
        yaw, t, result = 0.0, 0.0, None
        while result is None and t < 40:
            cmd = turn.command(row(t, yaw=yaw))
            yaw += cmd[45] * 0.02
            t += 0.02
            result = turn.update(row(t, yaw=yaw))
        self.assertEqual(result[0], "completed")
        self.assertAlmostEqual(yaw, -math.pi / 2, delta=math.radians(6))


class HardStopTests(unittest.TestCase):
    def setUp(self):
        self.control = v2.ControlV2(ACTION, acquire=None, wrist_motion=None, robot_bounds={})

    def test_tilt_is_terminal(self):
        self.control.start("wait", {"duration": 1.0}, row(0.0))
        self.control.update(row(0.02, tilt=math.radians(20)))
        self.assertEqual(self.control.terminal_reason, "body_tilt_limit")
        self.assertEqual(self.control.result["status"], "failed")

    def test_box_loss_stops_carrying_walk(self):
        self.assertEqual(self.control.start("walk", {"forward_m": 0.5}, row(0.0))["status"], "running")
        result = None
        for i in range(1, 40):
            self.control.command(row(i * .02, held=False))
            result = self.control.update(row(i * .02, held=False)) or result
        self.assertEqual(self.control.result["reason"], "box_retention_lost")
        self.assertEqual(self.control.last_action[43:46], [0.0, 0.0, 0.0])

    def test_table_corner_is_advice_not_a_stop(self):
        near = dict(source=dict(min_xy=[0.1, 0.1], max_xy=[0.8, 0.8]))     # corner 0.14 m from the pelvis
        def r(t):
            x = row(t, held=False, box=False, tables=near)
            x["body_poses"] = {"pelvis": {"pos": [0.0, 0.0, 0.8]}}
            return x
        self.control.start("walk", {"forward_m": -0.3}, r(0.0))
        for i in range(1, 30):
            self.control.command(r(i * .02))
            self.control.update(r(i * .02))
        self.assertIsNone(self.control.result)                                # still walking
        self.assertEqual(self.control.skill.nearest_corner, 0.141)
        self.assertEqual(self.control.events[-1]["advisory"], "near_table_corner")

    def test_rejects_unknown_tool_and_busy(self):
        self.assertEqual(self.control.start("fly", {}, row(0.0))["status"], "rejected")
        self.control.start("wait", {"duration": 1.0}, row(0.0))
        self.assertEqual(self.control.start("wait", {"duration": 1.0}, row(0.0))["reason"], "operation_active")


class HoldCheckTests(unittest.TestCase):
    def test_stable_held_box_ready_after_one_second(self):
        rows = [row(i * .02) for i in range(0, 60)]
        for r in rows:
            r["box"]["age_s"] = 0.0 if r["step"] % 5 == 0 else 0.02
        self.assertTrue(v2.camera_rate_hold(rows)["ready"])

    def test_moving_box_not_ready(self):
        rows = [row(i * .02) for i in range(0, 60)]
        for i, r in enumerate(rows):
            r["box"]["pos"][2] += 0.001 * i
        self.assertFalse(v2.camera_rate_hold(rows)["ready"])


class VersionBTests(unittest.TestCase):
    def test_carried_turn_rate_is_capped(self):
        turn = v2.Turn(row(0.0, held=True), ACTION, math.pi / 2)
        rates = [turn.command(row(i * .02, held=True))[45] for i in range(300)]
        self.assertLessEqual(max(abs(r) for r in rates), 0.15 + 1e-9)

    def test_contact_stop_allows_backing_away(self):
        control = v2.ControlV2(ACTION, acquire=None, wrist_motion=None, robot_bounds={})
        slab = dict(bounds=dict(min=[0.3, -1, -0.04], max=[1.0, 1, 0.0]))
        def r(t, x):
            base = row(t, x=x, held=True)
            base["box"]["bounds"] = dict(min=[x + .2, -.1, 0.01], max=[x + .4, .1, .21])
            base["surfaces"] = {"destination": slab}
            return base
        control.start("walk", {"forward_m": -0.3}, r(0.0, 0.0))
        x = 0.0
        for i in range(1, 60):
            cmd = control.command(r(i * .02, x))
            x += cmd[43] * .02
            control.update(r(i * .02, x))
        self.assertNotEqual((control.result or {}).get("reason"), "predicted_table_contact")


if __name__ == "__main__":
    unittest.main()


class GraspApproachTests(unittest.TestCase):
    """Scripted grasp approach, from grasp-q-*-10: width after squaring and the shift contact guard."""

    NAMES = [f"j{i}" for i in range(29)] + v2.GRASP_ARM_JOINTS

    def grasp_row(self, box_yaw, robot_yaw=0.0, t=0.0, size=(0.2, 0.24, 0.16)):
        r = row(t, yaw=robot_yaw, held=False)
        r["box"].update(quat=[math.cos(box_yaw / 2), 0.0, 0.0, math.sin(box_yaw / 2)], size_m=list(size))
        r["box"]["pos"] = [0.51, 0.03, 0.12]
        return r

    def make(self, r):
        return v2.Grasp(r, ACTION, wrist_motion=None, names=self.NAMES)

    def test_width_is_measured_after_squaring_up(self):
        g = self.make(self.grasp_row(0.35))
        self.assertEqual(g.phase, "square_up")
        self.assertAlmostEqual(g.width, 0.24 * math.cos(0.35) + 0.2 * math.sin(0.35), places=3)   # 0.294, stale
        g.set_width(self.grasp_row(0.35, robot_yaw=0.35))
        self.assertAlmostEqual(g.width, 0.24, places=3)

    def test_opening_clears_inboard_fingers(self):
        for size, width in (((0.2, 0.2, 0.2), 0.2), ((0.2, 0.24, 0.16), 0.24)):
            g = self.make(self.grasp_row(0.0, size=size))
            span = v2.GRASP_REF_SPAN_M + g.open_rad * v2.GRASP_SPAN_PER_RAD
            self.assertGreaterEqual(span + 1e-6, width + 2 * v2.GRASP_FINGER_INBOARD_M)
            self.assertLessEqual(g.open_rad, v2.GRASP_MAX_OPEN_RAD)
            self.assertGreater(g.expected_travel, 0.0)

    def test_arm_lag_in_shift_backs_off_then_fails(self):
        r = self.grasp_row(0.0)
        g = self.make(r)
        g.phase, g.phase_start = "shift", 0.0
        g.shift = [0.0, 0.0, -0.06]
        g.reach_refs = [g.ref[i] for i in g.arm_index]
        free = dict(r, time=0.1, joint_pos=list(g.ref[:43]))
        self.assertIsNone(g.update(free))
        self.assertEqual(g.phase, "shift")
        pressed = list(g.ref[:43])
        pressed[g.arm_index[7]] -= 0.05          # right elbow held back by the box
        g.update(dict(r, time=0.2, joint_pos=pressed))
        self.assertEqual(g.phase, "back_off")
        self.assertIsNone(g.update(dict(r, time=1.0, joint_pos=pressed)))
        self.assertEqual(g.update(dict(r, time=1.8, joint_pos=pressed)), ("failed", "hands_hit_box"))

    def test_lift_counter_rotates_the_wrists(self):
        g = self.make(self.grasp_row(0.0))
        g.start_lift(self.grasp_row(0.0), 0.0)
        action = g.command(dict(self.grasp_row(0.0), time=10.0))
        by_name = dict(zip(v2.GRASP_ARM_JOINTS, [action[i] - g.lift_from[k] for k, i in enumerate(g.arm_index)]))
        for side in ("left", "right"):
            self.assertAlmostEqual(by_name[f"{side}_shoulder_pitch_joint"], -g.dpitch)
            self.assertAlmostEqual(by_name[f"{side}_wrist_pitch_joint"], g.dpitch)

    def test_lineup_stands_closer_and_lower_reaches_preclose_height(self):
        for size, box_z in (((0.2, 0.2, 0.2), 0.10), ((0.2, 0.24, 0.16), 0.08)):
            r = self.grasp_row(0.0, size=size)
            r["box"]["pos"][2] = box_z
            g = self.make(r)
            reach_at = g.lineup_target(r)
            g.stepped_in = True
            g.step_wrist = v2.GRASP_REACH_WRIST_IN_PELVIS     # arms held where reach left them
            g.step_box_z = box_z
            target = g.lineup_target(r)
            self.assertTrue(0.36 < target[0] < 0.44, target)      # was 0.51
            self.assertAlmostEqual(reach_at[0] - target[0], v2.GRASP_REACH_STANDOFF_M)
            dp = g.lower_pitch(v2.GRASP_REACH_WRIST_IN_PELVIS[2], box_z)
            wrist_z = v2.GRASP_REACH_WRIST_IN_PELVIS[2] - dp * v2.GRASP_LIFT_PER_RAD
            self.assertAlmostEqual(box_z - wrist_z, g.preclose_offset()[2], places=6)

    def test_lower_uses_the_lift_motion_in_reverse(self):
        g = self.make(self.grasp_row(0.0))
        g.start_lower(0.0, 0.2)
        action = g.command(dict(self.grasp_row(0.0), time=10.0))
        by_name = dict(zip(v2.GRASP_ARM_JOINTS, [action[i] - g.lift_from[k] for k, i in enumerate(g.arm_index)]))
        self.assertAlmostEqual(by_name["left_shoulder_pitch_joint"], 0.2)
        self.assertAlmostEqual(by_name["left_wrist_pitch_joint"], -0.2)
        self.assertLess(by_name["left_shoulder_roll_joint"], 0.0)       # adducts to hold the span

    def test_reach_swings_the_thumbs_out(self):
        names = self.NAMES[:29] + ["left_hand_thumb_1_joint", "right_hand_thumb_1_joint"] + v2.GRASP_ARM_JOINTS
        r = self.grasp_row(0.0)
        g = v2.Grasp(r, [0.0] * 46 + [0.75, 0.0, 0.0, 0.0], wrist_motion=None, names=names)
        g.phase, g.phase_start = "reach", 0.0
        action = g.command(dict(r, time=5.0))
        self.assertAlmostEqual(action[names.index("left_hand_thumb_1_joint")], -0.72)
        self.assertAlmostEqual(action[names.index("right_hand_thumb_1_joint")], 0.72)


class GripKeepTests(unittest.TestCase):
    def make(self):
        c = v2.ControlV2.__new__(v2.ControlV2)
        c.grip = dict(index=[0, 1], roll=[0.30, -0.30], extra=0.0, floor=0.04)
        return c

    def test_holds_targets_while_load_is_kept(self):
        c = self.make()
        action = [0.0] * 50
        c.keep_grip(action, dict(joint_pos=[0.35, -0.35] + [0.0] * 41))   # lag 0.05 >= floor
        self.assertEqual(action[:2], [0.30, -0.30])
        self.assertEqual(c.grip["extra"], 0.0)

    def test_closes_further_when_load_decays_up_to_a_cap(self):
        c = self.make()
        for _ in range(1000):
            action = [0.0] * 50
            roll = c.grip["roll"]                     # box slipping: measured stays 0.01 behind
            c.keep_grip(action, dict(joint_pos=[roll[0] + 0.01, roll[1] - 0.01] + [0.0] * 41))
        self.assertAlmostEqual(c.grip["extra"], v2.GRIP_KEEP_MAX_RAD, places=3)
        self.assertLess(action[0], 0.30)             # left closes by decreasing roll
        self.assertGreater(action[1], -0.30)


class GuardDirectionTests(unittest.TestCase):
    def test_moving_away_never_trips_and_moving_closer_trips_after_5mm(self):
        away, closer = v2.Wait.__new__(v2.Wait), v2.Wait.__new__(v2.Wait)
        self.assertFalse(any(v2.worsened(away, "c", 0.010 + 0.001 * k + 0.002 * (-1) ** k) for k in range(20)))
        values = [0.019 - 0.001 * k for k in range(10)]
        tripped = [v for v in values if v2.worsened(closer, "c", v)]
        self.assertAlmostEqual(tripped[0], 0.013)

    def test_map_is_frozen_per_skill_so_a_growing_map_does_not_look_like_approach(self):
        skill = v2.Wait.__new__(v2.Wait)
        first = v2.frozen_map(skill, dict(tables={"source": {"min_xy": [0.4, -0.4]}, "destination": None}), "tables")
        later = v2.frozen_map(skill, dict(tables={"source": {"min_xy": [0.3, -0.5]},
                                                  "destination": {"min_xy": [1.0, 1.0]}}), "tables")
        self.assertEqual(first, {"source": {"min_xy": [0.4, -0.4]}})
        self.assertEqual(later["source"], {"min_xy": [0.4, -0.4]})           # kept as first seen
        self.assertEqual(later["destination"], {"min_xy": [1.0, 1.0]})       # newly seen tables join


class PlaceViewMarginTests(unittest.TestCase):
    def test_near_margin_sees_a_box_hanging_over_the_near_edge(self):
        r = row(0.0)                                         # box x 0.4..0.6 ahead, y -0.1..0.1
        table = np.array([[x, y, 0.0] for x in np.arange(0.30, 1.2, 0.02) for y in np.arange(-0.5, 0.5, 0.02)])
        m = v2.Place.view_margins(r, table)
        self.assertAlmostEqual(m["near"], 0.1, delta=0.02)   # table starts 10 cm before the box
        over = table[table[:, 0] >= 0.52]                     # table starts under the box
        self.assertIsNone(v2.Place.view_margins(r, over)["near"])

    def test_open_releases_the_arm_squeeze(self):
        p = v2.Place.__new__(v2.Place)
        p.ref = [0.0] * 50
        p.ref[3] = -0.30                                  # commanded roll, squeezing
        p.arm_index, p.phase, p.phase_start = [3], "open", 0.0
        p.closed, p.open_fingers = [0.0] * 14, [0.0] * 14
        p.arm_closed, p.arm_relaxed = {3: -0.30}, {3: -0.22}   # measured: held out by the box
        action = p.command(dict(row(2.0), joint_pos=[0.0] * 43))
        self.assertAlmostEqual(action[3], -0.22)


class RaiseTests(unittest.TestCase):
    def test_raise_lifts_by_shoulder_pitch_not_wrist_ik(self):
        r = row(0.0, clearance=0.06)
        arm = list(range(15, 29))
        raise_ = v2.Raise(r, ACTION, 0.12, arm)
        self.assertEqual(raise_.phase, "lift")
        action = raise_.command(dict(r, time=10.0))
        by = dict(zip(v2.GRASP_ARM_JOINTS, [action[i] - raise_.lift_from[k] for k, i in enumerate(arm)]))
        dp = 0.06 / v2.GRASP_LIFT_PER_RAD
        self.assertAlmostEqual(by["left_shoulder_pitch_joint"], -dp)
        self.assertAlmostEqual(by["right_wrist_pitch_joint"], dp)
        self.assertAlmostEqual(by["left_shoulder_roll_joint"], v2.GRASP_SPAN_PER_PITCH * dp)
