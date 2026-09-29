"""Sensor-state v2 box-transfer tools: a few skills, few hard stops, advice in results.

Every decision here reads only the estimated row built from ``SensorState``
(encoders, IMU, Dex3 finger positions, head RGB-D). Simulator truth is never
passed in; it is recorded separately for scoring.

Actions: 50 native values = 43 joint references [rad], body vx, vy [m/s], yaw
rate [rad/s], pelvis height [m], torso roll/pitch/yaw [rad] (HOMIE).

Hard stops (terminal or operation-ending) are deliberately few:
  1. body tilt > 15 deg (IMU)                       -> episode fault
  2. carried box no longer between the hands        -> operation fails, navigation zeroed
  3. predicted body/box overlap with an observed table top while walking or turning
Table-corner proximity during walk/turn is advice only (nearest_table_corner_m).
Everything else (margins, estimate ages, uncertainties) is returned as advice.
"""
from collections import deque
import math
import struct

import numpy as np

from .arena_surfaces import transformed_bounds
from .toolkit.arena_approach import GUARDED_BODIES

DT = 0.02
TOOLS = {"observe": [], "wait": ["duration"], "pickup_box": ["object_id"], "grasp_box": ["lift_m"], "raise_held_box": ["clearance_m"],
         "hold_box": ["duration"], "walk": ["forward_m", "left_m", "yaw_rad"], "turn": ["yaw_rad"],
         "approach_surface": ["surface_id", "standoff_m"], "place_box": ["surface_id"]}


def f32(action):
    return list(struct.unpack("50f", struct.pack("50f", *action)))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def separation(a, b):
    return math.sqrt(sum(max(0., a["min"][i] - b["max"][i], b["min"][i] - a["max"][i]) ** 2 for i in range(3)))


# ---------------------------------------------------------------- estimated row
def estimated_row(state, robot_bounds, names, packet, hands, history):
    """Controller observation in odom from SensorState; no truth fields.

    ``history`` is a deque of previous rows (for closing speeds and box-held
    stability). Unmeasurable quantities (contact forces) are simply absent.
    """
    from .hand_sensors import measured_joint_positions
    from .sensor_state import matrix_to_wxyz
    T = state.odom.pose()
    now = packet["time_s"]
    poses = {}
    for link in robot_bounds:
        if state.kin.model.existFrame(link):
            W = T @ state.kin.frame(link)
            q = matrix_to_wxyz(W[:3, :3])
            poses[link] = dict(pos=W[:3, 3].tolist(), xyzw=[q[1], q[2], q[3], q[0]])
    row = dict(time=now, step=packet["step"], state_age_s=0.0, frame="odom",
               root_pos=T[:3, 3].tolist(), root_quat=matrix_to_wxyz(T[:3, :3]),
               tilt=state.odom.tilt(), floor_z=state.odom.floor_z,
               joint_pos=measured_joint_positions(packet, hands, names),
               body_poses=poses, wrists_world={k: poses[k] for k in ("left_wrist_yaw_link", "right_wrist_yaw_link")},
               walked_m=state.odom.walked_m)
    # Box: last camera pose, carried with the body between frames when held.
    box = state.box
    row["box"] = None
    held_prev = bool(history and history[-1].get("box_held"))
    if box is not None:
        if held_prev and "pelvis_T_box" in box and box["time_s"] < now:
            B = T @ box["pelvis_T_box"]
            pos, R = B[:3, 3], B[:3, :3]
        else:
            pos, R = box["pos"], box["R"]
        half = 0.5 * np.asarray(box["size"])
        corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * half @ R.T + pos
        bounds = dict(min=corners.min(axis=0).tolist(), max=corners.max(axis=0).tolist())
        row["box"] = dict(pos=list(map(float, pos)), quat=matrix_to_wxyz(R), size_m=list(box["size"]),
                          bounds=bounds, observed_at_s=box["time_s"], age_s=now - box["time_s"])
        row["box_pos"], row["box_quat"] = row["box"]["pos"], row["box"]["quat"]
    # Visual grasp hypothesis: box centre between the two wrists (no force sensing).
    row["box_between_hands"] = False
    if row["box"] is not None and row["box"]["age_s"] <= 0.25:
        wl = np.array(row["wrists_world"]["left_wrist_yaw_link"]["pos"])
        wr = np.array(row["wrists_world"]["right_wrist_yaw_link"]["pos"])
        c = np.array(row["box"]["pos"])
        axis = wl - wr
        span = float(np.linalg.norm(axis))
        if span > 0.05:
            # G1/Dex3 hold the box with the fingers, ~0.22-0.26 m ahead of the
            # wrist-yaw line (recorded pickups). Require: centre laterally between
            # the wrists, within 0.35 m ahead of them, and wrist spacing matching
            # the box width along that line within 12 cm (it squeezes the sides).
            t = float((c - wr) @ axis / span ** 2)
            off = float(np.linalg.norm(c - (wr + t * axis)))
            R = np.asarray(box["R"]) if box is not None else np.eye(3)
            width = float(np.sum(np.abs(R.T @ (axis / span)) * np.asarray(row["box"]["size_m"])))
            # Also require the box to be at hand height: standing in front of a box
            # on a table otherwise satisfied this test (grasp qualification trial).
            height_gap = float(abs(c[2] - 0.5 * (wl[2] + wr[2])))
            row["box_between_hands"] = bool(0.25 < t < 0.75 and off < 0.32 and abs(span - width) < 0.12
                                            and height_gap < 0.12)
    # Tables and surfaces.
    tables = state.table_summaries()
    row["tables"] = tables
    surfaces = {}
    lower = [transformed_bounds(s, poses[n]) for n in GUARDED_BODIES if n in poses for s in robot_bounds.get(n, [])]
    for name, t in tables.items():
        if t is None:
            continue
        slab = t["bounds"]
        entry = dict(bounds=slab, position_uncertainty_m=t["position_uncertainty_m"],
                     lower_body_clearance_m=min((separation(b, slab) for b in lower), default=None))
        if row["box"] is not None:
            bb = row["box"]["bounds"]
            entry["clearance_m"] = bb["min"][2] - slab["max"][2]
            m = 0.02
            entry["contained"] = all(slab["min"][i] + m <= bb["min"][i] and bb["max"][i] <= slab["max"][i] - m for i in range(2))
        surfaces[name] = entry
    row["surfaces"] = surfaces
    # Box height above the source (for pickup), from the same camera frame.
    src = surfaces.get("source")
    row["clearance"] = src.get("clearance_m") if src else None
    # Front clearance to the source. Prefer the current camera view in the pelvis
    # frame (independent of odometry drift); fall back to the odom map.
    view = state.views.get("source")
    if view is not None and now - view["time_s"] <= 0.25 and len(view["points_pelvis"]) >= 300:
        pts = view["points_pelvis"]
        front = float(np.percentile(pts[:, 0], 2))          # robust nearest top-edge x
        lower_pelvis = []
        for n in GUARDED_BODIES:
            if n in poses:
                F = state.kin.frame(n)
                for shp in robot_bounds.get(n, []):
                    corners = np.array([[a, b, c] for a in (shp["min"][0], shp["max"][0])
                                        for b in (shp["min"][1], shp["max"][1]) for c in (shp["min"][2], shp["max"][2])])
                    lower_pelvis.append(float((corners @ F[:3, :3].T + F[:3, 3])[:, 0].max()))
        row["approach_clearance_m"] = front - max(lower_pelvis)
        row["approach_source"] = "current_view"
    elif tables.get("source") is not None:
        front = tables["source"]["min_xy"][0]
        xs = [b["max"][0] for b in lower]
        row["approach_clearance_m"] = front - max(xs)
        row["approach_source"] = "odom_map"
    if row.get("approach_clearance_m") is not None:
        old = next((r for r in list(history)[-5:] if r.get("approach_clearance_m") is not None), None)
        row["approach_closing_speed_m_s"] = ((old["approach_clearance_m"] - row["approach_clearance_m"]) /
                                             max(now - old["time"], DT)) if old else 0.0
    # Carried box: held = between hands and not resting on a table.
    resting = any(abs(s.get("clearance_m", 1.0)) <= 0.015 and s.get("contained") for s in surfaces.values())
    row["box_held"] = bool(row["box_between_hands"] and not resting)
    return row


def camera_rate_hold(rows, seconds=1.0, require_held=True):
    """Stable held box over the last `seconds`, judged at camera frames.

    Box displacement relative to the pelvis < 1 cm and planar body speed < 5 cm/s.
    Uses only rows whose box pose is a fresh camera observation.
    """
    fresh = [r for r in rows if r.get("box") and r["box"]["age_s"] < 1e-6]
    if len(fresh) < 3 or fresh[-1]["time"] - fresh[0]["time"] < seconds - 0.11:
        return dict(ready=False, reason="need_one_second")
    def rel(r):
        q = r["root_quat"]; w, x, y, z = q
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        return R.T @ (np.array(r["box"]["pos"]) - np.array(r["root_pos"]))
    ref = rel(fresh[0])
    drift = max(float(np.linalg.norm(rel(r) - ref)) for r in fresh)
    # Per-frame (10 Hz) speeds, not window averages: gait sway must have settled.
    pairs = list(zip(fresh, fresh[1:]))
    body = max(math.dist(a["root_pos"][:2], b["root_pos"][:2]) / (b["time"] - a["time"]) for a, b in pairs)
    box = max(math.dist(a["box"]["pos"], b["box"]["pos"]) / (b["time"] - a["time"]) for a, b in pairs)
    key = "box_held" if require_held else "box_between_hands"
    ok = drift < 0.01 and body < 0.04 and box < 0.04 and all(r[key] for r in fresh)
    return dict(ready=ok, reason="stable_held_box" if ok else "moving_or_not_held",
                box_drift_in_pelvis_m=drift, body_speed_m_s=body, box_speed_m_s=box)


ARM_WORDS = ("hand", "wrist", "elbow", "shoulder")


def carried_clearance(row, robot_bounds, target=None):
    """Min distance from the robot and carried box to observed table top slabs (odom map).

    For the `target` table of an approach, the arms and carried box are exempt:
    they are meant to go over it. Near tables the map can be off by ~0.1 m, so
    close-range lower-body protection uses `under_table_clearance` instead.
    """
    best = None
    for name, s in row["surfaces"].items():
        slab = s["bounds"]
        for link, pose in row["body_poses"].items():
            if name == target:
                continue   # approach: guarded by under_table_clearance and the live view instead
            # Empty hands near a table are normal (reaching for the box); only a
            # carried load makes arm/table proximity meaningful.
            if any(w in link for w in ARM_WORDS) and not row.get("box_held"):
                continue
            for shp in robot_bounds.get(link, []):
                d = separation(transformed_bounds(shp, pose), slab)
                best = d if best is None else min(best, d)
        if row.get("box_held") and row["box"] is not None and name != target:
            d = separation(row["box"]["bounds"], slab)
            best = d if best is None else min(best, d)
    return best


def under_table_clearance_view(points_pelvis, row, robot_bounds):
    """Planar distance from lower-body links to the currently visible tabletop, pelvis frame.

    Uses this camera frame's top-surface points directly (no odometry). Negative
    means some top points lie inside a lower-body link's planar footprint.
    """
    T = np.eye(4)
    w, x, y, z = row["root_quat"]
    T[:3, :3] = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                          [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    T[:3, 3] = row["root_pos"]
    Tinv = np.linalg.inv(T)
    xy = np.asarray(points_pelvis)[:, :2]
    best = None
    for link in GUARDED_BODIES:
        pose = row["body_poses"].get(link)
        for shp in robot_bounds.get(link, []) if pose else []:
            b = transformed_bounds(shp, pose)
            corners = np.array([[b[i][0], b[j][1], b[k][2], 1.0] for i in ("min", "max") for j in ("min", "max") for k in ("min", "max")])
            local = (corners @ Tinv.T)[:, :2]
            lo, hi = local.min(axis=0), local.max(axis=0)
            d = np.maximum(np.maximum(lo - xy, xy - hi), 0.0)
            dist = np.sqrt((d * d).sum(axis=1))
            m = -0.01 if np.any(dist == 0.0) else float(dist.min())
            best = m if best is None else min(best, m)
    return best


def under_table_clearance(row, robot_bounds, name):
    """Planar distance from lower-body links to the observed footprint of table `name` (odom).

    Table legs are not mapped, so the lower body must stay out from under the
    top. Negative/zero means a lower-body link is over the footprint.
    """
    t = row["tables"].get(name)
    if t is None:
        return None
    (x0, y0), (x1, y1) = t["min_xy"], t["max_xy"]
    best = None
    for link in GUARDED_BODIES:
        pose = row["body_poses"].get(link)
        for shp in robot_bounds.get(link, []) if pose else []:
            b = transformed_bounds(shp, pose)
            dx = max(x0 - b["max"][0], b["min"][0] - x1, 0.0)
            dy = max(y0 - b["max"][1], b["min"][1] - y1, 0.0)
            inside = dx == 0.0 and dy == 0.0
            d = -min(b["max"][0] - x0, x1 - b["min"][0], b["max"][1] - y0, y1 - b["min"][1]) if inside else math.hypot(dx, dy)
            best = d if best is None else min(best, d)
    return best


# ---------------------------------------------------------------- skills
class Skill:
    """Base: holds the retained reference action; subclasses override step()."""
    carrying = False

    def __init__(self, row, action):
        self.ref = list(action)
        self.ref[43:46] = [0., 0., 0.]
        self.start = row["time"]
        self.rows = deque(maxlen=60)
        self.phase = type(self).__name__.lower()
        self.result = None
        self.advice = {}

    def finish(self, status, reason):
        self.result = (status, reason)
        return self.result

    def command(self, row):
        return list(self.ref)


class Wait(Skill):
    def __init__(self, row, action, duration):
        super().__init__(row, action)
        if not 0 < duration <= 5:
            raise ValueError("duration_outside_envelope")
        self.duration = duration

    def update(self, row):
        if row["time"] - self.start >= self.duration - 1e-9:
            return self.finish("completed", "dwell_elapsed")


class Hold(Skill):
    def __init__(self, row, action, duration):
        super().__init__(row, action)
        if not 0 < duration <= 3:
            raise ValueError("duration_outside_envelope")
        self.duration = max(1.0, duration)
        self.carrying = True

    def update(self, row):
        self.rows.append(row)
        elapsed = row["time"] - self.start
        hold = camera_rate_hold(self.rows)
        self.advice = hold
        if elapsed >= self.duration and hold["ready"]:
            return self.finish("completed", "hold_verified")
        if elapsed >= self.duration + 5:
            return self.finish("failed", "hold_did_not_settle")


class Pickup(Skill):
    """GR00T acquisition, then (if needed) a bounded paired-wrist lift, then a 1 s stable-hold check."""

    def __init__(self, row, action, acquire, wrist_motion):
        super().__init__(row, action)
        if row["tables"].get("source") is None:
            raise ValueError("source_table_not_observed")
        self.acquire, self.wrist_motion = acquire, wrist_motion
        self.phase, self.count, self.motion = "acquire", 0, None
        self.rate = 0.0
        # Square up to a rotated box first (the learned grasp fails at ~25 deg,
        # sweep-a scene 03): box yaw from the tracked pose, modulo 90 deg.
        delta = self.box_yaw_offset(row)
        if delta is not None and abs(delta) > math.radians(8):
            self.align_target = wrap(yaw_of(row["root_quat"]) + delta)
            self.phase = "align"
        self.advice = dict(box_yaw_offset_rad=delta)

    @staticmethod
    def box_yaw_offset(row):
        if row.get("box") is None:
            return None
        w, x, y, z = row["box"]["quat"]
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        horizontal = [R[:, i] for i in range(3) if abs(R[2, i]) < 0.5]
        if not horizontal:
            return None
        a = horizontal[0]
        yaw = math.atan2(a[1], a[0]) - yaw_of(row["root_quat"])
        return (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4

    def command(self, row):
        if self.phase == "align":
            action = list(self.ref)
            err = wrap(self.align_target - yaw_of(row["root_quat"]))
            want = math.copysign(min(0.2, max(0.08, abs(err))), err) if abs(err) > math.radians(2) else 0.0
            self.rate += max(-0.01, min(0.01, want - self.rate))
            action[43:46] = [0.0, 0.0, self.rate]
            return action
        if self.phase == "acquire":
            action = list(self.acquire(row))
            self.ref = list(action)
            return action
        action = list(self.ref)
        action[43:46] = [0., 0., 0.]
        if self.phase == "lift" and self.motion is not None:
            q = row["root_quat"]
            action[:43] = self.motion.command(row["joint_pos"], row["root_pos"], [*q[1:], q[0]], action[:43],
                                              row["time"] - self.phase_start + DT)
            self.ref = list(action)
        return action

    def update(self, row):
        self.rows.append(row)
        now = row["time"]
        fresh = row.get("box") and row["box"]["age_s"] < 1e-6
        if self.phase == "align":
            err = wrap(self.align_target - yaw_of(row["root_quat"]))
            if abs(err) <= math.radians(3) and abs(self.rate) < 0.01:
                self.phase, self.start = "acquire", now   # acquisition timeout counts from here
            elif now - self.start >= 10:
                return self.finish("failed", "align_to_box_timeout")
            return None
        if self.phase == "acquire":
            if fresh:
                # Hand over from GR00T only once the box is clearly lifted (>= 3.5 cm
                # for two 10 Hz frames); a 2 cm hand-off let the box settle back.
                raised = row["box_between_hands"] and (row.get("clearance") or 0) >= 0.035
                self.count = self.count + 1 if raised else 0
            if self.count >= 2:
                self.phase, self.phase_start = "verify", now
                self.rows.clear()
            elif now - self.start >= 15:
                return self.finish("failed", "acquisition_timeout")
            return None
        if not row["box_between_hands"]:
            return self.finish("failed", "grasp_lost")
        hold = camera_rate_hold(self.rows)
        self.advice = dict(hold, box_height_above_source_m=row.get("clearance"))
        if self.phase == "verify":
            if hold["ready"] and (row.get("clearance") or 0) >= 0.05:
                return self.finish("completed", "raised_and_stable")
            gripped = camera_rate_hold(self.rows, require_held=False)["ready"]   # may still rest on the table
            if (hold["ready"] or gripped) and now - self.phase_start >= 1.0:
                lift = min(0.06, 0.08 - (row.get("clearance") or 0))
                self.motion = self.wrist_motion(row, [0., 0., lift])
                self.phase, self.phase_start, self.lift = "lift", now, lift
            elif now - self.phase_start >= 5:
                return self.finish("failed", "pickup_hold_timeout")
        elif self.phase == "lift" and now - self.phase_start >= self.lift / 0.01 + 1.0:
            self.phase, self.phase_start = "verify_lift", now
            self.rows.clear()
        elif self.phase == "verify_lift":
            if hold["ready"]:
                ok = (row.get("clearance") or 0) >= 0.04
                return self.finish("completed" if ok else "failed", "raised_and_stable" if ok else "height_goal_not_reached")
            if now - self.phase_start >= 5:
                return self.finish("failed", "pickup_hold_timeout")


class Raise(Skill):
    """Raise the held box by shoulder pitch (pitch_lift), up to 6 cm per call."""

    def __init__(self, row, action, clearance_m, arm_index):
        super().__init__(row, action)
        if not row["box_held"]:
            raise ValueError("box_not_held")
        if not 0.03 <= clearance_m <= 0.15:
            raise ValueError("height_outside_envelope")
        current = row.get("clearance") or 0
        self.goal = clearance_m
        self.lift = max(0., min(0.06, clearance_m - current))
        self.arm_index = list(arm_index)
        self.lift_from = [self.ref[i] for i in self.arm_index]
        self.dpitch = self.lift / GRASP_LIFT_PER_RAD
        self.lift_time = max(1.5, self.dpitch / 0.06)
        self.phase = "lift" if self.lift > 0.005 else "verify"
        self.carrying = True

    def command(self, row):
        action = list(self.ref)
        if self.phase == "lift":
            pitch_lift(action, self.arm_index, self.lift_from,
                       min(1.0, (row["time"] - self.start + DT) / self.lift_time) * self.dpitch)
            self.ref = list(action)
        return action

    def update(self, row):
        self.rows.append(row)
        if self.phase == "lift" and row["time"] - self.start >= self.lift_time + 0.5:
            self.phase, self.verify_start = "verify", row["time"]
            self.rows.clear()
        elif self.phase == "verify":
            start = getattr(self, "verify_start", self.start)
            hold = camera_rate_hold(self.rows)
            self.advice = dict(hold, box_height_above_source_m=row.get("clearance"))
            if hold["ready"]:
                short = self.goal - (row.get("clearance") or 0)
                if short > 0.01 and not getattr(self, "topped_up", False):
                    # One follow-up for the remainder: the lift per rad varies with posture
                    # (raise-cube: asked 4.7 cm, got 3.6).
                    self.topped_up = True
                    self.lift_from = [self.ref[i] for i in self.arm_index]
                    self.dpitch = min(0.06, short) / GRASP_LIFT_PER_RAD
                    self.lift_time = max(1.5, self.dpitch / 0.06)
                    self.start, self.phase = row["time"], "lift"
                    return None
                ok = short <= 0.01
                return self.finish("completed" if ok else "failed", "raised_and_stable" if ok else "height_goal_not_reached")
            if row["time"] - start >= 5:
                return self.finish("failed", "raise_hold_timeout")


class Walk(Skill):
    """Odom-frame goal from a body-frame request; 0.22 m/s cap, 0.3 m/s^2 ramp."""

    def __init__(self, row, action, forward_m, left_m=0.0, yaw_rad=0.0):
        super().__init__(row, action)
        dist = math.hypot(forward_m, left_m)
        if not 0.05 <= dist <= 1.5:
            raise ValueError("distance_outside_envelope")
        if not math.isfinite(yaw_rad) or abs(yaw_rad) > math.pi / 2:
            raise ValueError("angle_outside_envelope")
        yaw = yaw_of(row["root_quat"])
        c, s = math.cos(yaw), math.sin(yaw)
        x0, y0 = row["root_pos"][:2]
        self.goal = (x0 + c * forward_m - s * left_m, y0 + s * forward_m + c * left_m)
        self.yaw0, self.yaw_delta, self.yaw_goal, self.dist = yaw, yaw_rad, wrap(yaw + yaw_rad), dist
        self.v = np.zeros(2)
        self.carrying = row["box_held"]
        self.phase = "walk"
        self.timeout = dist / 0.08 + 6.0

    def remaining(self, row):
        return np.array(self.goal) - np.array(row["root_pos"][:2])

    def command(self, row):
        action = list(self.ref)
        err = self.remaining(row)
        if self.phase == "walk":
            speed = 0.22 if float(np.linalg.norm(err)) > 0.15 else 0.09   # HOMIE deadband ~0.06 m/s
            want = err / max(float(np.linalg.norm(err)), 1e-9) * speed
            self.v += np.clip(want - self.v, -0.006, 0.006)
        else:
            self.v += np.clip(-self.v, -0.006, 0.006)
        yaw = yaw_of(row["root_quat"])
        c, s = math.cos(yaw), math.sin(yaw)
        action[43:45] = [c * self.v[0] + s * self.v[1], -s * self.v[0] + c * self.v[1]]
        if self.phase == "walk":
            # Advance heading with distance, so a requested turn is made while translating.
            # The odom-frame XY goal is fixed; this is a curved command, not path planning.
            progress = 1.0 - min(1.0, float(np.linalg.norm(err)) / self.dist)
            target_yaw = wrap(self.yaw0 + self.yaw_delta * progress)
        else:
            target_yaw = self.yaw_goal
        cap = 0.15 if self.carrying else 0.25  # Loaded turns above 0.15 rad/s loosened the grip.
        action[45] = max(-cap, min(cap, 1.2 * wrap(target_yaw - yaw))) if self.phase != "settle" else 0.0
        return action

    def update(self, row):
        self.rows.append(row)
        err = float(np.linalg.norm(self.remaining(row)))
        heading_error = wrap(self.yaw_goal - yaw_of(row["root_quat"]))
        self.advice = dict(remaining_m=err, heading_error_rad=heading_error,
                           odometry_note="leg odometry; ~10-20% of distance walked")
        if self.phase == "walk":
            # 4 cm: odometry is only good to 10-20% of the distance walked, and HOMIE will not
            # creep below ~0.06 m/s; at 2 cm walks timed out 2-4 cm short (sweep-d scene 05).
            if err <= 0.04:
                if abs(heading_error) > math.radians(6):
                    self.phase = "trim"
                else:
                    self.phase, self.settle_start = "settle", row["time"]
            elif row["time"] - self.start >= self.timeout:
                return self.finish("failed", "walk_timeout")
        elif self.phase == "trim":
            if abs(heading_error) <= math.radians(6):
                self.phase, self.settle_start = "settle", row["time"]
            elif row["time"] - self.start >= self.timeout:
                return self.finish("failed", "walk_timeout")
        elif row["time"] - self.settle_start >= 1.0 and float(np.linalg.norm(self.v)) < 1e-3:
            # At the 4 cm brake trigger, stopping from the 0.09 m/s near command
            # over a 0.3 m/s² ramp takes about 1.4 cm; allow that drift plus
            # odometry noise at the final stopped check.
            ok = err <= 0.06 and abs(heading_error) <= math.radians(6)
            if ok:
                return self.finish("completed", "walked_and_stopped")
            if row["time"] - self.start >= self.timeout:
                return self.finish("failed", "walk_timeout")
            # Braking can move the odometry estimate outside its own completion
            # window. Re-enter the same bounded goal instead of reporting a miss.
            self.phase = "walk" if err > 0.06 else "trim"


class Turn(Skill):
    """Relative yaw from the gyro; rate law and braking model reused from the privileged LoadedTurn."""

    def __init__(self, row, action, yaw_rad):
        super().__init__(row, action)
        if not 0.02 <= abs(yaw_rad) <= math.pi:
            raise ValueError("angle_outside_envelope")
        self.target = wrap(yaw_of(row["root_quat"]) + yaw_rad)
        self.delta = yaw_rad
        self.anchor = list(row["root_pos"][:2])
        self.rate = 0.
        self.v = np.zeros(2)
        self.carrying = row["box_held"]
        self.phase = "turn"
        self.timeout = abs(yaw_rad) / 0.10 + 8.0

    def command(self, row):
        action = list(self.ref)
        err = wrap(self.target - yaw_of(row["root_quat"]))
        # Carried boxes slipped out at 0.25 rad/s (sweep-a scene 02): 0.15 rad/s when loaded.
        cap = 0.15 if self.carrying else 0.25
        want = math.copysign(min(cap, max(0.08, abs(err))), err) if self.phase == "turn" else 0.
        self.rate += max(-0.006, min(0.006, want - self.rate)) if self.carrying else max(-0.01, min(0.01, want - self.rate))
        # Hold position while turning: world gain 1/s, cap 4 cm/s.
        xy = np.array(self.anchor) - np.array(row["root_pos"][:2]) if self.phase == "turn" else np.zeros(2)
        xy *= min(1., 0.04 / max(float(np.linalg.norm(xy)), 1e-9))
        self.v += np.clip(xy - self.v, -0.006, 0.006)
        yaw = yaw_of(row["root_quat"])
        c, s = math.cos(yaw), math.sin(yaw)
        action[43:46] = [c * self.v[0] + s * self.v[1], -s * self.v[0] + c * self.v[1], self.rate]
        return action

    def update(self, row):
        self.rows.append(row)
        err = wrap(self.target - yaw_of(row["root_quat"]))
        self.advice = dict(heading_error_rad=err)
        if self.phase == "turn":
            brake = self.rate * self.rate / (2 * 0.5) + math.radians(2)
            if abs(err) <= brake:
                self.phase, self.settle_start = "settle", row["time"]
            elif row["time"] - self.start >= self.timeout:
                return self.finish("failed", "turn_timeout")
        elif row["time"] - self.settle_start >= 1.5 and abs(self.rate) < 1e-3:
            ok = abs(err) <= math.radians(6)
            return self.finish("completed" if ok else "failed", "turned_and_stopped" if ok else "heading_outside_tolerance")


class Approach(Skill):
    """Go to a pose `standoff_m` out from the middle of the table edge facing the robot, then square up.

    1. plan: from the mapped footprint (odom, landmark-corrected while in view)
       pick the edge whose outward normal faces the robot; goal = edge midpoint
       + standoff along the normal, goal yaw = facing the edge.
    2. turn to the goal yaw where the robot is (away from the table),
    3. walk to the goal holding that heading (forward and sideways), 4. small yaw trim,
    5. final: in/out adjustment from the current camera view of the edge.
    Nearness to a mapped footprint corner (legs) is reported as advice, not a stop.
    HOMIE ignores < ~0.06 m/s, so walking commands are 0.09-0.12 m/s per axis or zero.
    """
    MIN_V = 0.09

    def __init__(self, row, action, surface_id, standoff_m=0.24, view=None, robot_bounds=None):
        super().__init__(row, action)
        if surface_id not in ("source", "destination"):
            raise ValueError("unsupported_surface")
        t = row["tables"].get(surface_id)
        if t is None:
            raise ValueError("surface_not_observed")
        if not 0.2 <= standoff_m <= 0.6:
            raise ValueError("standoff_outside_envelope")
        self.surface, self.standoff, self.view, self.bounds = surface_id, standoff_m, view, robot_bounds or {}
        self.carrying = row["box_held"]
        (x0, y0), (x1, y1) = t["min_xy"], t["max_xy"]
        px, py = row["root_pos"][:2]
        edges = [((x0, (y0 + y1) / 2), (-1.0, 0.0)), ((x1, (y0 + y1) / 2), (1.0, 0.0)),
                 (((x0 + x1) / 2, y0), (0.0, -1.0)), (((x0 + x1) / 2, y1), (0.0, 1.0))]
        mid, n = max(edges, key=lambda e: (px - e[0][0]) * e[1][0] + (py - e[0][1]) * e[1][1])
        self.goal = (mid[0] + n[0] * (standoff_m + 0.05), mid[1] + n[1] * (standoff_m + 0.05))
        self.edge_mid, self.edge_normal = mid, n
        self.goal_yaw = math.atan2(-n[1], -n[0])
        self.phase, self.rate, self.v, self.vy, self.sidestep = "turn_to_goal", 0.0, 0.0, 0.0, 0.0
        if math.dist((px, py), self.goal) < 0.15:
            self.phase = "face_edge"
        # A retained rectangular carry needed 57.12 s to finish its camera alignment,
        # 2.96 s beyond the old loaded deadline. Keep the same speed and bounded
        # distance allowance; give only a carried approach 20 s more final-view time.
        self.timeout = (50.0 if self.carrying else 30.0) + math.dist((px, py), self.goal) / 0.05
        self.advice = dict(goal_xy=list(self.goal), goal_yaw_rad=self.goal_yaw)

    def view_edge(self, row):
        """(distance to near edge ahead, yaw error of that edge) from the current view, or None."""
        pts = self.view(self.surface, row["time"]) if self.view else None
        if pts is None or len(pts) < 150:
            return None
        ahead = pts[pts[:, 0] > 0.05]
        if len(ahead) < 100:
            return None
        bins = np.floor(ahead[:, 1] / 0.04).astype(int)
        edge = np.array([[ahead[bins == k, 1].mean(), ahead[bins == k, 0].min()] for k in np.unique(bins)
                         if np.sum(bins == k) >= 3])
        near = edge[edge[:, 1] <= edge[:, 1].min() + 0.06]
        yaw_err, dist = 0.0, float(np.median(near[:, 1]))
        if len(near) >= 4 and np.ptp(near[:, 0]) >= 0.2:
            b, a = np.polyfit(near[:, 0], near[:, 1], 1)
            yaw_err = -math.atan(b)
            dist = float(a + b * np.clip(0.0, near[:, 0].min(), near[:, 0].max()))
        return dist, yaw_err

    def command(self, row):
        action = list(self.ref)
        px, py = row["root_pos"][:2]
        yaw = yaw_of(row["root_quat"])
        want_v = want_rate = 0.0
        want_vy = 0.0
        if self.phase == "turn_to_goal":
            # Rotate first, far from the table, to the final heading. Spinning 110-180 deg
            # at the goal beside the table swung a hip into a leg (cube-11 380 N; sweep-h s43
            # 370 N); every close pass came from a face_edge turn over ~110 deg.
            err = wrap(self.goal_yaw - yaw)
            cap = 0.15 if self.carrying else 0.25  # Match Turn's loaded grip limit.
            want_rate = math.copysign(min(cap, max(0.08, abs(err))), err)
            outward_m = (px - self.edge_mid[0]) * self.edge_normal[0] + (py - self.edge_mid[1]) * self.edge_normal[1]
            # The held box can reach about 0.5 m ahead of the pelvis. Overlap only the
            # last 30 degrees of alignment with a slow walk while still far from the edge.
            if abs(err) <= math.radians(30) and outward_m > self.standoff + 0.6:
                dx, dy = self.goal[0] - px, self.goal[1] - py
                fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
                left = -dx * math.sin(yaw) + dy * math.cos(yaw)
                want_v = math.copysign(0.12, fwd) if abs(fwd) > 0.04 else 0.0
                want_vy = math.copysign(0.12, left) if abs(left) > 0.04 else 0.0
        elif self.phase == "walk_to_goal":
            # Then translate with the heading held (forward and sideways), so the robot
            # arrives already facing the edge. HOMIE ignores < ~0.06 m/s per axis.
            ex, ey = self.goal[0] - px, self.goal[1] - py
            fwd, left = ex * math.cos(yaw) + ey * math.sin(yaw), -ex * math.sin(yaw) + ey * math.cos(yaw)
            distance = math.hypot(fwd, left)
            if distance > 0.3:
                # Keep the distant loaded approach at 0.12 m/s total. A matched
                # seed-44 faster approach reached 68° box tilt before placement.
                want_v = 0.12 * fwd / distance if abs(fwd) > 0.04 else 0.0
                want_vy = 0.12 * left / distance if abs(left) > 0.04 else 0.0
            else:
                # Near the goal, each small active axis must clear HOMIE's ~0.06 m/s deadband.
                want_v = math.copysign(self.MIN_V, fwd) if abs(fwd) > 0.04 else 0.0
                want_vy = math.copysign(self.MIN_V, left) if abs(left) > 0.04 else 0.0
            want_rate = max(-0.2, min(0.2, 0.8 * wrap(self.goal_yaw - yaw)))
        elif self.phase == "face_edge":
            err = wrap(self.goal_yaw - yaw)
            want_rate = math.copysign(min(0.25, max(0.08, abs(err))), err)
        elif self.phase == "final":
            edge = self.view_edge(row)
            margins = self.lateral_margins(row)
            if margins is not None:
                left, right = margins
                # Sidestep toward the table centre when one side is short (<= 0.4 m total).
                if (right is None or right < 0.08) and left is not None and left > 0.15 and self.sidestep < 0.4:
                    want_vy = 0.06
                elif (left is None or left < 0.08) and right is not None and right > 0.15 and self.sidestep < 0.4:
                    want_vy = -0.06
            if edge is not None:
                dist, yaw_err = edge
                gap = dist - self.standoff
                # Hysteresis: stop inside [-0.04, 0.03]; resume only outside [-0.06, 0.06].
                moving = abs(self.v) > 1e-3
                if moving:
                    want_v = self.MIN_V if gap > 0.03 else (-self.MIN_V if gap < -0.04 else 0.0)
                else:
                    want_v = self.MIN_V if gap > 0.06 else (-self.MIN_V if gap < -0.06 else 0.0)
                want_rate = max(-0.15, min(0.15, 0.8 * yaw_err)) if abs(yaw_err) > math.radians(3) else 0.0
        self.rate += max(-0.01, min(0.01, want_rate - self.rate))
        self.v += max(-0.006, min(0.006, want_v - self.v))
        self.vy += max(-0.006, min(0.006, want_vy - self.vy))
        planar_speed = math.hypot(self.v, self.vy)
        if planar_speed > 0.20:
            # Per-axis ramps can briefly put the combined command over the
            # 0.20 m/s approach envelope even when the target vector is capped.
            scale = 0.20 / planar_speed
            self.v *= scale
            self.vy *= scale
        self.sidestep += abs(self.vy) * DT
        action[43:46] = [self.v, self.vy, self.rate]
        return action

    def lateral_margins(self, row):
        """(left, right) table extent beyond the carried box in the current view, or None."""
        if row.get("box") is None or not self.view:
            return None
        pts = self.view(self.surface, row["time"])
        if pts is None or len(pts) < 150:
            return None
        m = Place.view_margins(row, pts)
        return m["left"], m["right"]

    def update(self, row):
        self.rows.append(row)
        now = row["time"]
        px, py = row["root_pos"][:2]
        yaw = yaw_of(row["root_quat"])
        # Table legs stand at the footprint corners; feet under the top edge are fine.
        # Keep pelvis and knees clear of each mapped corner while walking (CORNER_KEEPOUT_M).
        t = frozen_map(self, row, "tables")[self.surface]
        corners = [(t["min_xy"][0], t["min_xy"][1]), (t["min_xy"][0], t["max_xy"][1]),
                   (t["max_xy"][0], t["min_xy"][1]), (t["max_xy"][0], t["max_xy"][1])]
        parts = [row["body_poses"][n]["pos"][:2] for n in ("pelvis", "left_knee_link", "right_knee_link") if n in row["body_poses"]]
        corner = min(math.dist(c, q) for c in corners for q in parts)
        edge = self.view_edge(row)
        self.advice.update(phase=self.phase, distance_to_goal_m=math.dist((px, py), self.goal),
                           nearest_corner_m=corner, view_edge=edge,
                           table_uncertainty_m=row["tables"][self.surface]["position_uncertainty_m"])
        # Advice only, as for walk/turn: in sweep-f all 11 stops here were false (estimate ~0.21 m,
        # truth 0.28-0.57 m; the destination map smears with odometry drift), and none of them
        # prevented a contact. A real bump still fails the task in scoring.
        if corner < CORNER_KEEPOUT_M and not getattr(self, "corner_warned", False):
            self.corner_warned = True
            self.advice["near_table_corner"] = True
        if now - self.start >= self.timeout:
            return self.finish("failed", "approach_timeout")
        if self.phase == "turn_to_goal":
            if abs(wrap(self.goal_yaw - yaw)) <= math.radians(5) and abs(self.rate) < 0.05:
                self.phase = "walk_to_goal"
        elif self.phase == "walk_to_goal":
            ex, ey = self.goal[0] - px, self.goal[1] - py
            fwd, left = ex * math.cos(yaw) + ey * math.sin(yaw), -ex * math.sin(yaw) + ey * math.cos(yaw)
            if abs(fwd) <= 0.05 and abs(left) <= 0.05:
                self.phase = "face_edge"
        elif self.phase == "face_edge":
            if abs(wrap(self.goal_yaw - yaw)) <= math.radians(4) and abs(self.rate) < 0.02:
                self.phase = "final"
        elif self.phase == "final":
            if edge is None:
                if now - self.start >= self.timeout - 5:
                    return self.finish("failed", "surface_not_in_view")
            else:
                dist, yaw_err = edge
                lr = self.lateral_margins(row) if self.carrying else None
                centred = lr is None or not ((lr[0] is None or lr[0] < 0.08 or lr[1] is None or lr[1] < 0.08)
                                             and self.sidestep < 0.4 and abs(self.vy) > 1e-3)
                if -0.06 <= dist - self.standoff <= 0.06 and abs(yaw_err) <= math.radians(6) and centred and abs(self.vy) < 1e-3:
                    if not hasattr(self, "settle_start"):
                        self.settle_start = now
                    elif now - self.settle_start >= 1.5 and abs(self.v) < 1e-3:
                        return self.finish("completed", "at_surface")
                elif hasattr(self, "settle_start"):
                    del self.settle_start


FINGERS = slice(29, 43)   # Dex3 finger references in the 43-joint native order

# Arm references at the moment five successful GR00T grasps first closed firmly on
# the box (gate-a-s42/43/44, smoke-11/15), with the box pose those grasps had in
# the pelvis frame (standing position is now set by lineup_target). Joint spread across the five is <= 0.11 rad. The hands squeeze
# the box between the palms; finger references stay open throughout.
GRASP_ARM_JOINTS = ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
                    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
                    "left_elbow_joint", "right_elbow_joint",
                    "left_wrist_roll_joint", "right_wrist_roll_joint",
                    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
                    "left_wrist_yaw_joint", "right_wrist_yaw_joint"]
GRASP_ARM_REFS = dict(zip(GRASP_ARM_JOINTS,
                          [-0.4871, -0.5621, 0.0804, 0.0639, 0.0476, 0.0603, -0.0515,
                           0.2596, 0.0405, -0.1530, 0.2871, -0.1822, -0.0814, 0.0704]))
# Where the box sits relative to the wrist-yaw midpoint in those same five grasps
# (spread 2.5/1.0/1.2 cm). Gripping higher than this tips the box about its bottom
# edge instead of lifting it (trial 9), so the hands are aimed off the box itself.
# x is well forward of those grasps (0.230). Lifted boxes turn front-down inside the pinch
# (rect-11 17.8 deg, rect-12/13 and cube-13 4-7 deg): the pinch acts as a hinge near the
# box's top edge, 7-16 cm ahead of the palm-link midpoint (mean ~0.12 m, fitted from box
# motion in the palm frame), while the box centre sat 0.17-0.19 m ahead. A box hanging
# from a hinge above its centre is level only when the hinge is straight above the centre,
# so aim the centre ~0.12 m ahead of the palms. Measured shifts end ~1.5 cm short (cube-13:
# target 0.19, reached 0.207), hence 0.145. Keep the height: a hinge at the centre's
# height would give no restoring torque at all.
# z: grip at the box's centre height, as GR00T holds its box while carrying (box centre
# 0.01 m below the wrist midpoint in sweep-a 01, sweep-b 04, sweep-c 10; max tilt 8-12 deg).
# -0.034 was GR00T's first-contact height; our carries at ~-0.05 twisted 16-45 deg.
GRASP_BOX_FROM_WRISTS = np.array([0.145, 0.000, -0.010])
GRASP_MAX_SHIFT_M = 0.06     # the paired wrist controller's common-translation limit
# Where `reach` leaves the wrist-yaw midpoint in the pelvis frame: (0.294, 0.021-0.025,
# 0.189-0.196) in cube-11/13/14 and rect-13/14. The wrist IK cannot carry the hands much
# further (cube-14: asked for 12 cm forward and 5 cm down, it refolded the elbows from 0
# to 1.4 rad and moved the wrists 6 cm forward, none down). So the body lines up for the
# forward offset and a joint-space `lower` (the lift in reverse) supplies the height.
GRASP_REACH_WRIST_IN_PELVIS = np.array([0.294, 0.022, 0.193])
GRASP_LOWER_BACK_PER_RAD = 0.14  # wrists draw back per rad of lowering pitch (FK, with wrist counter-pitch)
# Reach this much further back than the final stance, then step in with the arms up and
# open. Reaching at the final stance swept the rising hands into the box's rear face
# (grasp-q-*-15: pushed 3.5-4 cm, tilted 7.5-14 deg); at the old ~0.55 m stance the reach
# touched nothing in trials 11-14, and the fingertips end above the top and beside the faces.
GRASP_REACH_STANDOFF_M = 0.14
# Raise the hands this much (the lift motion, before any contact) before stepping in: at
# reach height the right hand's lowest fingers skimmed the 20 cm cube's top and caught it
# near the end of the step (grasp-q-cube-17: 4 N down on the box, tilted 7 deg).
GRASP_STEP_IN_RAISE_M = 0.04
GRASP_LINEUP_MAX_M = 0.40    # further than this, the caller should approach or walk first
GRASP_WIDTH_M = 0.20         # box width those grasps closed on
GRASP_SPAN_PER_RAD = 0.435   # palm separation gained per rad of symmetric shoulder roll (measured)
# The one real knee/table-leg collision (agent-01) built force from 0.185 m and hit
# 399 N at 0.10 m, measured pelvis/knees to footprint corner. 0.22 m keeps a 3-4 cm
# margin over the onset; 0.30 m rejected approaches that never touched anything.
CORNER_KEEPOUT_M = 0.22
# A proximity stop fires only when the motion has made things worse than where it started by
# more than this. The step-to-step test (any 0.1 mm decrease) tripped on gait sway and blocked
# walking away from a table: sweep-c, 6/12 episodes deadlocked with no true contact.
GUARD_WORSEN_M = 0.005


def worsened(skill, key, value):
    """True if `value` (a clearance or distance, m) fell more than GUARD_WORSEN_M below the
    value first seen inside the limit during this skill. Callers check the limit first, so a
    motion that starts inside it may always move out, and one entering it stops 5 mm in."""
    start = skill.__dict__.setdefault("guard_start", {}).setdefault(key, value)
    return value < start - GUARD_WORSEN_M


def frozen_map(skill, row, field):
    """`row[field]` ("tables" or "surfaces") with each table as it was first seen during this
    skill. The map keeps growing while the robot moves (the source far edge ends 13-41 cm
    beyond the real one, sweep-c/d), so a live map made retreating look like approaching."""
    snap = skill.__dict__.setdefault("guard_map_" + field, {})
    for name, t in (row.get(field) or {}).items():
        if t is not None:
            snap.setdefault(name, t)
    return {name: snap[name] for name in snap}
GRASP_SQUEEZE_RAD = 0.06     # extra inward shoulder roll held while carrying (~2.6 cm, ~12 N)
GRASP_OPEN_RAD = 0.10        # minimum extra shoulder opening held until the hands straddle the box
# Approach span (grasp-q-*-10): the right hand's index/middle fingers curl up to 5 cm
# inboard of its wrist and reach ~0.21 m ahead of it, so with the wrists only ~5 cm
# outside the faces the fingertips came down on the box top while the wrists
# descended (46 N, box tipped 18-32 deg). Open until those fingers clear the faces.
GRASP_REF_SPAN_M = 0.226     # wrist-yaw span at GRASP_ARM_REFS (rect-10: 0.335 m at +0.26 rad)
GRASP_FINGER_INBOARD_M = 0.085  # finger reach inboard of the wrist (5 cm) + pad + 2 cm margin
GRASP_MAX_OPEN_RAD = 0.45
# Thumbs swung out (thumb_1 at its outward limit) from reach onward. At rest the distal
# thumb links reach 3.5-7 cm inboard of the box faces, and once the body stands close they
# came down on the box top (grasp-q-rect-18 lower: 15 N down; cube-18 step-in). With these,
# every Dex3 collision box stays outside the faces (FK + bounds at the rect-18/cube-18 poses:
# left 0.3 cm clear before the extra centimetre of opening above, right 1.9 cm).
GRASP_THUMB_OUT = {"left_hand_thumb_1_joint": -0.72, "right_hand_thumb_1_joint": 0.72}
# Wrist-midpoint motion per rad of closing (pelvis x, y, z), measured in rect-10's close:
# the hands drop and draw back slightly as the shoulders adduct.
GRASP_CLOSE_WRIST_PER_RAD = np.array([-0.059, 0.0, -0.083])
# Contact while shifting, sensor-only: any arm joint lagging its command by more than
# this. Free shifting lags <= 0.009 rad (cube-8/9, rect-8/9/10); a hand on the box
# passes 0.03 rad at 2-3 N (rect-10: right elbow 0.021 at 1.2 N, 0.050 at 4.9 N).
GRASP_SHIFT_CONTACT_LAG_RAD = 0.03
GRASP_DISTURB_M = 0.02       # backstop: the box moved this far before closing
GRASP_DISTURB_TILT_RAD = 0.07  # or tilted this much (~4 deg)
GRASP_LIFT_PER_RAD = 0.31    # hand rise per rad of shoulder pitch at this posture (measured)
# Roll opening per rad of lift pitch that keeps the palm span constant, with the wrists
# counter-pitched as below. FK at the cube-11/12 and rect-11 grasps: 0.21-0.23 (0.27
# without the wrist term). 0.40 opened the palms 2.3-4 cm per lift and both 12-series
# grips decayed 11 N -> 0 while lifting.
GRASP_SPAN_PER_PITCH = 0.22
GRASP_CONTACT_LAG_RAD = 0.025  # encoder lag that means the palms are loaded (~5-8 N)
GRASP_PINCH_RAD = 0.05       # extra closing past first contact (~12 N; 0.035 gave ~9 N and the 0.5 kg box slipped)
# Carrying: keep the palm load (shoulder-roll lag) from decaying. cube-19's grip fell from
# 17 to 11 N while walking and the box turned 18 deg in the palms before dropping.
GRIP_KEEP_FRACTION = 0.9     # re-tighten below this fraction of the load measured at pickup
GRIP_KEEP_RATE = 0.02        # rad/s of extra inward roll while below it
GRIP_KEEP_MAX_RAD = 0.06     # total extra closing allowed while carrying
GRIP_KEEP_METHODS = ("hold_box", "walk", "turn", "approach_surface", "wait")


def pitch_lift(action, arm_index, start, dpitch):
    """Raise both hands by dpitch rad of shoulder pitch (GRASP_LIFT_PER_RAD m/rad), keeping the
    grip: wrist pitch counter-rotates the hands and shoulder roll holds the palm span.
    `arm_index` / `start` follow GRASP_ARM_JOINTS. Wrist IK is not used: it re-solves the
    arm and lets a palm pinch go (trial 8; sweep-k: raise_held_box dropped the box twice)."""
    for k, (joint, i) in enumerate(zip(GRASP_ARM_JOINTS, arm_index)):
        if joint.endswith("shoulder_pitch_joint"):
            action[i] = start[k] - dpitch
        elif joint.endswith("wrist_pitch_joint"):
            action[i] = start[k] + dpitch
        elif joint == "left_shoulder_roll_joint":
            action[i] = start[k] + GRASP_SPAN_PER_PITCH * dpitch
        elif joint == "right_shoulder_roll_joint":
            action[i] = start[k] - GRASP_SPAN_PER_PITCH * dpitch


class Grasp(Skill):
    """Scripted two-hand grasp: recorded closing posture, shifted by the measured box offset.

    Commanding wrist poses directly does not work: local IK stalls about a
    centimetre short of contact (qualification trials 4 and 5), because the
    recorded hand spacing is where the hands rest while already pressing. Instead
    this adopts the arm posture those grasps closed with, then moves both hands
    together by the difference between the measured box position and the recorded
    one. It does not walk: the box must already be within reach.
    """

    def __init__(self, row, action, wrist_motion, names, lift_m=0.06):
        super().__init__(row, action)
        if row.get("box") is None or row["box"]["age_s"] > 0.25:
            raise ValueError("box_not_observed")
        if row["box_held"]:
            raise ValueError("box_already_held")
        offset = Pickup.box_yaw_offset(row) or 0.0
        box = self.box_in_pelvis(row)
        self.advice = {}
        # A wider box needs the palms further apart: open both shoulders symmetrically.
        self.set_width(row)
        error = self.lineup_error(row)
        if float(np.linalg.norm(error)) > GRASP_LINEUP_MAX_M:
            raise ValueError("box_out_of_reach",
                             dict(box_in_pelvis_m=[round(v, 3) for v in box],
                                  wanted_in_pelvis_m=[round(v, 3) for v in self.lineup_target(row)],
                                  suggest=f"approach_surface(\"source\") or walk(forward_m={error[0]:+.2f}, "
                                          f"left_m={error[1]:+.2f}), then grasp again"))
        self.wrist_motion, self.names = wrist_motion, list(names)
        self.yaw0 = yaw_of(row["root_quat"])
        self.v = np.zeros(2)
        index = {n: i for i, n in enumerate(self.names)}
        missing = [n for n in GRASP_ARM_JOINTS if n not in index]
        if missing:
            raise ValueError("arm_joints_not_found", dict(missing=missing))
        self.arm_index = [index[n] for n in GRASP_ARM_JOINTS]
        self.start_refs = [self.ref[i] for i in self.arm_index]
        self.thumb_index = {index[n]: v for n, v in GRASP_THUMB_OUT.items() if n in index}
        self.thumb_start = {i: self.ref[i] for i in self.thumb_index}
        self.lift = min(GRASP_MAX_SHIFT_M, max(0.03, lift_m))
        self.shift = np.zeros(3)
        self.lifts = 0
        # The palms must meet flat faces, so square the body to the box first.
        self.square_error = offset
        self.square_target = wrap(self.yaw0 + offset)
        self.rate = 0.0
        self.phase = "square_up" if abs(offset) > 0.12 else "lineup"
        self.phase_start = row["time"]
        self.timeout = float(np.linalg.norm(error)) / 0.05 + 10.0
        self.advice = dict(box_width_m=round(self.width, 3), shoulder_open_rad=round(self.open_rad, 3))

    def set_width(self, row):
        """Shoulder opening for the box width across the hands (pelvis y).

        Recomputed once the body is squared up: the width projected at the initial
        yaw read 0.292 m for a 0.24 m box turned 0.35 rad (rect-8/9/10).
        """
        self.width = self.grasp_width(row)
        self.roll_delta = float(np.clip((self.width - GRASP_WIDTH_M) / GRASP_SPAN_PER_RAD, -0.08, 0.16))
        clear = (self.width + 2 * GRASP_FINGER_INBOARD_M - GRASP_REF_SPAN_M) / GRASP_SPAN_PER_RAD
        self.open_rad = float(np.clip(clear, self.roll_delta + GRASP_OPEN_RAD, GRASP_MAX_OPEN_RAD))
        # Closing travel expected before the palms load: from the open span down to the
        # span the recorded grasps pressed at (width + 0.065 m).
        contact = (self.width + 0.065 - GRASP_REF_SPAN_M) / GRASP_SPAN_PER_RAD
        self.expected_travel = max(0.0, self.open_rad - contact)
        self.advice.update(box_width_m=round(self.width, 3), shoulder_open_rad=round(self.open_rad, 3))

    @staticmethod
    def box_tilt(row):
        """Angle (rad) between the estimated box's nearest-vertical axis and world z."""
        w, x, y, z = row["box"]["quat"]
        zrow = [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        return float(math.acos(min(1.0, max(abs(v) for v in zrow))))

    def start_close(self, now):
        self.close_from = [self.ref[i] for i in self.arm_index]
        self.motion = None
        self.travel, self.contact_at = 0.0, None
        self.phase, self.phase_start = "close", now

    def start_lift(self, row, now, rise_m=None):
        rise = self.lift if rise_m is None else rise_m
        self.lift_from = [self.ref[i] for i in self.arm_index]
        self.dpitch = min(0.35, rise / GRASP_LIFT_PER_RAD)
        self.lift_time = max(1.5, self.dpitch / 0.06)
        self.lifts = getattr(self, "lifts", 0) + 1
        self.motion = None
        self.phase, self.phase_start = "lift", now

    def start_lower(self, now, dpitch):
        """The lift motion in reverse (same joints and compensations), before closing."""
        self.lift_from = [self.ref[i] for i in self.arm_index]
        self.reach_refs = list(self.lift_from)     # back_off returns here
        self.dpitch = -dpitch
        self.lift_time = max(1.5, abs(dpitch) / 0.06)
        self.phase, self.phase_start = "lower", now

    def start_shift_or_close(self, row, now):
        self.plan_shift(row)
        self.reach_refs = [self.ref[i] for i in self.arm_index]   # contact-free posture for back_off
        if float(np.linalg.norm(self.shift)) > 0.01:
            self.motion = self.wrist_motion(row, self.world_shift(row, self.shift).tolist())
            self.phase, self.phase_start = "shift", now
        else:
            self.start_close(now)

    box_before = None

    def disturbed(self, row, tilt_only=False):
        """Before closing, the box should not move: if it shifts or tilts the hands are on it.

        While the robot walks (step_in), the odom-frame box position drifts with leg
        odometry (~2 cm over 14 cm in grasp-q-*-16, box untouched), so only tilt counts.
        """
        if self.box_before is None or row.get("box") is None or row["box"]["age_s"] > 0.25:
            return False
        moved = 0.0 if tilt_only else float(np.linalg.norm(np.array(row["box"]["pos"]) - self.box_before[0]))
        tilted = self.box_tilt(row) - self.box_before[1]
        self.advice["box_moved_before_close_m"] = round(moved, 3)
        return moved > GRASP_DISTURB_M or tilted > GRASP_DISTURB_TILT_RAD

    def arm_lag(self, row):
        """Largest |commanded - measured| over the 14 arm joints (rad), from encoders."""
        return max(abs(self.ref[i] - row["joint_pos"][i]) for i in self.arm_index)

    def palm_load(self, row):
        """How far the shoulders lag their closing command: the sensor-only contact signal."""
        index = {n: i for i, n in enumerate(self.names)}
        left = row["joint_pos"][index["left_shoulder_roll_joint"]] - self.ref[index["left_shoulder_roll_joint"]]
        right = self.ref[index["right_shoulder_roll_joint"]] - row["joint_pos"][index["right_shoulder_roll_joint"]]
        return float(left), float(right)

    def preclose_offset(self):
        """Box centre minus wrist midpoint (pelvis frame) wanted before closing: the grasp
        offset, less the draw-back and drop the wrists make while the shoulders close."""
        return GRASP_BOX_FROM_WRISTS + self.expected_travel * GRASP_CLOSE_WRIST_PER_RAD

    def lower_pitch(self, wrist_z, box_z):
        """Shoulder pitch (rad) that lowers the wrists from `wrist_z` to the pre-close height."""
        drop = wrist_z - (box_z - self.preclose_offset()[2])
        return float(np.clip(drop / GRASP_LIFT_PER_RAD, 0.0, 0.35))

    def lineup_target(self, row):
        """Box position (pelvis x, y) to stand at, so that after reach and lower the
        wrists sit at the pre-close offset without a long IK shift."""
        box_z = self.box_in_pelvis(row)[2]
        dp = self.lower_pitch(GRASP_REACH_WRIST_IN_PELVIS[2], box_z)
        wrist = GRASP_REACH_WRIST_IN_PELVIS[:2] - [GRASP_LOWER_BACK_PER_RAD * dp, 0.0]
        if self.stepped_in:
            # Arms are raised and held: use where the wrists actually are, and the box height
            # measured standing (it bobs up to 5 cm in the pelvis frame while walking).
            dp = self.lower_pitch(self.step_wrist[2], self.step_box_z)
            wrist = self.step_wrist[:2] - [GRASP_LOWER_BACK_PER_RAD * dp, 0.0]
            return wrist + self.preclose_offset()[:2]
        target = wrist + self.preclose_offset()[:2]
        target[0] += GRASP_REACH_STANDOFF_M
        return target

    stepped_in = False

    def lineup_error(self, row):
        """Where the box is, minus where the grasp wants it (pelvis frame, x/y)."""
        return self.box_in_pelvis(row)[:2] - self.lineup_target(row)

    @staticmethod
    def wrist_mid_in_pelvis(row):
        w, x, y, z = row["root_quat"]
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        o = np.array(row["root_pos"])
        mid = 0.5 * (np.array(row["wrists_world"]["left_wrist_yaw_link"]["pos"])
                     + np.array(row["wrists_world"]["right_wrist_yaw_link"]["pos"]))
        return R.T @ (mid - o)

    def plan_shift(self, row):
        """Common wrist translation that puts the hands where the box is, from encoders."""
        # Aim so the wrists land on the recorded offset after closing moves them.
        want = self.box_in_pelvis(row) - GRASP_BOX_FROM_WRISTS - self.expected_travel * GRASP_CLOSE_WRIST_PER_RAD
        shift = want - self.wrist_mid_in_pelvis(row)
        norm = float(np.linalg.norm(shift))
        if norm > GRASP_MAX_SHIFT_M:
            shift *= GRASP_MAX_SHIFT_M / norm
        self.shift = shift
        self.advice["box_shift_m"] = [round(v, 3) for v in shift]
        self.advice["wrist_offset_m"] = [round(v, 3) for v in want - self.wrist_mid_in_pelvis(row)]

    @staticmethod
    def grasp_width(row):
        """Box extent across the hands (metres): its size projected on the pelvis y axis."""
        box = row.get("box") or {}
        size = box.get("size_m")
        if not size or "quat" not in box:
            return GRASP_WIDTH_M
        w, x, y, z = box["quat"]
        B = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        yaw = yaw_of(row["root_quat"])
        across = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        return float(sum(abs(float(across @ B[:, i])) * float(size[i]) for i in range(3)))

    @staticmethod
    def box_in_pelvis(row):
        w, x, y, z = row["root_quat"]
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        return R.T @ (np.array(row["box"]["pos"]) - np.array(row["root_pos"]))

    def world_shift(self, row, vector):
        w, x, y, z = row["root_quat"]
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        return R @ vector

    def command(self, row):
        action = list(self.ref)
        el = row["time"] - self.phase_start
        if self.phase in ("square_up", "square_settle"):
            err = wrap(self.square_target - yaw_of(row["root_quat"]))
            want = math.copysign(min(0.25, max(0.08, abs(err))), err) if self.phase == "square_up" else 0.0
            self.rate += max(-0.01, min(0.01, want - self.rate))
            action[43:46] = [0., 0., self.rate]
            self.ref = list(action)
            return action
        if self.phase in ("lineup", "lineup_settle", "step_in", "step_in_settle"):
            walking = self.phase in ("lineup", "step_in")
            if walking and row.get("box") is not None and row["box"]["age_s"] <= 0.5:
                err = self.lineup_error(row)
                distance = float(np.linalg.norm(err))
                speed = 0.10 if distance > 0.12 else 0.075     # HOMIE deadband is ~0.06 m/s
                want = err / max(distance, 1e-9) * speed
                self.v += np.clip(want - self.v, -0.006, 0.006)
            else:
                self.v += np.clip(-self.v, -0.006, 0.006)
            yaw = yaw_of(row["root_quat"])
            action[43:45] = list(self.v)
            action[45] = max(-0.2, min(0.2, wrap(self.yaw0 - yaw))) if walking else 0.0
            self.ref = list(action)
            return action
        if self.phase == "reach":
            # 3 s ramp to the recorded posture, but with the shoulders held wider: closing
            # on the way in clips a corner and shoves the box away (trial 7).
            f = min(1.0, (el + DT) / 3.0)
            for k, (joint, i) in enumerate(zip(GRASP_ARM_JOINTS, self.arm_index)):
                goal = GRASP_ARM_REFS[joint]
                if joint == "left_shoulder_roll_joint":
                    goal += self.open_rad
                elif joint == "right_shoulder_roll_joint":
                    goal -= self.open_rad
                action[i] = self.start_refs[k] + f * (goal - self.start_refs[k])
            for i, goal in self.thumb_index.items():
                action[i] = self.thumb_start[i] + f * (goal - self.thumb_start[i])
        elif self.phase == "close":
            # Close at 0.05 rad/s until the encoders say both palms are loaded, then pinch
            # a fixed amount further. Driving blind to a fixed span crushed a 20 cm box at
            # 25 N and squirted it out (trial 8).
            self.travel = min(self.travel + 0.05 * DT, self.open_rad + GRASP_SQUEEZE_RAD + 0.06)
            if self.contact_at is not None:
                self.travel = min(self.travel, self.contact_at + GRASP_PINCH_RAD)
            for k, (joint, i) in enumerate(zip(GRASP_ARM_JOINTS, self.arm_index)):
                if joint == "left_shoulder_roll_joint":
                    action[i] = self.close_from[k] - self.travel
                elif joint == "right_shoulder_roll_joint":
                    action[i] = self.close_from[k] + self.travel
        elif self.phase in ("lift", "lower", "raise"):
            # `lower` runs this with a negative dpitch. Raise with the shoulders, not wrist IK: IK re-solves the arm and lets the
            # pinch go (trial 8, grip lag 0.12 -> 0.00 and the box dropped). Shoulder
            # pitch also pitches the hands (FK: 11.5 deg for a 6 cm lift, and cube-11's box
            # tilted 8.4 deg with it while fixed in the palms); the same amount of wrist
            # pitch the other way cancels it to ~1 deg (FK at the cube-11/rect-11 grasps).
            pitch_lift(action, self.arm_index, self.lift_from, min(1.0, (el + DT) / self.lift_time) * self.dpitch)
        elif self.phase == "back_off":
            f = min(1.0, (el + DT) / 1.0)
            for k, i in enumerate(self.arm_index):
                action[i] = self.back_from[k] + f * (self.reach_refs[k] - self.back_from[k])
        elif self.phase == "shift" and self.motion is not None:
            q = row["root_quat"]
            action[:43] = self.motion.command(row["joint_pos"], row["root_pos"], [*q[1:], q[0]], action[:43], el + DT)
        action[43:46] = [0., 0., 0.]
        self.ref = list(action)
        return action

    motion = None

    def update(self, row):
        self.rows.append(row)
        now = row["time"]
        el = now - self.phase_start
        self.advice["phase"] = self.phase
        if row.get("box") is not None:
            self.advice["box_in_pelvis_m"] = [round(v, 3) for v in self.box_in_pelvis(row)]
        if self.phase in ("step_in", "step_in_settle", "lower", "shift"):
            # Encoder lag is only a clean contact sign during the slow IK shift; the
            # joint-space lower relies on the box-motion backstop.
            lag = self.arm_lag(row) if self.phase == "shift" else 0.0
            self.advice["arm_lag_rad"] = round(lag, 3)
            if lag > GRASP_SHIFT_CONTACT_LAG_RAD or self.disturbed(row, tilt_only=self.phase.startswith("step_in")):
                # A hand met the box before closing: go back to the contact-free reach
                # posture instead of pressing on (rect-10 pressed 46 N and tipped it).
                self.back_from = [self.ref[i] for i in self.arm_index]
                self.motion = None
                self.phase, self.phase_start = "back_off", now
                return None
        if self.phase == "square_up":
            err = wrap(self.square_target - yaw_of(row["root_quat"]))
            self.advice["box_yaw_offset_rad"] = round(err, 3)
            if abs(err) <= self.rate * self.rate / 1.0 + math.radians(2):
                self.phase, self.phase_start = "square_settle", now
            elif now - self.start >= abs(self.square_error) / 0.10 + 10.0:
                return self.finish("failed", "square_up_timeout")
        elif self.phase == "square_settle":
            if el >= 1.5 and abs(self.rate) < 1e-3:
                self.yaw0 = yaw_of(row["root_quat"])
                self.timeout += el
                self.phase, self.phase_start = "lineup", now
        elif self.phase == "lineup":
            if row.get("box") is None or row["box"]["age_s"] > 0.5:
                return self.finish("failed", "box_not_observed")
            error = float(np.linalg.norm(self.lineup_error(row)))
            self.advice["lineup_error_m"] = round(error, 3)
            if error <= 0.03:
                self.phase, self.phase_start = "lineup_settle", now
            elif now - self.start >= self.timeout:
                return self.finish("failed", "lineup_timeout")
        elif self.phase == "lineup_settle":
            if el >= 1.0 and float(np.linalg.norm(self.v)) < 1e-3:
                if row.get("box") is not None and row["box"]["age_s"] <= 0.5:
                    self.set_width(row)          # now measured square to the box
                    self.box_before = (np.array(row["box"]["pos"]), self.box_tilt(row))
                self.phase, self.phase_start = "reach", now
        elif self.phase == "reach":
            if el >= 3.5:
                self.start_lower(now, -GRASP_STEP_IN_RAISE_M / GRASP_LIFT_PER_RAD)
                self.phase = "raise"
        elif self.phase == "raise":
            if el >= self.lift_time + 0.5:
                self.reach_refs = [self.ref[i] for i in self.arm_index]
                self.step_wrist = self.wrist_mid_in_pelvis(row)
                self.step_box_z = self.box_in_pelvis(row)[2]
                self.stepped_in = True
                # HOMIE's last centimetres (deadband ~0.06 m/s) and lateral trims are slow.
                self.timeout = now - self.start + GRASP_REACH_STANDOFF_M / 0.05 + 30.0
                self.phase, self.phase_start = "step_in", now
        elif self.phase == "step_in":
            if row.get("box") is None or row["box"]["age_s"] > 0.5:
                return self.finish("failed", "box_not_observed")
            error = float(np.linalg.norm(self.lineup_error(row)))
            self.advice["step_in_error_m"] = round(error, 3)
            if error <= 0.03:
                self.phase, self.phase_start = "step_in_settle", now
            elif now - self.start >= self.timeout:
                return self.finish("failed", "lineup_timeout")
        elif self.phase == "step_in_settle":
            if el >= 1.0 and float(np.linalg.norm(self.v)) < 1e-3:
                if row.get("box") is not None and row["box"]["age_s"] <= 0.5:
                    self.box_before = (np.array(row["box"]["pos"]), self.box_before[1] if self.box_before else self.box_tilt(row))
                dp = self.lower_pitch(self.wrist_mid_in_pelvis(row)[2], self.box_in_pelvis(row)[2])
                self.advice["lower_pitch_rad"] = round(dp, 3)
                if dp > 0.03:
                    self.start_lower(now, dp)
                else:
                    self.start_shift_or_close(row, now)
        elif self.phase == "lower":
            if el >= self.lift_time + 0.5:
                self.start_shift_or_close(row, now)
        elif self.phase == "back_off":
            if el >= 1.5:
                return self.finish("failed", "hands_hit_box")
        elif self.phase == "shift":
            if el >= float(np.linalg.norm(self.shift)) / 0.01 + 1.0:
                self.shifts = getattr(self, "shifts", 1)
                self.plan_shift(row)          # one hop is capped at 6 cm; re-aim if still short
                if float(np.linalg.norm(self.shift)) > 0.02 and self.shifts < 3:
                    self.shifts += 1
                    self.motion = self.wrist_motion(row, self.world_shift(row, self.shift).tolist())
                    self.phase_start = now
                else:
                    self.start_close(now)
        elif self.phase == "close":
            left, right = self.palm_load(row)
            self.advice["palm_load_rad"] = [round(left, 3), round(right, 3)]
            if self.contact_at is None and min(left, right) >= GRASP_CONTACT_LAG_RAD:
                self.contact_at = self.travel
            done = self.contact_at is not None and self.travel >= self.contact_at + GRASP_PINCH_RAD - 1e-6
            if done or self.travel >= self.open_rad + GRASP_SQUEEZE_RAD + 0.06 - 1e-6:
                if self.contact_at is None:
                    return self.finish("failed", "no_contact_with_box")
                self.start_lift(row, now)
        elif self.phase == "lift":
            if el >= self.lift_time + 0.5:
                self.phase, self.phase_start = "verify", now
                self.rows.clear()
        elif self.phase == "verify":
            hold = camera_rate_hold(self.rows)
            self.advice.update(hold, box_height_above_source_m=row.get("clearance"))
            if row.get("box") is not None:
                self.advice["box_tilt_rad"] = round(self.box_tilt(row), 3)
            if hold["ready"] and (row.get("clearance") or 0) >= 0.03:
                return self.finish("completed", "raised_and_stable")
            # In the hands but low. Not `box_held`: that excludes a box sensed within 1.5 cm of the
            # table, which is exactly the low case (speed-cube: lifted 2.5 cm, tilted 6 deg,
            # sensed 1.0 cm, so it read "resting" and was never re-lifted).
            if el >= 3.0 and self.lifts < 3 and row["box_between_hands"]:
                return self.start_lift(row, now, rise_m=0.05)
            if el >= 6.0:
                tipped = self.advice.get("box_tilt_rad", 0.0) > 0.14
                return self.finish("failed", "box_tipped_in_hands" if tipped else "grasp_not_stable")


class Place(Skill):
    """Lower onto the observed surface, open the fingers, withdraw the wrists, stand up, verify.

    Support is inferred from vision only: box bottom within 1 cm of the observed
    top for 0.3 s, or the box stops descending while the pelvis keeps lowering
    (it rises > 1 cm relative to the pelvis). The fingers then move back to their
    start-of-episode (open) references over 1.5 s before the wrists withdraw
    outward, so the box is not dragged. Release is verified by the box staying
    still on the surface with the hands no longer around it.
    """

    @staticmethod
    def view_margins(row, pts):
        """Table extent beyond the box on the left/right/far sides, from the current view (pelvis frame).

        The map extent can be inflated by accumulated drift (sweep-a scene 04 overhung
        by 5 cm); the live view is not. None when that side is not in view.
        """
        w, x, y, z = row["root_quat"]
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        root = np.array(row["root_pos"])
        b = row["box"]["bounds"]
        corners = np.array([[b[i][0], b[j][1], b["min"][2]] for i in ("min", "max") for j in ("min", "max")])
        local = (corners - root) @ R
        (bx0, by0), (bx1, by1) = local[:, :2].min(axis=0), local[:, :2].max(axis=0)
        band_x = (pts[:, 0] >= bx0) & (pts[:, 0] <= bx1)
        band_y = (pts[:, 1] >= by0) & (pts[:, 1] <= by1)
        right = pts[band_x & (pts[:, 1] < by0), 1]
        left = pts[band_x & (pts[:, 1] > by1), 1]
        far = pts[band_y & (pts[:, 0] > bx1), 0]
        near = pts[band_y & (pts[:, 0] < bx0), 0]
        return dict(right=float(by0 - right.min()) if len(right) else None,
                    left=float(left.max() - by1) if len(left) else None,
                    far=float(far.max() - bx1) if len(far) else None,
                    near=float(bx0 - near.min()) if len(near) else None)

    def __init__(self, row, action, surface_id, wrist_motion, open_fingers, view=None, arm_index=()):
        super().__init__(row, action)
        self.arm_index = list(arm_index)
        s = row["surfaces"].get(surface_id)
        if s is None:
            raise ValueError("surface_not_observed")
        # "held" excludes a box whose bounds touch a table; a box grasped tilted can
        # already reach the top over the target, so only require it between the hands.
        if not row["box_between_hands"]:
            raise ValueError("box_not_held")
        bb, slab = row["box"]["bounds"], s["bounds"]
        margin = min(min(bb["min"][i] - slab["min"][i], slab["max"][i] - bb["max"][i]) for i in range(2))
        if margin < 0.04:
            raise ValueError("box_not_well_inside_surface")
        if view is not None:
            pts = view(surface_id, row["time"])
            if pts is not None and len(pts) >= 150:
                side = self.view_margins(row, pts)
                self.advice = dict(view_margins_m=side)
                # Left/right/near: no table points beyond the box on a side that the
                # camera covers means the table ends there (sweep-a 04 overhung sideways;
                # sweep-f 05 set the box 12 cm over the near edge, trusting the drifted
                # map). The far side is hidden by the box itself and not checked.
                lr = [side["left"], side["right"], side["near"]]
                if any(v is None or v < 0.02 for v in lr):
                    raise ValueError("box_not_well_inside_surface", dict(view_margins_m=side,
                        suggest="walk sideways toward the side with more table (walk(0, left_m=...)), "
                                "or forward if the near margin is short"))
        # The crouch lowers the box up to `crouch`; the arms (pitch_lift) cover up to 6 cm more
        # either way. raise_held_box allows 15 cm, which the crouch alone refused (sweep-j 03,
        # sweep-k2 s43: 7 refusals at 15-17 cm).
        crouch = min(0.14, action[46] - 0.60)
        if not -0.01 < s["clearance_m"] <= crouch + 0.06:
            raise ValueError("surface_outside_lowering_envelope",
                             dict(clearance_m=s["clearance_m"], lowest_start_m=-0.01, highest_start_m=crouch + 0.06))
        self.surface, self.wrist_motion = surface_id, wrist_motion
        self.open_fingers = list(open_fingers)
        self.height = action[46]
        self.phase, self.count, self.phase_start = "lower", 0, row["time"]
        self.box_rel0 = None
        self.lower_start = row["time"]
        # Arm move before the crouch, with the grip kept (pitch_lift, not wrist IK): up when the
        # destination is higher than the carried box (sweep scene 06), down when the box is
        # above the crouch range.
        self.arm_move = 0.0
        if s["clearance_m"] < 0.03:
            self.arm_move = 0.05 - s["clearance_m"]
        elif s["clearance_m"] > crouch:
            self.arm_move = -(s["clearance_m"] - crouch + 0.01)
        if self.arm_move:
            self.arm_from = [self.ref[i] for i in self.arm_index]
            self.arm_dpitch = self.arm_move / GRASP_LIFT_PER_RAD
            self.arm_time = max(1.5, abs(self.arm_dpitch) / 0.06)
            self.phase, self.phase_start = "arm_move", row["time"]

    def rel_z(self, row):
        return row["box"]["pos"][2] - row["root_pos"][2]

    @staticmethod
    def box_tilt(row):
        """Angle between the box's most vertical axis and world up (radians)."""
        if row.get("box") is None:
            return None
        w, x, y, z = row["box"]["quat"]
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        return float(math.acos(min(1.0, max(abs(R[2, i]) for i in range(3)))))

    @staticmethod
    def wrist_gap(row):
        """Smallest planar distance from either wrist to the box footprint (m); 0 if inside."""
        if row.get("box") is None:
            return None
        b = row["box"]["bounds"]
        gaps = []
        for k in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
            px, py = row["wrists_world"][k]["pos"][:2]
            dx = max(b["min"][0] - px, px - b["max"][0], 0.0)
            dy = max(b["min"][1] - py, py - b["max"][1], 0.0)
            gaps.append(math.hypot(dx, dy))
        return min(gaps)

    @staticmethod
    def wrist_height_in_pelvis(row):
        """Mean wrist height in the pelvis frame (encoders + FK only)."""
        w, x, y, z = row["root_quat"]
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        root = np.array(row["root_pos"])
        return float(np.mean([(R.T @ (np.array(row["wrists_world"][k]["pos"]) - root))[2]
                              for k in ("left_wrist_yaw_link", "right_wrist_yaw_link")]))

    def command(self, row):
        action = list(self.ref)
        el = row["time"] - self.phase_start
        if self.phase == "arm_move":
            pitch_lift(action, self.arm_index, self.arm_from, min(1.0, (el + DT) / self.arm_time) * self.arm_dpitch)
        elif self.phase == "lower":
            action[46] = max(0.60, self.height - min(0.14, 0.02 * (row["time"] - self.lower_start + DT)))
        elif self.phase == "open":
            f = min(1.0, (el + DT) / 1.5)
            action[FINGERS] = [a + f * (b - a) for a, b in zip(self.closed, self.open_fingers)]
            # Also let go of any arm squeeze: move the arm commands to where the arms are.
            # A shoulder-roll pinch (grasp_box, plus the grip keeper) otherwise stays on while
            # the wrists withdraw: 48-101 N and the box dragged over an edge (sweep-f 04/05).
            for i in self.arm_index:
                action[i] = self.arm_closed[i] + f * (self.arm_relaxed[i] - self.arm_closed[i])
        elif self.phase in ("withdraw", "withdraw_more"):
            q = row["root_quat"]
            action[:43] = self.motion.command(row["joint_pos"], row["root_pos"], [*q[1:], q[0]], action[:43], el + DT)
        elif self.phase == "rise":
            action[46] = min(self.height, self.low + 0.02 * (el + DT))
        self.ref = list(action)
        return action

    def update(self, row):
        now = row["time"]
        s = row["surfaces"].get(self.surface)
        if s is None or row["box"] is None:
            return self.finish("failed", "surface_or_box_unobserved")
        fresh = row["box"]["age_s"] < 1e-6
        if self.box_rel0 is None:
            self.box_rel0 = self.rel_z(row)
        self.advice = dict(box_clearance_m=s.get("clearance_m"), contained=s.get("contained"), phase=self.phase)
        el = now - self.phase_start
        if self.phase == "arm_move":
            if el >= self.arm_time + 0.5:
                self.box_rel0 = None                   # support cue baseline after the arm move
                self.lower_start = now
                self.phase, self.phase_start = "lower", now
        elif self.phase == "lower":
            if fresh:
                on = abs(s["clearance_m"]) <= 0.01 or self.rel_z(row) - self.box_rel0 > 0.01
                self.count = self.count + 1 if on else 0
            # Encoder-only cue: once the table carries the box, continued lowering
            # pushes the wrists up relative to the pelvis. Vision can drop out here
            # because the pressed box fails the tracker's fit (agent-04).
            # Before contact the wrists drift up slowly (~0.1 mm/step); once the table
            # carries the box they rise nearly as fast as the pelvis lowers (2 cm/s).
            # Cue: rise over the last 0.3 s >= 60% of the commanded descent (agent-04 replay).
            hist = getattr(self, "wrist_hist", [])
            hist.append((now, self.wrist_height_in_pelvis(row)))
            self.wrist_hist = hist = [h for h in hist if now - h[0] <= 0.3 + 1e-6]
            rise = hist[-1][1] - hist[0][1]
            rate_cue = now - self.start > 0.5 and hist[-1][0] - hist[0][0] >= 0.28 and rise >= 0.6 * 0.02 * 0.3
            self.wrist_count = getattr(self, "wrist_count", 0) + 1 if rate_cue else 0
            self.advice["wrist_rise_last_0p3s_m"] = rise
            if not hasattr(self, "_lower_started"):
                self._lower_started, self.lower_start = True, now
            if self.count >= 3 or self.wrist_count >= 3:
                self.advice["support_cue"] = "vision" if self.count >= 3 else "wrists_pushed_up"
                self.phase, self.phase_start = "supported_hold", now
            elif now - self.lower_start >= 9:
                return self.finish("failed", "lowering_timeout")
        elif self.phase == "supported_hold" and el >= 1.0:
            self.closed = list(self.ref[FINGERS])
            self.arm_closed = {i: self.ref[i] for i in self.arm_index}
            self.arm_relaxed = {i: row["joint_pos"][i] for i in self.arm_index}
            self.phase, self.phase_start = "open", now
        elif self.phase == "open" and el >= 1.7:
            self.motion = self.wrist_motion(row, None)   # outward 6 cm at 3 cm/s along the wrist axis
            self.phase, self.phase_start = "withdraw", now
        elif self.phase == "withdraw" and el >= 3.0:
            if row["box_between_hands"]:
                self.advice["note"] = "hands still around the box after withdrawal; not standing up"
                return self.finish("failed", "hands_not_clear_after_withdrawal")
            self.low = self.ref[46]
            self.phase, self.phase_start = "rise", now
        elif self.phase == "rise" and el >= (self.height - self.low) / 0.02 + 0.5:
            self.phase, self.phase_start = "settle", now
            self.rows.clear()
        elif self.phase == "settle":
            if fresh:
                self.rows.append(row)
            if el >= 2.5:
                # Judge stillness over the last second only; the box may settle
                # a few millimetres right after release (smoke-15 passed truth).
                recent = [r for r in self.rows if now - r["time"] <= 1.0 + 1e-6]
                pairs = list(zip(recent, recent[1:]))
                still = len(recent) >= 5 and max(math.dist(a["box"]["pos"], b["box"]["pos"]) / (b["time"] - a["time"])
                                                 for a, b in pairs) < 0.03
                on = abs(s["clearance_m"]) <= 0.015 and s.get("contained")
                gap = self.wrist_gap(row)
                tilt = self.box_tilt(row)
                # A tilted box can sit wedged between the hands without looking
                # "between" them (sweep-b scene 01): require both wrists >= 8 cm
                # outside the box's footprint and the box level within 10 deg.
                clear = gap is not None and gap >= 0.08   # fingers reach ~0.1 m past the wrist; successes 0.09-0.14, wedged failure 0.06
                level = tilt is not None and tilt <= math.radians(10)
                self.advice.update(box_still=still, on_surface=on, hands_clear=clear, wrist_gap_m=gap,
                                   box_tilt_rad=tilt)
                if still and on and clear and level:
                    return self.finish("completed", "released_on_surface")
                # Only stillness missing: keep judging the rolling last second, up to 6 s. One
                # look at 2.5 s reported failure for 9 placements the scorer counted as
                # successes (sweep-f..k; box estimate jitter while the body settles).
                if on and clear and level and el < 6.0:
                    return None
                if not clear and not getattr(self, "second_withdraw", False):
                    self.second_withdraw = True
                    self.motion = self.wrist_motion(row, None)
                    self.phase, self.phase_start = "withdraw_more", now
                    return None
                return self.finish("failed", "release_not_verified")
        elif self.phase == "withdraw_more":
            if now - self.phase_start >= 3.0:
                self.phase, self.phase_start = "settle", now
                self.rows.clear()


# ---------------------------------------------------------------- owner
class ControlV2:
    """Same interface ArenaSession expects from BoxControl (start/command/update/cancel)."""

    def __init__(self, initial_action, acquire, wrist_motion, robot_bounds, view=None, joint_names=()):
        self.joint_names = list(joint_names)
        self.view = view   # callable(surface_id, now) -> fresh pelvis-frame table points or None
        self.last_action = f32(initial_action)
        self.last_action[43:46] = [0., 0., 0.]
        self.open_fingers = list(self.last_action[FINGERS])   # measured at episode start (open hands)
        self.acquire, self.wrist_motion, self.robot_bounds = acquire, wrist_motion, robot_bounds
        self.skill = None
        self.method = None
        self.phase = "idle"
        self.result = None
        self.terminal_reason = None
        self.last_row = None
        self.events = []   # advisory/interlock log
        self.grip = None   # after a scripted grasp: shoulder-roll targets kept while carrying

    def _reply(self, status, reason, row, advice=None):
        return dict(status=status, reason=reason, advice=advice or {}, time=row["time"] if row else None)

    def start(self, method, args, row):
        if self.terminal_reason:
            return self._reply("rejected", "episode_failed", row)
        if self.skill:
            return self._reply("rejected", "operation_active", row)
        if method not in TOOLS or not isinstance(args, dict) or set(args) - set(TOOLS[method]):
            return self._reply("rejected", "invalid_request", row)
        try:
            if method == "wait":
                skill = Wait(row, self.last_action, float(args.get("duration", 1.0)))
            elif method == "hold_box":
                skill = Hold(row, self.last_action, float(args.get("duration", 1.0)))
            elif method == "pickup_box":
                if args.get("object_id", "brown_box") != "brown_box":
                    raise ValueError("unsupported_object")
                skill = Pickup(row, self.last_action, self.acquire, self.wrist_motion)
            elif method == "grasp_box":
                skill = Grasp(row, self.last_action, self.wrist_motion, self.joint_names,
                              float(args.get("lift_m", 0.06)))
            elif method == "raise_held_box":
                skill = Raise(row, self.last_action, float(args.get("clearance_m", 0.08)),
                              [self.joint_names.index(n) for n in GRASP_ARM_JOINTS])
            elif method == "walk":
                skill = Walk(row, self.last_action, float(args.get("forward_m", 0.)),
                             float(args.get("left_m", 0.)), float(args.get("yaw_rad", 0.)))
            elif method == "turn":
                skill = Turn(row, self.last_action, float(args["yaw_rad"]))
            elif method == "approach_surface":
                skill = Approach(row, self.last_action, args.get("surface_id", "destination"),
                                 float(args.get("standoff_m", 0.24)), view=self.view, robot_bounds=self.robot_bounds)
            elif method == "place_box":
                skill = Place(row, self.last_action, args.get("surface_id", "destination"), self.wrist_motion,
                              self.open_fingers, view=self.view,
                              arm_index=[self.joint_names.index(n) for n in GRASP_ARM_JOINTS if n in self.joint_names])
            else:
                return self._reply("rejected", "invalid_request", row)
        except (ValueError, KeyError, TypeError) as error:
            advice = error.args[1] if len(error.args) > 1 and isinstance(error.args[1], dict) else None
            return self._reply("rejected", str(error.args[0] if error.args else error).split(":")[0][:80], row, advice)
        if method in ("place_box", "pickup_box", "grasp_box"):
            self.grip = None                       # releasing, or a new pickup
        elif self.grip and method in GRIP_KEEP_METHODS:
            self.grip["roll"] = [self.last_action[i] for i in self.grip["index"]]   # resync after e.g. a raise
        self.skill, self.method, self.result = skill, method, None
        self.phase = skill.phase
        return self._reply("running", "accepted", row)

    def _finish(self, status, reason, row):
        if isinstance(self.skill, Grasp) and status == "completed" and row is not None:
            index = [self.joint_names.index(n) for n in ("left_shoulder_roll_joint", "right_shoulder_roll_joint")]
            self.grip = dict(index=index, roll=[self.skill.ref[i] for i in index], extra=0.0,
                             floor=GRIP_KEEP_FRACTION * min(self.skill.palm_load(row)))
        advice = dict(self.skill.advice) if self.skill else {}
        if getattr(self.skill, "nearest_corner", None) is not None:
            advice["nearest_table_corner_m"] = self.skill.nearest_corner
        clearance = carried_clearance(row, self.robot_bounds) if row else None
        if clearance is not None:
            advice["carried_table_clearance_m"] = clearance
        if self.grip:
            advice["grip_retightened_rad"] = round(self.grip["extra"], 3)
        self.result = self._reply(status, reason, row, advice)
        self.skill, self.method, self.phase = None, None, "idle"
        self.last_action[43:46] = [0., 0., 0.]
        return self.result

    def cancel(self, reason, row):
        return self._finish("cancelled", reason, row)

    def command(self, row):
        self.last_row = row
        if self.skill is None or self.terminal_reason:
            action = list(self.last_action)
            action[43:46] = [0., 0., 0.]
        else:
            action = self.skill.command(row)
            self.phase = self.skill.phase
        if self.grip and (self.skill is None or self.method in GRIP_KEEP_METHODS):
            self.keep_grip(action, row)
        self.last_action = f32(action)
        return list(self.last_action)

    def keep_grip(self, action, row):
        """Hold the grasp's shoulder-roll targets and close further if the sensed palm load
        (commanded minus measured roll, as in Grasp.palm_load) falls below its pickup level."""
        li, ri = self.grip["index"]
        left = row["joint_pos"][li] - self.grip["roll"][0]
        right = self.grip["roll"][1] - row["joint_pos"][ri]
        if min(left, right) < self.grip["floor"] and self.grip["extra"] < GRIP_KEEP_MAX_RAD:
            step = GRIP_KEEP_RATE * DT
            self.grip["roll"] = [self.grip["roll"][0] - step, self.grip["roll"][1] + step]
            self.grip["extra"] += step
        action[li], action[ri] = self.grip["roll"]

    def update(self, row):
        self.last_row = row
        # Hard stop 1: body tilt (IMU).
        if row["tilt"] > math.radians(15):
            self.terminal_reason = "body_tilt_limit"
            if self.skill:
                return self._finish("failed", "body_tilt_limit", row)
            return self.result
        if self.skill is None:
            return self.result
        moving = isinstance(self.skill, (Walk, Turn, Approach))
        # Hard stop 2: carried box left the hands during locomotion.
        if moving and self.skill.carrying:
            recent = [r for r in list(self.skill.rows)[-15:] if r.get("box")]
            if len(recent) >= 15 and not any(r["box_between_hands"] for r in recent):
                self.events.append(dict(time=row["time"], interlock="box_retention_lost"))
                return self._finish("failed", "box_retention_lost", row)
        # Hard stop 3: predicted overlap with an observed table while walking/turning.
        if moving:
            # An approach deliberately brings the box and forearms over its target
            # table (the audit showed this check false-alarms there); other tables
            # are still checked.
            target = self.skill.surface if isinstance(self.skill, Approach) else None
            clearance = carried_clearance(dict(row, surfaces=frozen_map(self.skill, row, "surfaces")),
                                          self.robot_bounds, target=target)
            # Only when the motion is reducing clearance: backing away from a table
            # the box is already over must stay possible (sweep-a 05, sweep-c deadlocks).
            if clearance is not None and clearance < 0.02 and worsened(self.skill, "clearance", clearance):
                self.events.append(dict(time=row["time"], interlock="predicted_table_contact", clearance_m=clearance))
                return self._finish("failed", "predicted_table_contact", row)
        # Hard stop 5: during the learned grasp, GR00T may walk. Keep it from wandering
        # (> 0.35 m from where pickup started) or reaching another table
        # (sweep-b scene 02: 524 N against the destination while re-acquiring).
        if isinstance(self.skill, Pickup) and self.skill.phase == "acquire":
            start = getattr(self.skill, "start_xy", None)
            if start is None:
                self.skill.start_xy = start = list(row["root_pos"][:2])
            if math.dist(start, row["root_pos"][:2]) > 0.35:
                self.events.append(dict(time=row["time"], interlock="acquisition_wandered"))
                return self._finish("failed", "acquisition_wandered", row)
            others = [n for n in (row.get("surfaces") or {}) if n != "source"]
            for name in others:
                c = carried_clearance(dict(row, surfaces={name: row["surfaces"][name]}, box_held=False), self.robot_bounds)
                if c is not None and c < 0.10:
                    self.events.append(dict(time=row["time"], interlock="acquisition_near_other_table", table=name))
                    return self._finish("failed", "acquisition_near_other_table", row)
        # Advice only: pelvis/knees near a mapped table corner (legs) while walking/turning.
        # It used to be a hard stop, but the estimated distance wobbles 1-2 cm per step and
        # drifts the wrong way over short walks, so it refused walks moving away from the
        # leg (sweep-c/d/e scene 03). A real bump fails the episode in scoring instead.
        if isinstance(self.skill, (Walk, Turn)):
            parts = [row["body_poses"][n]["pos"][:2] for n in ("pelvis", "left_knee_link", "right_knee_link")
                     if n in row["body_poses"]]
            corners = [(a, b) for t in (row.get("tables") or {}).values() if t is not None
                       for a in (t["min_xy"][0], t["max_xy"][0]) for b in (t["min_xy"][1], t["max_xy"][1])]
            if corners and parts:
                corner = min(math.dist(c, q) for c in corners for q in parts)
                self.skill.nearest_corner = round(corner, 3)
                if corner < CORNER_KEEPOUT_M and not getattr(self.skill, "corner_warned", False):
                    self.skill.corner_warned = True
                    self.events.append(dict(time=row["time"], advisory="near_table_corner", distance_m=corner))
        out = self.skill.update(row)
        if self.skill:
            self.phase = self.skill.phase
        if out:
            return self._finish(out[0], out[1], row)
        return None


# ---------------------------------------------------------------- agent-facing view
def public_observation(row, control):
    """What generated programs see: sensor estimates in odom, no joint dumps or truth.

    ``robot`` pose is leg odometry (drifts ~10-20% of distance walked); tables are
    observed extents with an uncertainty that grows with walking since last seen.
    """
    if row is None:
        return dict(observation_mode="sensor_state_v2", time=0.0, status="initializing")
    box = row.get("box")
    tables = {}
    for name, t in (row.get("tables") or {}).items():
        tables[name] = None if t is None else dict(
            observed=True, top_z_m=round(t["top_z"], 4), min_xy_m=[round(v, 3) for v in t["min_xy"]],
            max_xy_m=[round(v, 3) for v in t["max_xy"]], last_seen_s=t["last_seen_s"],
            position_uncertainty_m=round(t["position_uncertainty_m"], 3))
    surfaces = {name: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
                       if k in ("clearance_m", "contained", "position_uncertainty_m")}
                for name, s in (row.get("surfaces") or {}).items()}
    return dict(
        observation_mode="sensor_state_v2", frame="odom (origin = start pose; x = initial forward, z up)",
        time=row["time"], step=row["step"],
        robot=dict(position_m=[round(v, 3) for v in row["root_pos"]], yaw_rad=round(yaw_of(row["root_quat"]), 4),
                   tilt_rad=round(row["tilt"], 4), walked_m=round(row["walked_m"], 3)),
        box=None if box is None else dict(position_m=[round(v, 4) for v in box["pos"]], size_m=[round(v, 4) for v in box["size_m"]],
                                          age_s=round(box["age_s"], 3), held=row["box_held"],
                                          between_hands=row["box_between_hands"],
                                          height_above_source_m=None if row.get("clearance") is None else round(row["clearance"], 4)),
        tables=tables, surfaces=surfaces, controller_phase=control.phase if control else "idle")
