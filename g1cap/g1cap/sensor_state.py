"""One sensor-only state estimate for the box-transfer controllers.

Inputs are permitted measurements only: body encoders/IMU packets, Dex3 finger
positions, head RGB-D frames with known calibration, and accepted box poses from
the existing RGB-D cuboid tracker. No simulator pose, contact force, mass or
table geometry enters this module.

Frames and units
- ``odom``: fixed at episode start. Origin = pelvis at the first packet, z up,
  x = initial pelvis forward. Metres, radians, WXYZ quaternions for
  compatibility with the older controllers' observation dictionaries.
- ``pelvis``: G1 pelvis frame (x forward, y left, z up), from encoders.
- camera optical frame: x right, y down, z forward.

Assumptions (simulation development values, not hardware calibration)
- Yaw/roll/pitch come from integrating the pelvis gyro. Offline replays show
  about 1 degree drift over 30 s.
- Position is leg-kinematic odometry. The lower foot is taken as the stance foot
  and is assumed not to slip. Offline replays show about 10-20% error in the
  distance walked. This is enough for coarse navigation only; final approach
  and placement use the current view of the table.
- Tables are static. Each is represented by its observed top-surface points,
  accumulated in odom. Uncertainty grows with the distance and rotation walked
  since the points were seen. Only observed extent is kept, never an invented
  full footprint.
"""
import math

import numpy as np
import pinocchio as pin

SOLE_POINTS = np.array([[-.05, .025, -.03], [-.05, -.025, -.03], [.12, .03, -.03], [.12, -.03, -.03]])
TABLE_CELL_M = 0.02
TOP_THICKNESS_M = 0.04
ODOMETRY_DRIFT_PER_M = 0.20      # position uncertainty added per metre walked
ODOMETRY_DRIFT_PER_RAD = 0.03    # position uncertainty added per radian turned (lever arm ~1 m x 1.7 deg)
TILT_GAIN = 0.005                # accelerometer roll/pitch correction per 20 ms step


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.]])


def rotation_step(R, omega_dt):
    angle = float(np.linalg.norm(omega_dt))
    if angle < 1e-12:
        return R
    K = skew(omega_dt / angle)
    return R @ (np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K)


def matrix_to_wxyz(R):
    q = pin.Quaternion(R)
    return [float(q.w), float(q.x), float(q.y), float(q.z)]


class Kinematics:
    """Encoder forward kinematics with one reused pinocchio Data (fast)."""

    def __init__(self, urdf):
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        self.pelvis = self.model.getFrameId("pelvis")
        self._index = {}

    def update(self, names, values):
        for name, value in zip(names, values):
            idx = self._index.get(name)
            if idx is None:
                if not self.model.existJointName(name):
                    continue
                idx = self._index[name] = self.model.joints[self.model.getJointId(name)].idx_q
            self.q[idx] = value
        pin.framesForwardKinematics(self.model, self.data, self.q)
        self._base = self.data.oMf[self.pelvis].inverse()

    def frame(self, name):
        """T_pelvis_frame (4x4) at the last update."""
        return (self._base * self.data.oMf[self.model.getFrameId(name)]).homogeneous

    def link_boxes(self, bounds):
        """[(T_pelvis_link, min, max)] for every link with collision bounds."""
        out = []
        for link, shapes in bounds.items():
            if not self.model.existFrame(link):
                continue
            T = self.frame(link)
            out.extend((T, np.asarray(b["min"]) - 0.01, np.asarray(b["max"]) + 0.01) for b in shapes)
        return out

    def camera(self, calibration):
        mount = np.eye(4)
        xyzw = np.asarray(calibration["offset_quaternion_xyzw"], float)
        mount[:3, :3] = pin.Quaternion(xyzw / np.linalg.norm(xyzw)).matrix()
        mount[:3, 3] = calibration["offset_position_m"]
        return self.frame(calibration["parent_frame"]) @ mount


class LegOdometry:
    """Gyro orientation + stance-foot kinematic position, in odom."""

    def __init__(self):
        self.R = np.eye(3)
        self.p = np.zeros(3)
        self.stance = None
        self.anchor = None
        self.time_s = None
        self.gyro = None
        self.walked_m = 0.0
        self.turned_rad = 0.0
        self.yaw_history = []   # (time, integrated yaw) for the last 0.5 s

    def update(self, time_s, gyro, feet_in_pelvis, specific_force=None):
        gyro = np.asarray(gyro, float)
        if self.time_s is not None:
            dt = time_s - self.time_s
            if not 0 < dt <= 0.1 + 1e-9:
                raise ValueError("odometry_time_gap")
            step = (gyro + self.gyro) * 0.5 * dt
            self.R = rotation_step(self.R, step)
            self.turned_rad += abs(float(step[2]))
        self.time_s, self.gyro = time_s, gyro
        if specific_force is not None:
            # Slow complementary correction of roll/pitch toward the measured
            # specific force (walking accelerations average out over ~10 s).
            # Gain 0.005/step chosen on 5 recorded episodes: tilt error 4.6 -> 0.3 deg.
            f = np.asarray(specific_force, float)
            if abs(np.linalg.norm(f) - 9.81) < 1.5:
                up_est = self.R.T @ np.array([0.0, 0.0, 1.0])
                up_meas = f / np.linalg.norm(f)
                axis = np.cross(up_meas, up_est)
                sn = float(np.linalg.norm(axis))
                if sn > 1e-9:
                    self.R = rotation_step(self.R, axis / sn * TILT_GAIN * math.atan2(sn, float(up_est @ up_meas)))
        self.yaw_history.append((time_s, self.yaw_unwrapped()))
        self.yaw_history = [h for h in self.yaw_history if time_s - h[0] <= 0.5 + 1e-9]
        lowest = {side: float(min((self.R @ (T[:3, :3] @ SOLE_POINTS.T + T[:3, 3:4]))[2]))
                  for side, T in feet_in_pelvis.items()}
        lower = min(lowest, key=lowest.get)
        if lower != self.stance:
            self.stance, self.anchor = lower, None
        foot = self.R @ feet_in_pelvis[self.stance][:3, 3]
        if self.anchor is None:
            self.anchor = self.p + foot
        previous = self.p
        self.p = self.anchor - foot
        self.walked_m += float(np.linalg.norm((self.p - previous)[:2]))
        self.floor_z = float(self.p[2] + lowest[self.stance])

    def yaw_unwrapped(self):
        if not self.yaw_history:
            return self.yaw()
        last = self.yaw_history[-1][1]
        return last + math.atan2(math.sin(self.yaw() - last), math.cos(self.yaw() - last))

    def recent_yaw_rate(self):
        if len(self.yaw_history) < 2:
            return 0.0
        (t0, y0), (t1, y1) = self.yaw_history[0], self.yaw_history[-1]
        return (y1 - y0) / max(t1 - t0, 1e-6)

    def pose(self):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.p
        return T

    def yaw(self):
        return math.atan2(self.R[1, 0], self.R[0, 0])

    def tilt(self):
        return math.acos(max(-1.0, min(1.0, float(self.R[2, 2]))))

    def drift_since(self, walked_m, turned_rad):
        return (ODOMETRY_DRIFT_PER_M * max(0.0, self.walked_m - walked_m)
                + ODOMETRY_DRIFT_PER_RAD * max(0.0, self.turned_rad - turned_rad))


class TableEstimate:
    """Observed top points of one static table, accumulated on a 2 cm odom grid."""

    def __init__(self, name):
        self.name = name
        self.cells = {}          # (i, j) -> [count, sum_z]
        self.last_seen_s = None
        self.seen_walked_m = 0.0
        self.seen_turned_rad = 0.0
        self._extent = None      # (cells version, min_count) -> region extent; recomputed after add()
        self.version = 0

    def add(self, points_odom, time_s, odometry):
        """Cells store height above the floor estimated at the same instant."""
        ij = np.floor(points_odom[:, :2] / TABLE_CELL_M).astype(int)
        for (i, j), z in zip(map(tuple, ij), points_odom[:, 2] - odometry.floor_z):
            cell = self.cells.setdefault((i, j), [0, 0.0])
            cell[0] += 1
            cell[1] += float(z)
        self.last_seen_s = time_s
        self.seen_walked_m, self.seen_turned_rad = odometry.walked_m, odometry.turned_rad
        self.version += 1

    def largest_region(self, min_count):
        """Largest 8-connected group of occupied cells; drops isolated spurious cells."""
        occupied = {k for k, v in self.cells.items() if v[0] >= min_count}
        best, seen = [], set()
        for start in occupied:
            if start in seen:
                continue
            group, stack = [], [start]
            seen.add(start)
            while stack:
                i, j = stack.pop()
                group.append((i, j))
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        n = (i + di, j + dj)
                        if n in occupied and n not in seen:
                            seen.add(n)
                            stack.append(n)
            if len(group) > len(best):
                best = group
        return best

    def height(self):
        """Median stored height above floor, or None before any observation."""
        if not self.cells:
            return None
        return float(np.median([v[1] / v[0] for v in self.cells.values()]))

    def extent(self, min_count=2):
        """(lo_xy, hi_xy, height, cells) of the largest region, or None. Cached until the
        next add(): the region search ran every 20 ms step and took ~9% of wall time."""
        if self._extent is None or self._extent[0] != (self.version, min_count):
            region = self.largest_region(min_count)
            value = None
            if len(region) >= 20:
                occupied = [(k, self.cells[k]) for k in region]
                xy = (np.array([k for k, _ in occupied], float) + 0.5) * TABLE_CELL_M
                value = (xy.min(axis=0) - TABLE_CELL_M / 2, xy.max(axis=0) + TABLE_CELL_M / 2,
                         float(np.median([v[1] / v[0] for _, v in occupied])), len(occupied))
            self._extent = ((self.version, min_count), value)
        return self._extent[1]

    def summary(self, odometry, min_count=2):
        ext = self.extent(min_count)
        if ext is None:
            return None
        lo, hi, height, count = ext
        top = odometry.floor_z + height
        uncertainty = odometry.drift_since(self.seen_walked_m, self.seen_turned_rad)
        return dict(top_z=top, height_above_floor_m=height, min_xy=lo.tolist(), max_xy=hi.tolist(), observed_cells=count,
                    last_seen_s=self.last_seen_s, position_uncertainty_m=uncertainty,
                    bounds=dict(min=[float(lo[0]), float(lo[1]), top - TOP_THICKNESS_M],
                                max=[float(hi[0]), float(hi[1]), top]))

    def align(self, points_odom, search_m=0.10, step_m=0.01, min_cells=150, min_points=300):
        """Planar shift (dx, dy) that best places current top points onto the stored cells.

        Score = points landing on stored cells minus points landing outside them,
        so partial views still lock onto the visible edges. Returns None when the
        stored map or the current view is too small, or the best shift is not
        clearly better than no shift.
        """
        stored = [k for k, v in self.cells.items() if v[0] >= 2]
        if len(stored) < min_cells or len(points_odom) < min_points:
            return None
        ij = np.array(stored)
        lo = ij.min(axis=0) - 20
        grid = np.zeros(tuple(ij.max(axis=0) - lo + 41), bool)
        grid[tuple((ij - lo).T)] = True
        xy = points_odom[:, :2]
        scores = {}
        steps = np.round(np.arange(-search_m, search_m + 1e-9, step_m), 4)
        for dx in steps:
            for dy in steps:
                k = np.floor((xy + (dx, dy)) / TABLE_CELL_M).astype(int) - lo
                inside = np.all((k >= 0) & (k < grid.shape), axis=1)
                hit = np.zeros(len(k), bool)
                hit[inside] = grid[k[inside, 0], k[inside, 1]]
                scores[(float(dx), float(dy))] = int(hit.sum()) - int((~hit).sum())
        best_score = max(scores.values())
        # Directions the view does not constrain (sliding along a single edge)
        # give ties; take the smallest shift among near-best scores.
        near = [d for d, v in scores.items() if v >= best_score - 0.005 * len(xy)]
        best = min(near, key=lambda d: d[0] ** 2 + d[1] ** 2)
        if best_score - scores[(0.0, 0.0)] < 0.01 * len(xy):
            return (0.0, 0.0)
        return best

    def clear(self):
        """Forget accumulated cells, e.g. when a fresh close view should replace drifted memory."""
        self.cells.clear()
        self.version += 1


def outside_robot(points_odom, T_odom_pelvis, link_boxes):
    """Mask of points not inside any (1 cm inflated) robot link box, from encoders."""
    keep = np.ones(len(points_odom), bool)
    if not len(points_odom):
        return keep
    local = (points_odom - T_odom_pelvis[:3, 3]) @ T_odom_pelvis[:3, :3]
    for T, lo, hi in link_boxes:
        q = (local - T[:3, 3]) @ T[:3, :3]
        keep &= ~np.all((q >= lo) & (q <= hi), axis=1)
    return keep


def table_points(depth, rgb, intrinsic, T_odom_camera, floor_z, box_bounds=None, stride=4, robot=None):
    """Classify head RGB-D pixels into candidate table-top points (odom frame).

    Returns {'destination': Nx3, 'source': Nx3}. Destination is the declared green
    table. Other non-robot, non-box points on a common horizontal level at least
    0.4 m above the floor are the source table. Colour thresholds are fixture
    assumptions (green destination, white robot, brown box).
    """
    d = depth[::stride, ::stride]
    c = rgb[::stride, ::stride, :3].astype(float)
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    valid = np.isfinite(d) & (d > 0.15) & (d < 4.0)
    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
    pts = np.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], axis=-1)[valid]
    col = c[valid]
    world = pts @ T_odom_camera[:3, :3].T + T_odom_camera[:3, 3]
    height = world[:, 2] - floor_z
    elevated = (height > 0.4) & (height < 1.2)
    bright = col.min(axis=1) > 200                                     # white robot shell
    green = (col[:, 1] > col[:, 0] + 25) & (col[:, 1] > col[:, 2] + 10)
    brown = (col[:, 0] > col[:, 2] + 40) & (col[:, 0] > col[:, 1] + 10)
    keep = elevated & ~bright & ~brown
    if robot is not None:
        T_odom_pelvis, boxes = robot
        idx = np.flatnonzero(keep)
        keep[idx] = outside_robot(world[idx], T_odom_pelvis, boxes)
    if box_bounds is not None:
        lo, hi = np.asarray(box_bounds["min"]) - 0.05, np.asarray(box_bounds["max"]) + 0.05
        keep &= ~np.all((world >= lo) & (world <= hi), axis=1)
    # Grey rims/legs next to green pixels belong to the destination, not the source.
    near_green = np.zeros(len(world), bool)
    if np.any(keep & green):
        cells = {tuple(c) for c in np.floor(world[keep & green, :2] / 0.05).astype(int)}
        ij = np.floor(world[:, :2] / 0.05).astype(int)
        near_green = np.array([any((i + a, j + b) in cells for a in (-3, -2, -1, 0, 1, 2, 3) for b in (-3, -2, -1, 0, 1, 2, 3))
                               for i, j in ij]) if len(cells) else near_green
    result = {}
    for name, mask in (("destination", keep & green), ("source", keep & ~green & ~near_green)):
        p = world[mask]
        if len(p) < 150:
            continue
        # Keep only the dominant horizontal level (the top surface).
        hist, edges = np.histogram(p[:, 2], bins=np.arange(p[:, 2].min(), p[:, 2].max() + 0.02, 0.01))
        if not len(hist):
            continue
        k = int(hist.argmax())
        level = 0.5 * (edges[k] + edges[k + 1])
        top = p[np.abs(p[:, 2] - level) < 0.012]
        if len(top) >= 150:
            result[name] = top
    return result


class SensorState:
    """Owner of the estimate; call update_proprio each 20 ms, update_camera per frame."""

    def __init__(self, urdf, bounds_file=None, box_size_hint_m=None):
        import json
        from pathlib import Path
        self.kin = Kinematics(urdf)
        bounds_file = bounds_file or Path(urdf).with_name("arena_g1_rev1_0_bounds.json")
        self.link_bounds = json.loads(Path(bounds_file).read_text())
        self.odom = LegOdometry()
        self.tables = {"source": TableEstimate("source"), "destination": TableEstimate("destination")}
        self.box = None           # dict(pos, R, size, time_s, epoch)
        self.packet = None
        self.hands = None
        self.box_size_hint_m = box_size_hint_m
        self.landmark_correction = True
        self.last_correction = None
        self.views = {}

    # ---- 50 Hz proprioception ----
    def update_proprio(self, packet, hands=None):
        self.kin.update(packet["joint_names"], packet["q_rad"])
        if hands is not None:
            self.kin.update(hands["joint_names"], hands["q_rad"])
        feet = {s: self.kin.frame(s + "_ankle_roll_link") for s in ("left", "right")}
        self.odom.update(packet["time_s"], packet["gyro_rad_s"], feet, packet.get("specific_force_m_s2"))
        self.packet, self.hands = packet, hands

    # ---- camera rate ----
    def update_camera(self, camera, box_estimate=None, map_tables=True):
        if self.packet is None or abs(camera["time_s"] - self.packet["time_s"]) > 1e-6:
            raise ValueError("camera_encoder_mismatch")
        T_odom_camera = self.odom.pose() @ self.kin.camera(camera["calibration"])
        if box_estimate is not None and box_estimate.get("status") == "accepted":
            R = T_odom_camera[:3, :3] @ np.asarray(box_estimate["axes_camera"], float)
            pos = T_odom_camera[:3, :3] @ np.asarray(box_estimate["center_camera_m"], float) + T_odom_camera[:3, 3]
            B = np.eye(4)
            B[:3, :3], B[:3, 3] = R, pos
            self.box = dict(pos=pos, R=R, size=list(box_estimate["dimensions_m"]), time_s=camera["time_s"],
                            epoch=box_estimate.get("track_epoch"),
                            pelvis_T_box=np.linalg.inv(self.odom.pose()) @ B)  # for carrying between frames
        if map_tables:
            found = table_points(camera["depth_m"], camera["rgb"], camera["calibration"]["intrinsic"],
                                 T_odom_camera, self.odom.floor_z, self.box_bounds(),
                                 robot=(self.odom.pose(), self.kin.link_boxes(self.link_bounds)))
            self.last_correction = None
            self.views = {}
            for name, pts in found.items():
                known = self.tables[name].height()
                level = float(np.median(pts[:, 2])) - self.odom.floor_z
                if known is not None and abs(level - known) > 0.06:
                    continue  # a different horizontal surface (e.g. box top), not this table
                # Net rotation over the last 0.5 s (gait sway averages out).
                turning = abs(self.odom.recent_yaw_rate()) > 0.05
                if self.landmark_correction and not turning:
                    # Translation-only alignment would misread a yaw error as a shift
                    # while rotating, so corrections are skipped during turns.
                    # Static tables are landmarks: if the current view sits shifted
                    # from the stored map, the robot (not the table) moved in the
                    # estimate. Move odometry by at most 2 cm per frame toward it.
                    shift = self.tables[name].align(pts)
                    if shift is not None and shift != (0.0, 0.0):
                        d = np.clip(np.array(shift), -0.02, 0.02)
                        self.odom.p[:2] += d
                        self.odom.anchor[:2] += d
                        if self.box is not None:
                            self.box["pos"] = self.box["pos"] + np.array([d[0], d[1], 0.0])
                        pts = pts + np.array([d[0], d[1], 0.0])
                        self.last_correction = dict(table=name, shift_m=[float(d[0]), float(d[1])])
                self.tables[name].add(pts, camera["time_s"], self.odom)
                # Current-frame view in the pelvis frame: relative geometry that
                # does not depend on odometry (used for close approach).
                T_inv = np.linalg.inv(self.odom.pose())
                local = pts @ T_inv[:3, :3].T + T_inv[:3, 3]
                self.views[name] = dict(time_s=camera["time_s"], points_pelvis=local)
        return T_odom_camera

    # ---- derived quantities ----
    def box_bounds(self):
        if self.box is None:
            return None
        half = 0.5 * np.asarray(self.box["size"])
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
        world = corners @ self.box["R"].T + self.box["pos"]
        return dict(min=world.min(axis=0).tolist(), max=world.max(axis=0).tolist())

    def wrists(self):
        T = self.odom.pose()
        out = {}
        for side in ("left", "right"):
            W = T @ self.kin.frame(side + "_wrist_yaw_link")
            q = pin.Quaternion(W[:3, :3])
            out[side + "_wrist_yaw_link"] = dict(pos=W[:3, 3].tolist(), xyzw=[q.x, q.y, q.z, q.w])
        return out

    def table_summaries(self):
        return {name: t.summary(self.odom) for name, t in self.tables.items()}

    def observation(self, now_s):
        """Controller observation in odom, with the old dictionary keys where meaningful.

        Fields that no permitted sensor measures (contact forces, support force)
        are NOT fabricated: they are absent, and the v2 controllers use the
        explicit sensor substitutes below instead.
        """
        T = self.odom.pose()
        box = self.box
        bounds = self.box_bounds()
        tables = self.table_summaries()
        source = tables.get("source")
        obs = dict(time=now_s, state_age_s=0.0, frame="odom",
                   root_pos=T[:3, 3].tolist(), root_quat=matrix_to_wxyz(T[:3, :3]),
                   yaw_rad=self.odom.yaw(), tilt=self.odom.tilt(), floor_z=self.odom.floor_z,
                   walked_m=self.odom.walked_m, turned_rad=self.odom.turned_rad,
                   wrists_world=self.wrists(), tables=tables,
                   box=None if box is None else dict(pos=box["pos"].tolist(), quat=matrix_to_wxyz(box["R"]),
                                                     size_m=box["size"], observed_at_s=box["time_s"],
                                                     age_s=now_s - box["time_s"], bounds=bounds))
        if box is not None and source is not None:
            obs["box_height_above_source_m"] = bounds["min"][2] - source["top_z"]
        return obs
