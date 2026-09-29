"""sim_server.py: persistent Isaac Lab process that owns the env + adapter and serves a tiny JSON RPC.

RPC (HTTP on localhost, JSON):  POST /  {"method": name, "params": {...}} -> {"ok": true, "result": ...}
methods: reset(seed, record_dir=None), move_to(targets), get_observation(include_third_person=False),
         get_state(), describe().   GET /health -> {"ok": true, ...}

Run on the box:  ./start_server.sh   (wraps run_on_box.sh sim_server.py)
"""

import argparse
import base64
import io
import json
import math
import os
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import pinocchio  # noqa: F401  (must precede the Isaac app, see NOTES.md)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8765)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
args.enable_pinocchio = True
simulation_app = AppLauncher(args).app

# ----------------------------------------------------------------------------------------------- imports after app
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import isaaclab.envs.mdp as base_mdp  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: E402, F401  (registers the env)
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (  # noqa: E402
    LocomanipulationG1EnvCfg,
    LocomanipulationG1SceneCfg,
)

# ----------------------------------------------------------------------------------------------- scene
TABLE_SIZE = (0.8, 0.6, 0.75)  # x, y, height (m); top at z=0.75
TABLE_A_XY = (0.0, 0.6)  # world; robot faces world +y at reset
TABLE_B_XY = (-1.4, -1.4)  # ~2 m away, behind-left of the robot
BOX = 0.07  # cube edge (m)
IMG_W, IMG_H = 320, 240  # LLM cameras
# tier-0 vision-only observation transforms (no sim state: only camera calibration + forward kinematics)
MAP_PX, MAP_M, MAP_X0, MAP_Y0 = 320, 1.6, -0.3, -0.8  # top-down map: 320 px = 1.6 m; pelvis x in [-0.3, 1.3], y in [-0.8, 0.8]
REACH_M, SHOULDER = 0.55, {"left": (0.0, 0.16, 0.36), "right": (0.0, -0.16, 0.36)}  # from describe(): arm reach, shoulder in pelvis frame
MARK_COLOR = {"left": (255, 220, 0), "right": (255, 0, 255)}
FINGER_LINKS = ["index_1", "middle_1", "thumb_2"]
REC_W, REC_H = 960, 540  # third-person recording camera (never shown to the LLM)
RECORD_CAMS = ["head", "left_wrist", "right_wrist", "third_person"]


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def q_yaw_pitch(yaw, pitch):
    """quat (wxyz) for x-forward/z-up frame: yaw about z, then pitch (positive = nose down) about y."""
    qz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    qy = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
    return qmul(qz, qy)


# camera mounts (link, offset pos, offset quat in the "world" convention) - the calibration a real robot would have.
# Used instead of the sensor's pos_w/quat_w, which do not follow physics-driven links (stale after the first frame).
CAM_MOUNT = {"head": ("torso_link", (0.06, 0.0, 0.42), q_yaw_pitch(0.0, math.radians(35))),
             "left_wrist": ("left_wrist_yaw_link", (-0.06, 0.0, 0.09), q_yaw_pitch(0.0, math.radians(35))),
             "right_wrist": ("right_wrist_yaw_link", (-0.06, 0.0, 0.09), q_yaw_pitch(0.0, math.radians(35)))}
WC_TO_ROS = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # columns: ROS x=-y, y=-z, z=+x of the world-convention camera


def _table(prim, xy, color):
    return RigidObjectCfg(
        prim_path=prim,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(xy[0], xy[1], TABLE_SIZE[2] / 2)),
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


def _cam(prim, pos, rot, focal, size=(IMG_W, IMG_H)):
    return CameraCfg(
        prim_path=prim,
        update_period=0.0,
        height=size[1],
        width=size[0],
        data_types=["rgb", "distance_to_image_plane"],  # depth only feeds measure() (an idealised RGB-D sensor)
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal, clipping_range=(0.05, 30.0)),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
    )


@configclass
class AstraSceneCfg(LocomanipulationG1SceneCfg):
    packing_table = None  # drop the stock packing table
    table_a = _table("{ENV_REGEX_NS}/TableA", TABLE_A_XY, (0.55, 0.45, 0.35))
    table_b = _table("{ENV_REGEX_NS}/TableB", TABLE_B_XY, (0.20, 0.35, 0.70))
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.55, TABLE_SIZE[2] + BOX / 2 + 0.005)),
        spawn=sim_utils.CuboidCfg(
            size=(BOX, BOX, BOX),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
    )
    # G1 has no head link; torso_link sits 4.4 cm above the pelvis, eyes are ~0.45 m higher.
    head_cam = _cam("{ENV_REGEX_NS}/Robot/torso_link/head_cam", (0.06, 0.0, 0.42), q_yaw_pitch(0.0, math.radians(35)), 14.0)
    # wrist cams sit above/behind the wrist (outside the hand mesh) looking down along the fingers
    left_wrist_cam = _cam("{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/wrist_cam", (-0.06, 0.0, 0.09), q_yaw_pitch(0.0, math.radians(35)), 12.0)
    right_wrist_cam = _cam("{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/wrist_cam", (-0.06, 0.0, 0.09), q_yaw_pitch(0.0, math.radians(35)), 12.0)
    third_person_cam = _cam("{ENV_REGEX_NS}/ThirdPersonCam", (2.4, -0.8, 1.9), (1.0, 0.0, 0.0, 0.0), 18.0, size=(REC_W, REC_H))


@configclass
class AstraTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class AstraEnvCfg(LocomanipulationG1EnvCfg):
    scene: AstraSceneCfg = AstraSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    terminations: AstraTerminationsCfg = AstraTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 3600.0  # the server decides when an episode ends, never the env


# ----------------------------------------------------------------------------------------------- adapter
HIP_HEIGHT = 0.72
V_MIN, V_MAX = 0.2, 0.3  # walking speed command band (policy ignores < ~0.15, saturates ~0.25 m/s)
WZ_WALK, WZ_TURN = 0.6, 1.0  # yaw rate while walking / turning in place (in-place needs >= 0.9)
YAW_STOP = math.radians(6)  # stop the in-place turn here; ~5 deg of coast follows
KP_XY, KP_YAW = 1.5, 2.0
BASE_TOL_XY, BASE_TOL_YAW = 0.05, math.radians(5)
BASE_STOP_XY = 0.03  # stop walking here; gait sway + coast leaves ~4-5 cm
HAND_SPEED, HAND_ANG_SPEED = 0.15, 1.0  # m/s, rad/s
MOVE_TIMEOUT_S, SETTLE_S = 15.0, 0.5
STICKY_GRASP_DIST = 0.03  # grasp point to box surface
FALL_Z, BOX_FLOOR_Z = 0.45, 0.30
PI = math.pi
BOUNDS = {
    "base_x": (-3.0, 3.0), "base_y": (-3.0, 3.0), "base_yaw": (-PI, PI),
    "left_x": (-0.1, 0.65), "left_y": (-0.15, 0.6), "left_z": (-0.35, 0.7),
    "right_x": (-0.1, 0.65), "right_y": (-0.6, 0.15), "right_z": (-0.35, 0.7),
    "left_roll": (-PI, PI), "left_pitch": (-PI / 2, PI / 2), "left_yaw": (-PI, PI),
    "right_roll": (-PI, PI), "right_pitch": (-PI / 2, PI / 2), "right_yaw": (-PI, PI),
    "left_hand": (0.0, 1.0), "right_hand": (0.0, 1.0),
}
# action indices of the 7 hand joints per side, and their fully-closed values (open = 0)
HAND_JOINT_IDX = {"left": [14, 15, 16, 20, 21, 22, 26], "right": [17, 18, 19, 23, 24, 25, 27]}
HAND_CLOSED = {"left": [-1.0, -1.0, 0.5, -1.0, -1.0, 0.4, 0.7], "right": [1.0, 1.0, -0.5, 1.0, 1.0, -0.4, -0.7]}
WRIST_LINK = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}
PALM_LINK = {"left": "left_hand_palm_link", "right": "right_hand_palm_link"}
# grasp point = palm link + this (palm frame): ~between the finger pads and the thumb (measured, see NOTES.md)
GRASP_OFFSET = {"right": (0.09, 0.03, 0.0), "left": (0.09, -0.03, 0.0)}


def wrap(a):
    return (a + PI) % (2 * PI) - PI


def log(msg):
    print(f"[sim_server {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Adapter:
    def __init__(self, env, sticky_grasp=True):
        self.env = env
        self.dev = env.device
        self.robot = env.scene["robot"]
        self.box = env.scene["object"]
        self.tables = {"A": env.scene["table_a"], "B": env.scene["table_b"]}
        self.cams = {k: env.scene[k + "_cam"] for k in ["head", "left_wrist", "right_wrist", "third_person"]}
        self.sticky_grasp = sticky_grasp
        names = self.robot.data.body_names
        self.bidx = {n: names.index(n) for n in names}
        self.step_dt = env.step_dt
        self.origin = env.scene.env_origins[0]
        self.record_dir, self.frame_i, self.step_i = None, 0, 0
        self.cmd = {}
        self._hand_goal = {}  # side -> (pos[3], quat[4]) in pelvis frame (interpolation target)
        self._hand_cur = {}  # side -> (pos, quat) currently commanded
        self.attached = {"left": None, "right": None}  # side -> (pos_offset, quat_offset) in palm frame
        self.grasp_events = []
        self.stage, self.max_stage, self._rest_steps = 0, 0, 0
        self.fallen = self.box_on_floor = False
        self._turn_boost, self._stall_ref = False, None
        self.cams["third_person"].set_world_poses_from_view(
            torch.tensor([[2.4, -0.8, 1.9]], device=self.dev), torch.tensor([[-0.4, -0.3, 0.6]], device=self.dev)
        )

    # ---------------- frames
    def _link(self, name):
        i = self.bidx[name]
        return self.robot.data.body_link_pos_w[0, i] - self.origin, self.robot.data.body_link_quat_w[0, i]

    def pelvis(self):
        return self._link("pelvis")

    def base_odom(self):
        p, q = self.pelvis()
        yaw = mu.euler_xyz_from_quat(q.unsqueeze(0))[2].item()
        d = (p[:2] - self.odom_p0).tolist()
        c, s = math.cos(-self.odom_yaw0), math.sin(-self.odom_yaw0)
        return (c * d[0] - s * d[1], s * d[0] + c * d[1], wrap(yaw - self.odom_yaw0))

    def wrist_in_pelvis(self, side):
        pp, pq = self.pelvis()
        wp, wq = self._link(WRIST_LINK[side])
        p, q = mu.subtract_frame_transforms(pp.unsqueeze(0), pq.unsqueeze(0), wp.unsqueeze(0), wq.unsqueeze(0))
        r, pi_, y = mu.euler_xyz_from_quat(q)
        return p[0].tolist(), [wrap(r.item()), wrap(pi_.item()), wrap(y.item())]

    def grasp_point(self, side):
        pp, pq = self._link(PALM_LINK[side])
        off = torch.tensor(GRASP_OFFSET[side], device=self.dev)
        return pp + mu.quat_apply(pq.unsqueeze(0), off.unsqueeze(0))[0]

    def box_surface_dist(self, point):
        bp = self.box.data.root_pos_w[0] - self.origin
        bq = self.box.data.root_quat_w[0]
        local = mu.quat_apply_inverse(bq.unsqueeze(0), (point - bp).unsqueeze(0))[0].abs()
        return torch.clamp(local - BOX / 2, min=0.0).norm().item()

    def hand_openness_measured(self, side):
        jn = "left_hand_index_0_joint" if side == "left" else "right_hand_index_0_joint"
        q = self.robot.data.joint_pos[0, self.robot.data.joint_names.index(jn)].item()
        return float(max(0.0, min(1.0, 1.0 - abs(q) / 1.0)))

    # ---------------- reset
    def reset(self, seed, record_dir=None):
        rng = np.random.default_rng(seed)
        self.env.reset(seed=int(seed))
        # randomize: box on table A, table B position
        bx = TABLE_A_XY[0] + rng.uniform(-0.15, 0.15)
        by = TABLE_A_XY[1] + rng.uniform(-0.10, 0.05)
        byaw = rng.uniform(-0.35, 0.35)
        tb = (TABLE_B_XY[0] + rng.uniform(-0.15, 0.15), TABLE_B_XY[1] + rng.uniform(-0.15, 0.15))
        self.table_xy = {"A": TABLE_A_XY, "B": tb}
        self._write_pose(self.tables["B"], (tb[0], tb[1], TABLE_SIZE[2] / 2), (1, 0, 0, 0))
        self._write_pose(self.tables["A"], (TABLE_A_XY[0], TABLE_A_XY[1], TABLE_SIZE[2] / 2), (1, 0, 0, 0))
        self._write_pose(self.box, (bx, by, TABLE_SIZE[2] + BOX / 2 + 0.003), (math.cos(byaw / 2), 0, 0, math.sin(byaw / 2)))
        self.attached = {"left": None, "right": None}
        self.grasp_events = []
        self.stage = self.max_stage = self._rest_steps = 0
        self.fallen = self.box_on_floor = False
        self.record_dir, self.frame_i, self.step_i = record_dir, 0, 0
        if record_dir:
            for cam in RECORD_CAMS:
                os.makedirs(os.path.join(record_dir, cam), exist_ok=True)
        # commanded hand pose = actual pose; hands open; odometry origin = pelvis now
        self._capture_hands()
        self.cmd = {"left_hand": 1.0, "right_hand": 1.0}
        self._set_odom_origin()
        self._run(int(SETTLE_S / self.step_dt) + 10, base_active=False)  # settle
        self._capture_hands()
        self._set_odom_origin()
        self.cmd.update({"base_x": 0.0, "base_y": 0.0, "base_yaw": 0.0})
        self._sync_cmd_from_goals()
        log(f"reset seed={seed} box=({bx:.2f},{by:.2f}) tableB=({tb[0]:.2f},{tb[1]:.2f}) stage tracking on")
        return {"seed": seed, "box_xy": [bx, by], "table_b_xy": list(tb)}

    def _write_pose(self, asset, pos, quat):
        pose = torch.tensor([[*pos, *quat]], device=self.dev, dtype=torch.float32)
        pose[:, :3] += self.origin
        asset.write_root_pose_to_sim(pose)
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.dev))

    def _set_odom_origin(self):
        p, q = self.pelvis()
        self.odom_p0 = p[:2].clone()
        self.odom_yaw0 = mu.euler_xyz_from_quat(q.unsqueeze(0))[2].item()

    def _capture_hands(self):
        for s in ["left", "right"]:
            pp, pq = self.pelvis()
            wp, wq = self._link(WRIST_LINK[s])
            p, q = mu.subtract_frame_transforms(pp.unsqueeze(0), pq.unsqueeze(0), wp.unsqueeze(0), wq.unsqueeze(0))
            self._hand_goal[s] = (p[0].clone(), q[0].clone())
            self._hand_cur[s] = (p[0].clone(), q[0].clone())

    def _sync_cmd_from_goals(self):
        for s in ["left", "right"]:
            p, q = self._hand_goal[s]
            r, pi_, y = mu.euler_xyz_from_quat(q.unsqueeze(0))
            self.cmd.update({f"{s}_x": p[0].item(), f"{s}_y": p[1].item(), f"{s}_z": p[2].item(),
                             f"{s}_roll": wrap(r.item()), f"{s}_pitch": wrap(pi_.item()), f"{s}_yaw": wrap(y.item())})

    # ---------------- stepping
    def _action(self, base_cmd):
        a = torch.zeros(1, 32, device=self.dev)
        pp, pq = self.pelvis()
        for s, off in [("left", 0), ("right", 7)]:
            p, q = self._hand_cur[s]
            wp, wq = mu.combine_frame_transforms(pp.unsqueeze(0), pq.unsqueeze(0), p.unsqueeze(0), q.unsqueeze(0))
            a[0, off:off + 3] = wp[0]
            a[0, off + 3:off + 7] = wq[0]
            o = self.cmd.get(f"{s}_hand", 1.0)
            for j, v in zip(HAND_JOINT_IDX[s], HAND_CLOSED[s]):
                a[0, j] = v * (1.0 - o)
        a[0, 28:32] = torch.tensor([*base_cmd, HIP_HEIGHT] if len(base_cmd) == 3 else list(base_cmd), device=self.dev)
        return a

    def _advance_hands(self):
        done = True
        for s in ["left", "right"]:
            p, q = self._hand_cur[s]
            gp, gq = self._hand_goal[s]
            d = gp - p
            n = d.norm().item()
            step = HAND_SPEED * self.step_dt
            if n > step:
                p = p + d / n * step
                done = False
            else:
                p = gp.clone()
            ang = mu.quat_error_magnitude(q.unsqueeze(0), gq.unsqueeze(0)).item()
            astep = HAND_ANG_SPEED * self.step_dt
            if ang > astep:
                q = mu.quat_slerp(q, gq, astep / ang)  # non-batched: (4,) tensors
                done = False
            else:
                q = gq.clone()
            self._hand_cur[s] = (p, q)
        return done

    def _base_cmd(self):
        """Velocity command from pose error, shaped around the measured policy behaviour (see NOTES.md):
        walking needs |v| >= ~0.15 m/s and saturates ~0.25 m/s; wz is only honoured while walking (<= 0.6)
        or, standing, at |wz| >= 0.9 (turn in place, ~5 deg coast after the command stops)."""
        x, y, yaw = self.base_odom()
        ex, ey = self.cmd["base_x"] - x, self.cmd["base_y"] - y
        eyaw = wrap(self.cmd["base_yaw"] - yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        exb, eyb = c * ex + s * ey, -s * ex + c * ey
        dist = math.hypot(ex, ey)
        if dist > BASE_STOP_XY:  # walk (omnidirectional) while trimming yaw
            mag = max(V_MIN, min(V_MAX, KP_XY * dist))
            vx, vy = exb / dist * mag, eyb / dist * mag
            wz = max(-WZ_WALK, min(WZ_WALK, KP_YAW * eyaw))
            return (vx, vy, wz), False
        if abs(eyaw) > YAW_STOP:  # turn in place at the rate the policy actually responds to
            if self._turn_boost:  # the policy occasionally refuses to start an in-place turn: walk-turn instead
                return (0.15, 0.0, math.copysign(WZ_WALK, eyaw)), False
            return (0.0, 0.0, math.copysign(WZ_TURN, eyaw)), False
        return (0.0, 0.0, 0.0), True

    def _check_turn_stall(self):
        """Every 1.5 s of turning: if yaw barely changed, toggle the walk-turn fallback."""
        yaw = self.base_odom()[2]
        if self._stall_ref is None:
            self._stall_ref = (self.step_i, yaw)
            return
        if self.step_i - self._stall_ref[0] >= int(1.5 / self.step_dt):
            progressed = abs(wrap(yaw - self._stall_ref[1])) > math.radians(3)
            if not progressed:
                self._turn_boost = not self._turn_boost
                log(f"turn stalled for 1.5 s; walk-turn fallback -> {self._turn_boost}")
            self._stall_ref = (self.step_i, yaw)

    def _step(self, base_cmd):
        self.env.step(self._action(base_cmd))
        self.step_i += 1
        self._update_grasp()
        self._update_stage()
        p, _ = self.pelvis()
        if p[2].item() < FALL_Z:
            self.fallen = True
        if (self.box.data.root_pos_w[0, 2] - self.origin[2]).item() < BOX_FLOOR_Z:
            self.box_on_floor = True
        if self.record_dir and self.step_i % 5 == 0:  # 10 fps, every camera, raw pixels (overlays are for the LLM only)
            for cam in RECORD_CAMS:
                Image.fromarray(self._rgb(cam)).save(os.path.join(self.record_dir, cam, f"frame_{self.frame_i:05d}.jpg"), quality=85)
            self.frame_i += 1

    def _run(self, max_steps, base_active):
        """Step until hands + base are done (or max_steps). Returns (steps, base_reached)."""
        reached = not base_active
        self._turn_boost, self._stall_ref = False, None
        for i in range(max_steps):
            if base_active:
                cmd, reached = self._base_cmd()
                if cmd[0] == 0.0 and cmd[1] == 0.0 and cmd[2] != 0.0 or self._turn_boost:
                    self._check_turn_stall()
                else:
                    self._stall_ref = None
                if reached:
                    cmd = (0.0, 0.0, 0.0)
            else:
                cmd = (0.0, 0.0, 0.0)
            hands_done = self._advance_hands()
            self._step(cmd)
            if hands_done and reached:
                return i + 1, reached
            if self.fallen:
                return i + 1, reached
        return max_steps, reached

    # ---------------- sticky grasp
    def _update_grasp(self):
        if not self.sticky_grasp:
            return
        for s in ["left", "right"]:
            closed = self.cmd.get(f"{s}_hand", 1.0) < 0.5
            if self.attached[s] is None and closed:
                d = self.box_surface_dist(self.grasp_point(s))
                if d <= STICKY_GRASP_DIST and self.attached["left" if s == "right" else "right"] is None:
                    pp, pq = self._link(PALM_LINK[s])
                    bp = self.box.data.root_pos_w[0] - self.origin
                    bq = self.box.data.root_quat_w[0]
                    op, oq = mu.subtract_frame_transforms(pp.unsqueeze(0), pq.unsqueeze(0), bp.unsqueeze(0), bq.unsqueeze(0))
                    self.attached[s] = (op[0].clone(), oq[0].clone())
                    self.grasp_events.append({"step": self.step_i, "hand": s, "event": "attach", "dist": d})
                    log(f"STICKY GRASP: {s} hand closed {d * 100:.1f} cm from box -> attached")
            elif self.attached[s] is not None and not closed:
                self.attached[s] = None
                self.grasp_events.append({"step": self.step_i, "hand": s, "event": "release"})
                log(f"STICKY GRASP: {s} hand opened -> released box")
            if self.attached[s] is not None:
                pp, pq = self._link(PALM_LINK[s])
                op, oq = self.attached[s]
                bp, bq = mu.combine_frame_transforms(pp.unsqueeze(0), pq.unsqueeze(0), op.unsqueeze(0), oq.unsqueeze(0))
                self._write_pose(self.box, bp[0].tolist(), bq[0].tolist())

    # ---------------- scoring (privileged)
    def _update_stage(self):
        bp = self.box.data.root_pos_w[0] - self.origin
        held = any(v is not None for v in self.attached.values())
        dists = {s: self.box_surface_dist(self.grasp_point(s)) for s in ["left", "right"]}
        near = min(dists.values()) < 0.10
        table_top = TABLE_SIZE[2]
        tb = self.table_xy["B"]
        on_b_footprint = abs(bp[0].item() - tb[0]) < TABLE_SIZE[0] / 2 and abs(bp[1].item() - tb[1]) < TABLE_SIZE[1] / 2
        if self.max_stage < 1 and min(dists.values()) <= 0.05:
            self.max_stage = 1
        if self.max_stage < 2 and (held or near) and bp[2].item() > table_top + BOX / 2 + 0.05:
            self.max_stage = 2
        if self.max_stage < 3 and (held or near) and math.hypot(bp[0].item() - tb[0], bp[1].item() - tb[1]) < 0.5 and bp[2].item() > table_top:
            self.max_stage = 3
        resting = on_b_footprint and abs(bp[2].item() - (table_top + BOX / 2)) < 0.02 and not held \
            and self.box.data.root_lin_vel_w[0].norm().item() < 0.02
        self._rest_steps = self._rest_steps + 1 if resting else 0
        if self.max_stage >= 3 and self._rest_steps >= int(1.0 / self.step_dt):
            self.max_stage = 4
        self.stage = self.max_stage

    # ---------------- RPC-facing
    def move_to(self, targets):
        if self.fallen:
            return {"text": "robot has fallen; episode over", "fallen": True}
        clipped, unknown = [], [n for n in targets if n not in BOUNDS]
        if unknown:
            raise ValueError(f"unknown dimensions {unknown}; valid: {list(BOUNDS)}")
        base_named = any(n.startswith("base_") for n in targets)
        hand_named = any(not n.startswith("base_") and not n.endswith("_hand") for n in targets)
        vals = {}
        for n, v in targets.items():
            lo, hi = BOUNDS[n]
            if n.endswith("_yaw") or n.endswith("_roll"):
                v = wrap(float(v))  # angles wrap, they are never clipped
            cv = float(min(hi, max(lo, float(v))))
            if cv != float(v):
                clipped.append(f"{n}={float(v):.3f}->{cv:.3f}")
            vals[n] = cv
        self.cmd.update(vals)
        for s in ["left", "right"]:
            if any(n.startswith(s + "_") and not n.endswith("_hand") for n in vals):
                p = torch.tensor([self.cmd[f"{s}_x"], self.cmd[f"{s}_y"], self.cmd[f"{s}_z"]], device=self.dev)
                q = mu.quat_from_euler_xyz(*[torch.tensor([self.cmd[f"{s}_{k}"]], device=self.dev) for k in ["roll", "pitch", "yaw"]])[0]
                self._hand_goal[s] = (p, q)
        t0 = time.time()
        steps, reached = self._run(int(MOVE_TIMEOUT_S / self.step_dt), base_active=base_named)
        timed_out = steps >= int(MOVE_TIMEOUT_S / self.step_dt) and not (reached and self._advance_hands())
        n_settle = int(SETTLE_S / self.step_dt)
        for _ in range(n_settle):
            self._advance_hands()
            self._step((0.0, 0.0, 0.0))
        # errors
        err = {}
        bx, by, byaw = self.base_odom()
        for n, v in vals.items():
            if n == "base_x":
                err[n] = f"{self.cmd['base_x'] - bx:+.3f}m"
            elif n == "base_y":
                err[n] = f"{self.cmd['base_y'] - by:+.3f}m"
            elif n == "base_yaw":
                err[n] = f"{math.degrees(wrap(self.cmd['base_yaw'] - byaw)):+.1f}deg"
            elif n.endswith("_hand"):
                err[n] = f"{v - self.hand_openness_measured(n[:-5]):+.2f}"
            else:
                s, k = n.split("_", 1)
                p, rpy = self.wrist_in_pelvis(s)
                actual = dict(zip(["x", "y", "z", "roll", "pitch", "yaw"], p + rpy))[k]
                err[n] = f"{v - actual:+.3f}m" if k in "xyz" else f"{math.degrees(wrap(v - actual)):+.1f}deg"
        grasp = [e for e in self.grasp_events if e["step"] > self.step_i - steps - n_settle]
        parts = [
            "commanded: " + " ".join(f"{n}={v:.3f}" for n, v in vals.items()),
            f"motion {'TIMED OUT' if timed_out else 'completed'} after {steps * self.step_dt:.1f}s sim ({time.time() - t0:.1f}s wall)",
            "final error: " + " ".join(f"{n}={e}" for n, e in err.items()),
            "clipped to bounds: " + (", ".join(clipped) if clipped else "none"),
        ]
        if grasp:
            parts.append("sticky grasp: " + ", ".join(f"{g['hand']} {g['event']}" for g in grasp))
        if base_named and hand_named:
            parts.append("note: base and hands were commanded together")
        if self.fallen:
            parts.append("ROBOT FELL")
        return {"text": " | ".join(parts), "fallen": self.fallen, "box_on_floor": self.box_on_floor,
                "timed_out": timed_out, "clipped": clipped, "grasp_events": grasp, "sim_time": self.step_i * self.step_dt}

    def _rgb(self, cam):
        img = self.cams[cam].data.output["rgb"][0]
        return img[..., :3].detach().cpu().numpy().astype(np.uint8)

    # ---------------- tier-0 vision-only transforms: marks (FK + calibration) and an inverse-perspective top-down map
    def _cam_KRt(self, cam):
        """Pinhole intrinsics K, camera-to-world rotation R (OpenCV/ROS axes: +z forward, +x right, +y down), position t."""
        K = self.cams[cam].data.intrinsic_matrices[0].cpu().numpy()
        link, off_p, off_q = CAM_MOUNT[cam]
        lp, lq = self._link(link)
        cp, cq = mu.combine_frame_transforms(lp.unsqueeze(0), lq.unsqueeze(0),
                                             torch.tensor([off_p], device=self.dev, dtype=torch.float32),
                                             torch.tensor([off_q], device=self.dev, dtype=torch.float32))
        R = mu.matrix_from_quat(cq)[0].cpu().numpy() @ WC_TO_ROS
        return K, R, cp[0].cpu().numpy()

    def _project(self, cam, P):
        """(N,3) env-local world points -> pixel u, v and a validity mask (in front of the camera, inside the image)."""
        K, R, t = self._cam_KRt(cam)
        pc = (P - t) @ R
        z = np.maximum(pc[:, 2], 1e-6)
        u, v = K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]
        ok = (pc[:, 2] > 0.03) & (u >= 0) & (u < IMG_W - 1) & (v >= 0) & (v < IMG_H - 1)
        return u, v, ok

    def _pelvis_to_world(self, P_pel):
        pp, pq = self.pelvis()
        Rp = mu.matrix_from_quat(pq.unsqueeze(0))[0].cpu().numpy()
        return P_pel @ Rp.T + pp.cpu().numpy()

    def _annotate(self, cam, img):
        """Draw each hand's grasp point (L/R circle) and fingertips (dots) into a camera image, from FK only."""
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        for side in ["left", "right"]:
            pts = np.stack([self.grasp_point(side).cpu().numpy()] + [self._link(f"{side}_hand_{ln}_link")[0].cpu().numpy() for ln in FINGER_LINKS])
            u, v, ok = self._project(cam, pts)
            col = MARK_COLOR[side]
            for i in range(1, len(pts)):
                if ok[i]:
                    dr.ellipse([u[i] - 2, v[i] - 2, u[i] + 2, v[i] + 2], fill=col)
            if ok[0]:
                dr.ellipse([u[0] - 7, v[0] - 7, u[0] + 7, v[0] + 7], outline=col, width=2)
                dr.text((u[0] + 9, v[0] - 6), side[0].upper(), fill=col)
        return np.asarray(im)

    def _body_map(self):
        """Top-down image in the pelvis frame: the head camera image re-projected onto the horizontal plane at the current
        (lower) grasp-point height, with a 0.25 m grid, the robot, both grasp points and the arm-reach circles drawn on top."""
        pp, pq = self.pelvis()
        gp = {s: mu.quat_apply_inverse(pq.unsqueeze(0), (self.grasp_point(s) - pp).unsqueeze(0))[0].cpu().numpy() for s in ["left", "right"]}
        z_plane = float(min(gp["left"][2], gp["right"][2]))
        res = MAP_M / MAP_PX
        r, c = np.meshgrid(np.arange(MAP_PX), np.arange(MAP_PX), indexing="ij")
        x = MAP_X0 + MAP_M - (r + 0.5) * res  # +x up
        y = MAP_Y0 + MAP_M - (c + 0.5) * res  # +y left
        P = np.stack([x.ravel(), y.ravel(), np.full(x.size, z_plane)], axis=1)
        u, v, ok = self._project("head", self._pelvis_to_world(P))
        rgb = self._rgb("head")
        out = np.full((MAP_PX * MAP_PX, 3), 110, np.uint8)
        out[ok] = rgb[v[ok].astype(int), u[ok].astype(int)]
        im = Image.fromarray(out.reshape(MAP_PX, MAP_PX, 3))
        dr = ImageDraw.Draw(im)

        def px(xm, ym):  # pelvis (x, y) -> map pixel (col, row)
            return ((MAP_Y0 + MAP_M - ym) / res, (MAP_X0 + MAP_M - xm) / res)

        for g in np.arange(-1.0, 1.51, 0.25):
            col = (235, 235, 235) if abs(g % 0.5) < 1e-6 else (190, 190, 190)
            cx, _ = px(0, g)
            _, ry = px(g, 0)
            if 0 <= cx < MAP_PX:
                dr.line([cx, 0, cx, MAP_PX], fill=col, width=1)
                if abs(g % 0.5) < 1e-6:
                    dr.text((cx + 2, MAP_PX - 12), f"y={g:+.1f}", fill=(255, 255, 255))
            if 0 <= ry < MAP_PX:
                dr.line([0, ry, MAP_PX, ry], fill=col, width=1)
                if abs(g % 0.5) < 1e-6:
                    dr.text((2, ry - 11), f"x={g:+.1f}", fill=(255, 255, 255))
        for s in ["left", "right"]:  # reach circles around the shoulders
            sx, sy = px(SHOULDER[s][0], SHOULDER[s][1])
            rr = REACH_M / res
            dr.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], outline=MARK_COLOR[s], width=1)
        ox, oy = px(0, 0)  # robot: body circle + heading arrow
        dr.ellipse([ox - 24, oy - 24, ox + 24, oy + 24], outline=(255, 255, 255), width=2)
        dr.polygon([(ox, oy - 34), (ox - 8, oy - 20), (ox + 8, oy - 20)], fill=(255, 255, 255))
        for s in ["left", "right"]:
            gx, gy = px(gp[s][0], gp[s][1])
            dr.ellipse([gx - 6, gy - 6, gx + 6, gy + 6], outline=MARK_COLOR[s], width=2)
            dr.text((gx + 8, gy - 6), s[0].upper(), fill=MARK_COLOR[s])
        dr.rectangle([0, 0, MAP_PX, 13], fill=(0, 0, 0))
        dr.text((2, 1), f"top-down, pelvis frame, plane z={z_plane:+.2f} (hand height); +x up, +y left; grid 0.25 m", fill=(255, 255, 255))
        return np.asarray(im)

    def measure(self, camera, x, y, label=""):
        """Back-project pixel (x, y) of an LLM camera through the depth buffer: the 3D point in the pelvis frame and its
        offset from each hand's grasp point. Idealised sensor version of a two-view triangulation tool; no object poses used."""
        if camera not in ("head", "left_wrist", "right_wrist"):
            raise ValueError(f"unknown camera {camera}; use head, left_wrist or right_wrist")
        x, y = int(round(float(x))), int(round(float(y)))
        if not (0 <= x < IMG_W and 0 <= y < IMG_H):
            raise ValueError(f"pixel ({x}, {y}) outside the {IMG_W}x{IMG_H} image")
        depth = self.cams[camera].data.output["distance_to_image_plane"][0, ..., 0].detach().cpu().numpy()
        win = depth[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        win = win[np.isfinite(win) & (win > 0.02)]
        if win.size == 0:
            return {"text": f"measure({camera}, {x}, {y}): nothing there (sky/background)", "point": None}
        z = float(np.median(win))
        K, R, t = self._cam_KRt(camera)
        pc = np.array([(x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z])
        pw = R @ pc + t
        pp, pq = self.pelvis()
        pt = mu.quat_apply_inverse(pq.unsqueeze(0), (torch.tensor(pw, device=self.dev, dtype=torch.float32) - pp).unsqueeze(0))[0].cpu().numpy()
        parts = [f"measure({camera}, {x}, {y}){' ' + label if label else ''}: point at pelvis-frame x={pt[0]:.2f} y={pt[1]:.2f} z={pt[2]:.2f} "
                 f"({z:.2f} m from the camera)"]
        for s_ in ["left", "right"]:
            g = mu.quat_apply_inverse(pq.unsqueeze(0), (self.grasp_point(s_) - pp).unsqueeze(0))[0].cpu().numpy()
            d = pt - g
            parts.append(f"{s_} grasp point is {abs(d[0]) * 100:.0f} cm {'short of' if d[0] > 0 else 'beyond'} it, "
                         f"{abs(d[1]) * 100:.0f} cm to the {'right' if d[1] > 0 else 'left'} of it, "
                         f"{abs(d[2]) * 100:.0f} cm {'below' if d[2] > 0 else 'above'} it (straight-line {np.linalg.norm(d) * 100:.0f} cm)")
        return {"text": "; ".join(parts), "point": pt.tolist()}

    def debug_depth(self, camera="head"):
        """Debug: the camera's RGB and depth (normalised, 8-bit) as base64 JPEG/PNG plus depth stats, for checking sync."""
        d = self.cams[camera].data
        depth = d.output["distance_to_image_plane"][0, ..., 0].detach().cpu().numpy()
        rgb = self._rgb(camera)
        fin = np.isfinite(depth)
        dn = np.clip(depth, 0, 3.0) / 3.0
        dn[~fin] = 1.0
        out = {}
        for name, arr in [("rgb", rgb), ("depth", (dn * 255).astype(np.uint8))]:
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            out[name] = base64.b64encode(buf.getvalue()).decode()
        out["stats"] = {"shape": list(depth.shape), "min": float(np.nanmin(depth[fin])), "median": float(np.nanmedian(depth[fin])),
                        "rgb_shape": list(rgb.shape), "cam_pos_sensor": (d.pos_w[0] - self.origin).tolist(),
                        "R_sensor": mu.matrix_from_quat(d.quat_w_ros[0].unsqueeze(0))[0].tolist(),
                        "cam_pos_fk": self._cam_KRt(camera)[2].tolist(), "R_fk": self._cam_KRt(camera)[1].tolist(),
                        "K": d.intrinsic_matrices[0].tolist()}
        return out

    def get_observation(self, include_third_person=False, marks=True, body_map=True):
        cams = ["head", "left_wrist", "right_wrist"] + (["third_person"] if include_third_person else [])
        images = {}
        for c in cams:
            img = self._rgb(c)
            if marks and c != "third_person":
                img = self._annotate(c, img)
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=85)
            images[c] = base64.b64encode(buf.getvalue()).decode()
        if body_map:
            buf = io.BytesIO()
            Image.fromarray(self._body_map()).save(buf, format="JPEG", quality=85)
            images["body_map"] = base64.b64encode(buf.getvalue()).decode()
        bx, by, byaw = self.base_odom()
        text = {}
        for s in ["left", "right"]:
            p, rpy = self.wrist_in_pelvis(s)
            text.update({f"{s}_x": p[0], f"{s}_y": p[1], f"{s}_z": p[2], f"{s}_roll": rpy[0], f"{s}_pitch": rpy[1], f"{s}_yaw": rpy[2],
                         f"{s}_hand": self.cmd.get(f"{s}_hand", 1.0)})
        text.update({"base_x": bx, "base_y": by, "base_yaw": byaw})
        return {"images": images, "image_size": [IMG_W, IMG_H], "state_text": " ".join(f"{k}={v:.2f}" for k, v in text.items()),
                "state": text, "sim_time": self.step_i * self.step_dt}

    def get_state(self):
        bp = (self.box.data.root_pos_w[0] - self.origin).tolist()
        pp, pq = self.pelvis()
        st = {
            "sim_time": self.step_i * self.step_dt, "step": self.step_i,
            "box": {"pos": bp, "quat": self.box.data.root_quat_w[0].tolist(), "size": BOX},
            "tables": {k: {"center_xy": list(v), "top_z": TABLE_SIZE[2], "size_xy": list(TABLE_SIZE[:2])} for k, v in self.table_xy.items()},
            "pelvis": {"pos": pp.tolist(), "quat": pq.tolist(), "odom": list(self.base_odom())},
            "hands": {}, "commanded": dict(self.cmd),
            "attached": {s: self.attached[s] is not None for s in self.attached},
            "grasp_events": self.grasp_events, "stage": self.stage, "max_stage": self.max_stage,
            "fallen": self.fallen, "box_on_floor": self.box_on_floor,
        }
        def in_pelvis(pw):
            return mu.quat_apply_inverse(pq.unsqueeze(0), (pw - pp).unsqueeze(0))[0].tolist()

        for s in ["left", "right"]:
            gp = self.grasp_point(s)
            p, rpy = self.wrist_in_pelvis(s)
            fingers = {ln: in_pelvis(self._link(f"{s}_hand_{ln}_link")[0]) for ln in ["palm", "index_1", "middle_1", "thumb_2"]}
            st["hands"][s] = {"wrist_pelvis_pos": p, "wrist_pelvis_rpy": rpy, "grasp_point_world": gp.tolist(),
                              "grasp_point_pelvis": in_pelvis(gp), "box_surface_dist": self.box_surface_dist(gp),
                              "openness_measured": self.hand_openness_measured(s), "links_pelvis": fingers}
        st["box"]["pelvis"] = in_pelvis(self.box.data.root_pos_w[0] - self.origin)
        # table centres in the odometry frame (what base_x/base_y are expressed in)
        c, sn = math.cos(-self.odom_yaw0), math.sin(-self.odom_yaw0)
        for k, v in self.table_xy.items():
            dx, dy = v[0] - self.odom_p0[0].item(), v[1] - self.odom_p0[1].item()
            st["tables"][k]["center_odom"] = [c * dx - sn * dy, sn * dx + c * dy]
            st["tables"][k]["yaw_odom"] = wrap(-self.odom_yaw0)  # table edges are world-axis aligned
        return st

    def calibrate(self, vx, vy, wz, seconds=3.0, hip=HIP_HEIGHT):
        """Debug: hold a raw locomotion command for `seconds` and report the achieved body-frame velocities."""
        x0, y0, yaw0 = self.base_odom()
        n = int(seconds / self.step_dt)
        for _ in range(n):
            self._advance_hands()
            self._step((vx, vy, wz, hip))
            if self.fallen:
                break
        x1, y1, yaw1 = self.base_odom()
        dx, dy = x1 - x0, y1 - y0
        c, s = math.cos(yaw0), math.sin(yaw0)
        t = n * self.step_dt
        for _ in range(int(1.0 / self.step_dt)):
            self._step((0.0, 0.0, 0.0))
        x2, y2, yaw2 = self.base_odom()
        return {"cmd": [vx, vy, wz], "vx_meas": (c * dx + s * dy) / t, "vy_meas": (-s * dx + c * dy) / t,
                "wz_meas": wrap(yaw1 - yaw0) / t, "fallen": self.fallen,
                "coast_xy": math.hypot(x2 - x1, y2 - y1), "coast_yaw": wrap(yaw2 - yaw1)}

    def describe(self, marks=False, body_map=False):
        lines = [
            "Frames: +x forward, +y left, +z up. Hand targets (left_*/right_*) are in the PELVIS frame (origin at the pelvis,",
            "moves with the robot; roll/pitch/yaw in radians relative to the reset hand orientation, 0,0,0 = fingers pointing",
            "forward). base_x/base_y/base_yaw are in the ODOMETRY frame fixed at the robot's reset pose (yaw in radians).",
            f"Hands: left_hand/right_hand scalar, 0 = closed, 1 = open. Approx arm reach: 0.55 m from the shoulder (shoulders at",
            "pelvis x=0, y=+/-0.16, z=+0.36); comfortable workspace x 0.15..0.55, z -0.2..0.5.",
            f"Hand motion speed cap {HAND_SPEED} m/s; walking speed ~0.25 m/s, turning ~1 rad/s; a move_to call times out after {MOVE_TIMEOUT_S}s sim.",
            "Bounds: " + ", ".join(f"{n} [{lo:.2f}, {hi:.2f}]" for n, (lo, hi) in BOUNDS.items()),
        ]
        if marks:
            lines += ["Camera images are annotated from the robot's own kinematics: a circle labelled L (yellow) or R (magenta) marks each",
                      "hand's grasp point, i.e. where an object must be, between fingers and thumb, when the hand closes for it to be held;",
                      "small dots are the fingertips."]
        if body_map:
            lines += ["The top-down map is NOT a camera: it is the head image re-projected onto a horizontal plane at the current hand",
                      "height in the pelvis frame (+x up, +y left, 0.25 m grid, labelled every 0.5 m). Surfaces at that height appear at",
                      "true scale and position; anything above or below it is displaced; grey = outside the head camera's view.",
                      "Circles: arm reach around each shoulder, L/R grasp points."]
        return {"text": "\n".join(lines), "bounds": {k: list(v) for k, v in BOUNDS.items()}, "sticky_grasp": self.sticky_grasp}


# ----------------------------------------------------------------------------------------------- server
log("creating env ...")
t0 = time.time()
env = gym.make("Isaac-PickPlace-Locomanipulation-G1-Abs-v0", cfg=AstraEnvCfg()).unwrapped
adapter = Adapter(env)
adapter.reset(seed=0)
log(f"env + adapter ready in {time.time() - t0:.1f}s; serving on 127.0.0.1:{args.port}")
METHODS = {"reset": adapter.reset, "move_to": adapter.move_to, "get_observation": adapter.get_observation,
           "get_state": adapter.get_state, "describe": adapter.describe, "calibrate": adapter.calibrate, "measure": adapter.measure, "debug_depth": adapter.debug_depth}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"ok": True, "sim_time": adapter.step_i * adapter.step_dt, "stage": adapter.max_stage})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            fn = METHODS[req["method"]]
            self._send(200, {"ok": True, "result": fn(**req.get("params", {}))})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *a):
        log(f"rpc {self.command} {fmt % a}")


HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
