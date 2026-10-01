"""sim_server_simple.py: the Astra sim server backed by SIMPLE (MuJoCo physics + Isaac Sim 4.5 rendering, Unitree G1
with Dex3 hands and the AMO whole-body controller) instead of Isaac Lab. Same localhost JSON RPC as sim_server.py:
reset(seed, record_dir), move_to(targets), get_observation(), get_state(), measure(camera, x, y), describe().

Run on the box inside the SIMPLE venv:
  cd ~/SIMPLE && MUJOCO_GL=egl OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python ~/astra/sim_server_simple.py --port 8765

Named dimensions (all optional per call, unnamed ones hold): left_x/y/z/roll/pitch/yaw, right_*, left_hand, right_hand
(0 closed .. 1 open), base_x, base_y, base_yaw (odometry frame fixed at reset), height (metres relative to the standing
pelvis height, <= 0 crouches). Hand targets are the hand base (wrist_yaw link) pose in the PELVIS frame; orientation is
relative to the hand orientation at reset. Arms are position-controlled through a damped-least-squares IK on the MuJoCo
model, the legs/waist through AMO. `step_axes(...)` exposes safe, short finite-direction control ticks for the Jev
controller; unlike `move_to`, it never accepts a model-generated absolute pose target. No sticky grasp: the Dex3 fingers
have to hold the object physically.
"""
import argparse
from collections import deque
import base64
import json
import math
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--env-id", default="simple/G1WholebodyBendPickMP-v0")
    parser.add_argument("--task", default="g1_wholebody_bend_pick_mp")
    parser.add_argument("--sim-mode", choices=["mujoco", "mujoco_isaac"], default="mujoco_isaac",
                        help="mujoco omits Isaac rendering; use it for structured-state controllers such as Jev")
    parser.add_argument("--record-every", type=int, default=5, help="record a frame every N control steps (50 Hz)")
    parser.add_argument("--scene-uid", default=None, help="override the task's scene, e.g. hssd:scene7 (tasks that hard-code a scene ignore it)")
    parser.add_argument("--floor-box", default=None, help="replace the task target with a primitive box of this size 'dx,dy,dz' (m) "
                        "spawned on the floor ~0.5 m ahead of the robot; no distractors")
    parser.add_argument("--box-mass", type=float, default=0.5, help="mass (kg) of the --floor-box")
    parser.add_argument("--instruction", default=None, help="override the task instruction given to the LLM")
    parser.add_argument("--target-object", default=None, help="task target asset, e.g. graspnet1b:12 (apple, the tabletop default)")
    parser.add_argument("--legacy-close", action="store_true",
                        help="old hand model: close to SIMPLE's G1 cfg close_qpos (thumb barely moves, ~10 cm gap) and grasp point = "
                             "open-fingertip centroid; default is the pinch pose below")
    parser.add_argument("--sticky-grasp", action="store_true",
                        help="weld the target to a hand that closes within STICKY_DIST of it (as in the Isaac Lab version); "
                             "off = the Dex3 fingers must hold the object physically")
    return parser



import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import simple.envs  # noqa: E402,F401
from simple.core.action import ActionCmd  # noqa: E402

CTRL_DT = 0.02  # one env.step = 10 MuJoCo substeps of 2 ms
HAND_SPEED, HAND_ANG_SPEED = 0.15, 1.2  # m/s, rad/s caps for hand target interpolation
MOVE_TIMEOUT_S, SETTLE_S = 8.0, 0.5
V_MAX, WZ_MAX, KP_XY = 0.35, 0.8, 1.2
BASE_STOP_XY, YAW_STOP = 0.04, math.radians(5)
STICKY_DIST = 0.04  # grasp point to target centre
CONTACT_STOP_N = 40.0  # arm force limit: an arm pressing on anything (fixed or movable) harder than this is stopped and backed off
CONTACT_REPORT_N = 3.0  # external contact on an arm is reported (per arm, no body names) above this
CONTACT_HIST = 8  # hand poses kept (control steps) to roll back to the last contact-free one
# Pinch pose (right hand; the left hand is the exact sign flip). Found by an FK sweep (scratchpad hand_fk.py): along the
# straight line from open (all zeros) to this pose the thumb tip and the index/middle tips close onto one point,
# GRASP_CENTRE in the hand-base (wrist_yaw_link) frame, which drifts < 3 mm for object widths 2-10 cm. SIMPLE's cfg
# close_qpos barely moves the thumb and leaves a ~10 cm gap, so it never pinches anything smaller.
PINCH = {"right": {"thumb_0": 0.0, "thumb_1": -0.79, "thumb_2": -0.21, "index_0": 0.75, "index_1": 1.13, "middle_0": 0.75, "middle_1": 1.13}}
PINCH["left"] = {k: -v for k, v in PINCH["right"].items()}
SQUEEZE = 1.2  # command past the pinch pose so the finger PD keeps pressing on whatever it contacts
GRASP_CENTRE = {"right": np.array([0.146, 0.077, 0.0]), "left": np.array([0.146, -0.077, 0.0])}
LLM_CAM, THIRD_CAM = "head_stereo_left", "front_stereo_left"
RECORD_CAMS = {"head": LLM_CAM, "third_person": THIRD_CAM, "side": "side_left"}  # cameras the task lacks are skipped
MARK_COLOR = {"left": (0, 220, 255), "right": (255, 0, 255)}  # BGR for cv2
PI = math.pi
BOUNDS = {}
for _s in ["left", "right"]:
    BOUNDS.update({f"{_s}_x": (-0.15, 0.75), f"{_s}_y": (-0.7, 0.7), f"{_s}_z": (-0.9, 0.6),
                   f"{_s}_roll": (-PI, PI), f"{_s}_pitch": (-PI, PI), f"{_s}_yaw": (-PI, PI), f"{_s}_hand": (0.0, 1.0)})
BOUNDS.update({"base_x": (-4.0, 4.0), "base_y": (-4.0, 4.0), "base_yaw": (-PI, PI), "height": (-0.23, 0.02)})
AMO_HEIGHT_GAIN = 2.0  # AMO settles at about half the commanded d_height; -0.45 commanded => -0.23 m achieved
IK_WEIGHT = {"waist_yaw_joint": 0.3, "waist_roll_joint": 0.15, "waist_pitch_joint": 0.6}  # arms weight 1: prefer arm motion
ARM_JOINTS = {s: [f"{s}_{j}_joint" for j in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]]
              for s in ["left", "right"]}
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
HAND_BASE = {s: f"{s}_wrist_yaw_link" for s in ["left", "right"]}
FINGERTIPS = {s: [f"{s}_hand_{f}_finger_tip" for f in ["thumb", "index", "middle"]] for s in ["left", "right"]}


def wrap(a):
    return (a + PI) % (2 * PI) - PI


def log(msg):
    print(f"[sim_server_simple {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def mat_to_rpy(R):
    p = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    return math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])


def mat_to_rotvec(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


class Adapter:
    def __init__(self, env, record_every=5, sticky_grasp=False, legacy_close=False, instruction=None, box_mass=None):
        self.env = env
        self._instruction, self.box_mass = instruction, box_mass
        self.sticky_grasp = sticky_grasp
        self.attached = {"left": None, "right": None}  # side -> (pos offset, rot offset) of the target in the hand-base frame
        self.mj = env.mujoco
        self.isaac = getattr(env, "isaac", None)
        self.robot = env.task.robot
        self.task = env.task
        self.record_every = record_every
        self.legacy_close = legacy_close
        if legacy_close:
            close = {"left": [0.3523, -0.0964, 0.2790, -0.5058, -1.1950, -0.5389, -0.9835],
                     "right": [0.02331954, -0.02398408, -0.22170663, 0.25662386, 1.3371105, 0.3085137, 0.9805285]}  # G1Wholebody cfg
            self.close_qpos = {s: dict(zip([j for j in self.robot.hand_names if j.startswith(s)], close[s])) for s in close}
        else:
            self.close_qpos = {s: {f"{s}_hand_{n}_joint": SQUEEZE * v for n, v in PINCH[s].items()} for s in PINCH}
        self.record_dir, self.frame_i, self.step_i = None, 0, 0
        self.cmd = {}
        self._hand_goal, self._hand_cur, self._arm_q = {}, {}, {}
        self._R0 = {}
        self.stage = self.max_stage = self._hold_steps = 0
        self.fallen = self.obj_on_floor = False
        self._frames = None
        self._last_frames_step = -1
        self.grasp_events = []
        self._blocked = {"left": None, "right": None}  # side -> body name the hand was stopped against during this move_to
        self._hand_hist = {"left": deque(maxlen=CONTACT_HIST), "right": deque(maxlen=CONTACT_HIST)}
        self._trace = deque(maxlen=4000)  # per control step, for debug_trace
        self._q_prev = None

    # ---------------- model handles (rebuilt on every reset: SIMPLE rebuilds the MjModel per episode)
    def _bind(self):
        self.m, self.d = self.mj.mjModel, self.mj.mjData
        names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(self.m.nbody)]

        def bid(n):
            for i, nm in enumerate(names):
                if nm == n or nm.endswith("/" + n):
                    return i
            raise KeyError(f"body {n} not in model; have {names[:60]}")

        self.bid = {n: bid(n) for n in ["pelvis"] + [HAND_BASE[s] for s in HAND_BASE] + sum(FINGERTIPS.values(), [])}
        self.ik_joints = WAIST_JOINTS + ARM_JOINTS["left"] + ARM_JOINTS["right"]  # 17 dof, waist shared by both hands
        self.ik_qadr = [self.m.joint(j).qposadr[0] for j in self.ik_joints]
        self.ik_dadr = [self.m.joint(j).dofadr[0] for j in self.ik_joints]
        self.ik_range = np.array([self.m.joint(j).range for j in self.ik_joints])
        self.ik_range[:3] = [[-0.6, 0.6], [-0.15, 0.15], [-0.15, 0.35]]  # waist yaw/roll/pitch: keep AMO balanced
        self.ik_w = np.array([IK_WEIGHT.get(j, 1.0) for j in self.ik_joints])
        self.hand_qadr = {s: {j: self.m.joint(j).qposadr[0] for j in self.close_qpos[s]} for s in self.close_qpos}
        self.ikd = mujoco.MjData(self.m)
        cams = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_CAMERA, i) or "" for i in range(self.m.ncam)]
        self.cam_id = next(i for i, nm in enumerate(cams) if nm == LLM_CAM or nm.endswith("/" + LLM_CAM))
        cfg = self.task.sensor_cfgs["head_stereo"]
        self.img_h, self.img_w = int(cfg.height), int(cfg.width)
        self.depth_renderer = mujoco.Renderer(self.m, height=self.img_h, width=self.img_w)
        self.depth_renderer.enable_depth_rendering()
        # RGB recorders are deliberately separate from the depth renderer. They allow headless MuJoCo-only episodes
        # to emit a verification video without starting Isaac Sim.
        self.record_renderers = {}
        fovy = math.radians(self.m.cam_fovy[self.cam_id])
        fy = self.img_h / (2 * math.tan(fovy / 2))
        self.K = np.array([[fy, 0, self.img_w / 2], [0, fy, self.img_h / 2], [0, 0, 1]])
        self.target = self.mj.mj_objects["target"]
        jid = self.m.body_jntadr[self.target.id]
        self.target_qadr, self.target_dadr = int(self.m.jnt_qposadr[jid]), int(self.m.jnt_dofadr[jid])  # free joint
        if self.box_mass:  # primitive targets are built at 0.1 kg; scale mass + inertia to the requested mass
            b = self.target.id
            k = self.box_mass / float(self.m.body_mass[b])
            self.m.body_mass[b] *= k
            self.m.body_inertia[b] *= k
        self.target_geoms = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == self.target.id]
        try:
            chain = [self._body_id(f"right_{j}_link") for j in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]]
            self.arm_reach = float(sum(np.linalg.norm(self.d.xpos[b] - self.d.xpos[a]) for a, b in zip(chain[:-1], chain[1:]))) + 0.03
            self.arm_reach += 0.12  # the automatic waist bend moves the shoulder forward/down by about this much
        except KeyError:
            self.arm_reach = 0.6
        self.waist_pitch_max = min(float(self.ik_range[2][1]), float(self.m.joint("waist_pitch_joint").range[1]))

    @property
    def instruction(self):
        return self._instruction or self.task.instruction

    def dist_to_target(self, p):
        """Distance from a world point to the target's surface (min over its geoms' AABBs, in each geom's frame)."""
        best = float("inf")
        for g in self.target_geoms:
            R, t = self.d.geom_xmat[g].reshape(3, 3), self.d.geom_xpos[g]
            c, h = self.m.geom_aabb[g][:3], self.m.geom_aabb[g][3:]
            q = R.T @ (np.asarray(p) - t) - c
            best = min(best, float(np.linalg.norm(np.maximum(np.abs(q) - h, 0.0))))
        return best

    # ---------------- contacts
    def _is_static(self, b):
        """True for scene geometry welded to the world (table, walls, floor): not the robot and not a free-floating object."""
        root = self.m.body_rootid[b]
        if root == self.m.body_rootid[self.bid["pelvis"]]:
            return False
        if self.m.body_jntnum[root] and self.m.jnt_type[self.m.body_jntadr[root]] == mujoco.mjtJoint.mjJNT_FREE:
            return False
        return True

    def hand_contacts(self):
        """Per hand: {other body name: total normal force N} for contacts involving that hand/wrist/forearm."""
        out = {"left": {}, "right": {}}
        f = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            n1, n2 = self._bname(b1), self._bname(b2)
            for s in ["left", "right"]:
                mine = [n for n in (n1, n2) if f"{s}_hand" in n or f"{s}_wrist" in n or f"{s}_elbow" in n]
                if not mine:
                    continue
                other = n2 if mine[0] == n1 else n1
                ob = b2 if mine[0] == n1 else b1
                if self.m.body_rootid[ob] == self.m.body_rootid[self.bid["pelvis"]]:
                    continue  # contact with the robot's own body: not external
                mujoco.mj_contactForce(self.m, self.d, i, f)
                key = other.split("/")[-1]  # body names stay internal (stage logic, debug); the model never sees them
                out[s][key] = out[s].get(key, 0.0) + float(abs(f[0]))
        return out

    def _bname(self, b):
        return mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or str(b)

    def _record_trace(self):
        q = np.array([self.d.qpos[a] for a in self.ik_qadr])
        dq = float(np.abs(q - self._q_prev).max()) if self._q_prev is not None else 0.0
        self._q_prev = q
        con = self.hand_contacts()
        allc = {}
        f = np.zeros(6)
        for i in range(self.d.ncon):  # any robot-vs-static contact that is not a foot
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            rob = [b for b in (b1, b2) if self.m.body_rootid[b] == self.m.body_rootid[self.bid["pelvis"]]]
            oth = [b for b in (b1, b2) if b not in rob]
            if len(rob) != 1 or not oth or "ankle" in self._bname(rob[0]):
                continue
            mujoco.mj_contactForce(self.m, self.d, i, f)
            k = f"{self._bname(rob[0]).split('/')[-1]}~{self._bname(oth[0]).split('/')[-1]}"
            allc[k] = round(allc.get(k, 0.0) + float(abs(f[0])), 1)
        x, y, yaw = self.base_odom()
        pp, Rp = self.pelvis()
        hp = {s: float(np.linalg.norm(self._hand_cur[s][0] - self.hand_in_pelvis(s)[0])) for s in ["left", "right"]}
        self._trace.append({"step": self.step_i, "odom": [round(x, 3), round(y, 3), round(math.degrees(yaw), 1)],
                            "waist": [round(float(self.d.qpos[a]), 2) for a in self.ik_qadr[:3]], "dq_max": round(dq, 4),
                            "pelvis_tilt_deg": round(math.degrees(math.acos(min(1.0, Rp[2, 2]))), 1),
                            "hand_track_err_cm": {s: round(v * 100, 1) for s, v in hp.items()}, "contacts": allc})

    def debug_trace(self, every=5, last=600):
        t = list(self._trace)[-last:]
        return t[::every]

    def _check_contact_stop(self):
        """Arm force limit: if an arm presses on anything harder than CONTACT_STOP_N, roll its hand target back to the pose
        a few steps earlier (compliance). No knowledge of what was hit is used or reported."""
        con = self.hand_contacts()
        for s in ["left", "right"]:
            if not con[s] or max(con[s].values()) < CONTACT_STOP_N or self._blocked[s]:
                continue
            name = max(con[s], key=con[s].get)
            if self._hand_hist[s]:
                p, R = self._hand_hist[s][0]
                self._hand_goal[s] = (p.copy(), R.copy())
                self._hand_cur[s] = (p.copy(), R.copy())
                pr, rpy = p, mat_to_rpy(R @ self._R0[s].T)
                for k, v in zip(["x", "y", "z"], pr):
                    self.cmd[f"{s}_{k}"] = float(v)
                for k, v in zip(["roll", "pitch", "yaw"], rpy):
                    self.cmd[f"{s}_{k}"] = float(v)
            self._blocked[s] = True
            log(f"FORCE LIMIT: {s} arm pressing on {name} with {con[s][name]:.1f} N; target rolled back")

    # ---------------- frames
    def _body(self, name):
        i = self.bid[name]
        return self.d.xpos[i].copy(), self.d.xmat[i].reshape(3, 3).copy()

    def pelvis(self):
        return self._body("pelvis")

    def base_odom(self):
        p, R = self.pelvis()
        yaw = math.atan2(R[1, 0], R[0, 0])
        d = p[:2] - self.odom_p0
        c, s = math.cos(-self.odom_yaw0), math.sin(-self.odom_yaw0)
        return (c * d[0] - s * d[1], s * d[0] + c * d[1], wrap(yaw - self.odom_yaw0))

    def to_pelvis(self, pw):
        p, R = self.pelvis()
        return R.T @ (np.asarray(pw) - p)

    def hand_in_pelvis(self, side):
        """Hand base position (pelvis frame) and roll/pitch/yaw relative to the reset hand orientation."""
        pp, Rp = self.pelvis()
        hp, Rh = self._body(HAND_BASE[side])
        p = Rp.T @ (hp - pp)
        Rrel = (Rp.T @ Rh) @ self._R0[side].T
        return p, list(mat_to_rpy(Rrel))

    def grasp_point(self, side):
        """Where an object's centre must be for the closing hand to pinch it (fixed in the hand frame); legacy: open-fingertip centroid."""
        if self.legacy_close:
            return np.mean([self._body(n)[0] for n in FINGERTIPS[side]], axis=0)
        hp, Rh = self._body(HAND_BASE[side])
        return hp + Rh @ GRASP_CENTRE[side]

    def target_pos(self):
        return np.array(self.target.xpos)

    def openness_measured(self, side):
        q = self.d.qpos[self.hand_qadr[side][f"{side}_hand_index_1_joint"]]
        c = self.close_qpos[side][f"{side}_hand_index_1_joint"]
        return float(max(0.0, min(1.0, 1.0 - q / c)))

    # ---------------- reset
    def reset(self, seed, record_dir=None):
        if record_dir and not os.path.isabs(record_dir):  # the server's cwd is ~/SIMPLE; clients pass astra-relative paths
            record_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), record_dir)
        self.env.reset(seed=int(seed))
        self._bind()
        self.record_dir, self.frame_i, self.step_i = record_dir, 0, 0
        self.stage = self.max_stage = self._hold_steps = 0
        self.fallen = self.obj_on_floor = False
        self.grasp_events = []
        self.attached = {"left": None, "right": None}
        self._frames, self._last_frames_step = None, -1
        if record_dir:
            for cam in RECORD_CAMS:
                os.makedirs(os.path.join(record_dir, cam), exist_ok=True)
        self.cmd = {"left_hand": 1.0, "right_hand": 1.0, "height": 0.0, "base_x": 0.0, "base_y": 0.0, "base_yaw": 0.0}
        self._q_ik = np.array([self.d.qpos[a] for a in self.ik_qadr])
        self._capture_hands(first=True)
        self._set_odom_origin()
        self.amo_yaw0, self.amo_yaw_off = self.odom_yaw0, 0.0  # AMO measures yaw from its first observation (next step)
        self._height_dirty = True  # the first step must carry a loco_command or AMO is not in the loop
        self.pelvis_z0 = self.pelvis()[0][2]
        self.target_z0 = self.target_pos()[2]
        if self.box_mass:
            self._drop_box_on_floor()
            self.target_z0 = self.target_pos()[2]
        self._run(int(1.0 / CTRL_DT), base_active=False, min_steps=int(1.0 / CTRL_DT))  # AMO settles into standing
        self.fallen = self.obj_on_floor = False  # nothing that happened while settling counts
        self._capture_hands(first=True)
        self._set_odom_origin()
        self.amo_yaw_off = wrap(self.odom_yaw0 - self.amo_yaw0)
        self.pelvis_z0 = self.pelvis()[0][2]
        self.target_z0 = self.target_pos()[2]
        self.task.compute_reward(self.env._get_info(), mujoco_env=self.mj)  # sets the task's lift baseline now
        tp = self.to_pelvis(self.target_pos())
        log(f"reset seed={seed} target(pelvis)=({tp[0]:.2f},{tp[1]:.2f},{tp[2]:.2f}) instruction='{self.instruction}'")
        return {"seed": seed, "instruction": self.instruction, "target_pelvis": tp.tolist()}

    def _drop_box_on_floor(self, ahead=0.55, side=None):
        """Floor-box scenes: the task spawns the target on its 'ground' plane (the tabletop level here); move it onto the real
        floor `ahead` metres in front of the pelvis, found with a downward ray, and let physics settle it."""
        pp, Rp = self.pelvis()
        yaw = math.atan2(Rp[1, 0], Rp[0, 0])
        side = getattr(self, "drop_side", 0.0) if side is None else side
        xy = pp[:2] + ahead * np.array([math.cos(yaw), math.sin(yaw)]) + side * np.array([-math.sin(yaw), math.cos(yaw)])
        start = np.array([xy[0], xy[1], pp[2] + 0.3])
        geomid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(self.m, self.d, start, np.array([0.0, 0.0, -1.0]), None, 1, self.target.id, geomid)
        floor_z = start[2] - dist if dist >= 0 else pp[2] - 0.75
        dz = 2 * float(self.m.geom_size[self.target_geoms[0]][2])
        a = self.target_qadr
        self.d.qpos[a:a + 3] = [xy[0], xy[1], floor_z + dz / 2 + 0.01]
        self.d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.d.qvel[self.target_dadr:self.target_dadr + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)
        log(f"floor box placed at world ({xy[0]:.2f}, {xy[1]:.2f}) on floor z={floor_z:.2f} (hit geom {geomid[0]})")

    def _set_odom_origin(self):
        p, R = self.pelvis()
        self.odom_p0 = p[:2].copy()
        self.odom_yaw0 = math.atan2(R[1, 0], R[0, 0])

    def _capture_hands(self, first=False):
        pp, Rp = self.pelvis()
        for s in ["left", "right"]:
            hp, Rh = self._body(HAND_BASE[s])
            p, R = Rp.T @ (hp - pp), Rp.T @ Rh
            if first:
                self._R0[s] = R.copy()
            self._hand_goal[s] = (p.copy(), R.copy())
            self._hand_cur[s] = (p.copy(), R.copy())
            for k, v in zip(["x", "y", "z"], p):
                self.cmd[f"{s}_{k}"] = float(v)
            for k, v in zip(["roll", "pitch", "yaw"], mat_to_rpy(R @ self._R0[s].T)):
                self.cmd[f"{s}_{k}"] = float(v)

    # ---------------- IK: weighted damped least squares for both hand bases at once over waist + both arms (17 dof),
    # on a scratch MjData in the world frame; the waist is used only when the arms cannot reach (low weights + a
    # nullspace pull back to the upright waist).
    def _ik(self):
        pp, Rp = self.pelvis()
        goals = {s: (Rp @ self._hand_cur[s][0] + pp, Rp @ self._hand_cur[s][1]) for s in ["left", "right"]}
        m, d = self.m, self.ikd
        d.qpos[:] = self.d.qpos
        q = self._q_ik.copy()  # position control: the solver integrates the commanded pose, so pressing on something builds force
        w = self.ik_w
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(12):
            d.qpos[self.ik_qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e, rows = [], []
            for s in ["left", "right"]:
                bid = self.bid[HAND_BASE[s]]
                p_w, R_w = goals[s]
                e.append(np.concatenate([p_w - d.xpos[bid], 0.5 * mat_to_rotvec(R_w @ d.xmat[bid].reshape(3, 3).T)]))
                mujoco.mj_jacBody(m, d, jacp, jacr, bid)
                rows.append(np.vstack([jacp[:, self.ik_dadr], jacr[:, self.ik_dadr]]))
            e = np.concatenate(e)
            if np.linalg.norm(e[[0, 1, 2, 6, 7, 8]]) < 1e-3 and np.linalg.norm(e[[3, 4, 5, 9, 10, 11]]) < 2e-3:
                break
            Jw = np.vstack(rows) * w  # column-weighted jacobian
            Jw_pinv = Jw.T @ np.linalg.solve(Jw @ Jw.T + 0.02 * np.eye(12), np.eye(12))
            dq_w = Jw_pinv @ e
            null = np.eye(len(q)) - Jw_pinv @ Jw
            dq_w += null @ (-0.2 * q * (w < 1.0))  # nullspace: keep the waist upright
            dq = w * dq_w
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.ik_range[:, 0], self.ik_range[:, 1])
        self._q_ik = q
        return dict(zip(self.ik_joints, q.tolist()))

    # ---------------- stepping
    def _advance_hands(self):
        done = True
        for s in ["left", "right"]:
            p, R = self._hand_cur[s]
            self._hand_hist[s].append((p.copy(), R.copy()))
            gp, gR = self._hand_goal[s]
            dp = gp - p
            n = np.linalg.norm(dp)
            step = HAND_SPEED * CTRL_DT
            p = gp.copy() if n <= step else p + dp / n * step
            rv = mat_to_rotvec(gR @ R.T)
            ang = np.linalg.norm(rv)
            astep = HAND_ANG_SPEED * CTRL_DT
            if ang <= astep:
                R = gR.copy()
            else:
                Rd = np.zeros(9)
                qd = np.zeros(4)
                mujoco.mju_axisAngle2Quat(qd, rv / ang, astep)
                mujoco.mju_quat2Mat(Rd, qd)
                R = Rd.reshape(3, 3) @ R
            if n > step or ang > astep:
                done = False
            self._hand_cur[s] = (p, R)
        return done

    def _apply(self, base_cmd, base_active):
        target = self._ik()  # waist first (the robot reads the first 3 keys as the waist override), then both arms
        for s in ["left", "right"]:
            o = self.cmd.get(f"{s}_hand", 1.0)
            target.update({j: (1.0 - o) * v for j, v in self.close_qpos[s].items()})
        self.mj.apply_action(ActionCmd("move_qpos", target_qpos=target))
        if base_active or self._height_dirty:
            vx, vy, yaw_t, turning = base_cmd
            self.mj.apply_action(ActionCmd("loco_command", command=[vx, yaw_t, vy, AMO_HEIGHT_GAIN * self.cmd["height"], 0.0, -0.15, 0.0, turning]))
            self._height_dirty = False

    def _base_cmd(self):
        x, y, yaw = self.base_odom()
        ex, ey = self.cmd["base_x"] - x, self.cmd["base_y"] - y
        eyaw = wrap(self.cmd["base_yaw"] - yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        exb, eyb = c * ex + s * ey, -s * ex + c * ey
        dist = math.hypot(ex, ey)
        yaw_t = wrap(self.cmd["base_yaw"] + self.amo_yaw_off)  # AMO tracks an absolute yaw in its own reset frame
        if dist > BASE_STOP_XY:
            mag = min(V_MAX, max(0.15, KP_XY * dist))
            return (exb / dist * mag, eyb / dist * mag, yaw_t, 0.0), False
        if abs(eyaw) > YAW_STOP:
            return (0.0, 0.0, yaw_t, 1.0), False
        return (0.0, 0.0, yaw_t, 0.0), True

    def _step(self, base_cmd, base_active):
        self._apply(base_cmd, base_active)
        self.mj.step(render=False)
        self.step_i += 1
        self._check_contact_stop()
        self._record_trace()
        if self.sticky_grasp:
            self._update_sticky()
        self._update_stage()
        if self.pelvis()[1][2, 2] < 0.4 or self.pelvis()[0][2] < self.pelvis_z0 - 0.6:  # tilted > ~65 deg or on the floor
            self.fallen = True
        if self.target_pos()[2] < self.target_z0 - 0.15:
            self.obj_on_floor = True
        if self.record_dir and self.step_i % self.record_every == 0:
            fr = self._render()
            for cam, key in RECORD_CAMS.items():
                if key not in fr:
                    continue
                cv2.imwrite(os.path.join(self.record_dir, cam, f"frame_{self.frame_i:05d}.jpg"),
                            cv2.cvtColor(fr[key], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.frame_i += 1

    def _render(self):
        """Render recorder frames from the current MuJoCo state, without requiring Isaac Sim."""
        if self._last_frames_step == self.step_i:
            return self._frames
        frames = {}
        for key in set(RECORD_CAMS.values()):
            try:
                renderer = self.record_renderers.get(key)
                if renderer is None:
                    renderer = mujoco.Renderer(self.m, height=self.img_h, width=self.img_w)
                    self.record_renderers[key] = renderer
                renderer.update_scene(self.d, camera=key)
                frames[key] = renderer.render().copy()
            except (mujoco.FatalError, ValueError):
                # Some tasks omit optional cameras; the recorder simply skips them.
                continue
        self._frames = frames
        self._last_frames_step = self.step_i
        return frames

    def _run(self, max_steps, base_active, min_steps=0):
        reached = not base_active
        for i in range(max_steps):
            cmd = (0.0, 0.0, wrap(self.cmd["base_yaw"] + self.amo_yaw_off), 0.0)
            if base_active:
                cmd, reached = self._base_cmd()
            hands_done = self._advance_hands()
            self._step(cmd, base_active)
            if hands_done and reached and i + 1 >= min_steps:
                return i + 1, reached
            if self.fallen:
                return i + 1, reached
        return max_steps, reached

    # ---------------- optional sticky grasp (privileged): weld the target to a closed hand near it
    def _update_sticky(self):
        tp = self.target_pos()
        for s in ["left", "right"]:
            closed = self.cmd.get(f"{s}_hand", 1.0) < 0.5
            hp, Rh = self._body(HAND_BASE[s])
            if self.attached[s] is None and closed and self.attached["left" if s == "right" else "right"] is None:
                d = float(np.linalg.norm(self.grasp_point(s) - tp))
                if d <= STICKY_DIST:
                    Rt = np.zeros(9)
                    mujoco.mju_quat2Mat(Rt, np.array(self.target.xquat))
                    self.attached[s] = (Rh.T @ (tp - hp), Rh.T @ Rt.reshape(3, 3))
                    self.grasp_events.append({"step": self.step_i, "hand": s, "event": "attach", "dist": d})
                    log(f"STICKY GRASP: {s} hand closed {d * 100:.1f} cm from the target -> attached")
            elif self.attached[s] is not None and not closed:
                self.attached[s] = None
                self.grasp_events.append({"step": self.step_i, "hand": s, "event": "release"})
                log(f"STICKY GRASP: {s} hand opened -> released")
            if self.attached[s] is not None:
                op, oR = self.attached[s]
                q = np.zeros(4)
                mujoco.mju_mat2Quat(q, (Rh @ oR).reshape(-1))
                a = self.target_qadr
                self.d.qpos[a:a + 3] = hp + Rh @ op
                self.d.qpos[a + 3:a + 7] = q
                self.d.qvel[self.target_dadr:self.target_dadr + 6] = 0.0
                mujoco.mj_forward(self.m, self.d)

    # ---------------- scoring (privileged): stages for the pick tasks
    def _update_stage(self):
        tp = self.target_pos()
        lift = tp[2] - self.target_z0
        dists = {s: self.dist_to_target(self.grasp_point(s)) for s in ["left", "right"]}
        tname = self._bname(self.target.id).split("/")[-1]
        touching = any(k.split(":")[-1] == tname for c in self.hand_contacts().values() for k in c)  # any hand link on the target
        near = min(dists.values()) < 0.10 or touching
        if self.max_stage < 1 and min(dists.values()) < 0.06:
            self.max_stage = 1
        if self.max_stage < 2 and near and lift > 0.02:
            self.max_stage = 2
        if self.max_stage < 3 and near and lift > 0.10:
            self.max_stage = 3
        self._hold_steps = self._hold_steps + 1 if (near and lift > 0.10) else 0
        if self.max_stage >= 3 and self._hold_steps >= int(1.0 / CTRL_DT):
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
        hand_named = any(not n.startswith("base_") and not n.endswith("_hand") and n != "height" for n in targets)
        vals = {}
        for n, v in targets.items():
            lo, hi = BOUNDS[n]
            v = float(v)
            if n.endswith("_yaw") or n.endswith("_roll"):
                v = wrap(v)
            cv = min(hi, max(lo, v))
            if cv != v:
                clipped.append(f"{n}={v:.3f}->{cv:.3f}")
            vals[n] = cv
        self._height_dirty = "height" in vals and vals["height"] != self.cmd.get("height", 0.0)
        self._blocked = {"left": None, "right": None}
        self.cmd.update(vals)
        for s in ["left", "right"]:
            if any(n.startswith(s + "_") and not n.endswith("_hand") for n in vals):
                p = np.array([self.cmd[f"{s}_x"], self.cmd[f"{s}_y"], self.cmd[f"{s}_z"]])
                sh = self.to_pelvis(self.d.xpos[self._body_id(f"{s}_shoulder_pitch_link")])
                dist = float(np.linalg.norm(p - sh))
                if dist > self.arm_reach:  # project onto the reachable envelope instead of letting the IK return a wrong pose
                    p = sh + (p - sh) * (self.arm_reach / dist)
                    for k, v in zip(["x", "y", "z"], p):
                        self.cmd[f"{s}_{k}"] = float(v)
                    clipped.append(f"{s} hand target was {100 * (dist - self.arm_reach):.0f} cm beyond the arm's reach: moved to the nearest reachable point "
                                   f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
                R = rpy_to_mat(self.cmd[f"{s}_roll"], self.cmd[f"{s}_pitch"], self.cmd[f"{s}_yaw"]) @ self._R0[s]
                self._hand_goal[s] = (p, R)
        x0, y0, yaw0 = self.base_odom()
        t0 = time.time()
        n_max = int(MOVE_TIMEOUT_S / CTRL_DT)
        steps, reached = self._run(n_max, base_active=base_named, min_steps=int(2.0 / CTRL_DT) if "height" in vals else 0)
        timed_out = steps >= n_max
        for _ in range(int(SETTLE_S / CTRL_DT)):
            self._advance_hands()
            self._step((0.0, 0.0, wrap(self.cmd["base_yaw"] + self.amo_yaw_off), 0.0), False)
        err = {}
        bx, by, byaw = self.base_odom()
        for n, v in vals.items():
            if n == "base_x":
                err[n] = f"{v - bx:+.3f}m"
            elif n == "base_y":
                err[n] = f"{v - by:+.3f}m"
            elif n == "base_yaw":
                err[n] = f"{math.degrees(wrap(v - byaw)):+.1f}deg"
            elif n == "height":
                err[n] = f"{v - (self.pelvis()[0][2] - self.pelvis_z0):+.3f}m"
            elif n.endswith("_hand"):
                err[n] = f"{v - self.openness_measured(n[:-5]):+.2f}"
            else:
                s, k = n.split("_", 1)
                p, rpy = self.hand_in_pelvis(s)
                actual = dict(zip(["x", "y", "z", "roll", "pitch", "yaw"], list(p) + rpy))[k]
                err[n] = f"{v - actual:+.3f}m" if k in "xyz" else f"{math.degrees(wrap(v - actual)):+.1f}deg"
        # every controlled dimension, not only the named ones: what a target did not reach is reported regardless of what was asked
        missed = []
        for s in ["left", "right"]:
            p, rpy = self.hand_in_pelvis(s)
            actual = dict(zip(["x", "y", "z", "roll", "pitch", "yaw"], list(p) + rpy))
            for k in ["x", "y", "z"]:
                e = self.cmd[f"{s}_{k}"] - actual[k]
                if abs(e) > 0.02:
                    missed.append(f"{s}_{k} {e:+.2f}m")
            for k in ["roll", "pitch", "yaw"]:
                e = math.degrees(wrap(self.cmd[f"{s}_{k}"] - actual[k]))
                if abs(e) > 10:
                    missed.append(f"{s}_{k} {e:+.0f}deg")
            eo = self.cmd.get(f"{s}_hand", 1.0) - self.openness_measured(s)
            if abs(eo) > 0.15:
                missed.append(f"{s}_hand {eo:+.2f} (fingers stopped short of the commanded opening)")
        eh = self.cmd.get("height", 0.0) - (self.pelvis()[0][2] - self.pelvis_z0)
        if abs(eh) > 0.03:
            missed.append(f"height {eh:+.2f}m")
        moved = math.hypot(bx - x0, by - y0)
        parts = ["commanded: " + " ".join(f"{n}={v:.3f}" for n, v in vals.items()),
                 f"motion {'TIMED OUT' if timed_out else 'completed'} after {steps * CTRL_DT:.1f}s sim ({time.time() - t0:.1f}s wall)",
                 "final error: " + " ".join(f"{n}={e}" for n, e in err.items()),
                 "NOT REACHED (commanded minus actual, all dimensions): " + (", ".join(missed) if missed else "none"),
                 "clipped: " + (", ".join(clipped) if clipped else "none")]
        if base_named and moved < 0.03 and abs(wrap(byaw - yaw0)) < math.radians(3):
            parts.append("base did not move" + (" (the walking controller does not step at this body height)" if self.cmd.get("height", 0.0) < -0.15 else ""))
        grasp = [e for e in self.grasp_events if e["step"] > self.step_i - steps - int(SETTLE_S / CTRL_DT)]
        if grasp:
            parts.append("sticky grasp: " + ", ".join(f"{g['hand']} {g['event']}" for g in grasp))
        for s in ["left", "right"]:
            if self._blocked[s]:
                parts.append(f"{s.upper()} ARM STOPPED: its contact force exceeded the arm's limit; the hand backed off toward its previous pose "
                             "and its target was not reached")
        touching = self._touching_text()
        if touching:
            parts.append(touching)
        if base_named and hand_named:
            parts.append("note: base and hands were commanded together")
        if self.fallen:
            parts.append("ROBOT FELL")
        return {"text": " | ".join(parts), "fallen": self.fallen, "box_on_floor": self.obj_on_floor, "timed_out": timed_out,
                "clipped": clipped, "grasp_events": grasp, "sim_time": self.step_i * CTRL_DT}

    def step_axes(self, body_action, right_axes, gripper_action, duration_s=0.2):
        """Apply three finite signed hand increments plus body/gripper commands in one bounded tick.

        This deliberately exposes directions rather than target coordinates: each hand axis is exactly -1, 0, or +1,
        corresponding to a server-defined 4 cm increment. The server remains responsible for workspace/reach clipping.
        """
        body_action, gripper_action = str(body_action), str(gripper_action)
        body_valid = {"crouch", "stand", "base_forward", "base_back", "base_left", "base_right",
                      "turn_left", "turn_right", "hold"}
        if body_action not in body_valid or gripper_action not in {"open", "hold", "close"}:
            raise ValueError(f"invalid axis actions body={body_action!r}, gripper={gripper_action!r}")
        if not isinstance(right_axes, (list, tuple)) or len(right_axes) != 3:
            raise ValueError("right_axes must be a length-three list of -1, 0, or 1")
        try:
            right_axes = [int(v) for v in right_axes]
        except (TypeError, ValueError) as exc:
            raise ValueError("right_axes must contain integers") from exc
        if any(v not in {-1, 0, 1} for v in right_axes):
            raise ValueError("right_axes must contain only -1, 0, or 1")
        duration_s = float(duration_s)
        if not 0.04 <= duration_s <= 0.50:
            raise ValueError("duration_s must be between 0.04 and 0.50 seconds")
        if self.fallen:
            return {"body_action": body_action, "right_axes": right_axes, "gripper_action": gripper_action,
                    "fallen": True, "box_on_floor": self.obj_on_floor, "sim_time": self.step_i * CTRL_DT}

        self._blocked = {"left": None, "right": None}
        self._height_dirty = False
        _, _, yaw = self.base_odom()
        base_cmd, base_active = (0.0, 0.0, wrap(yaw + self.amo_yaw_off), 0.0), False

        if any(right_axes):
            p, rpy = self.hand_in_pelvis("right")
            p = np.array(p) + 0.04 * np.array(right_axes, dtype=float)
            for k, v in zip(["x", "y", "z"], p):
                lo, hi = BOUNDS[f"right_{k}"]
                self.cmd[f"right_{k}"] = float(min(hi, max(lo, v)))
            shoulder = self.to_pelvis(self.d.xpos[self._body_id("right_shoulder_pitch_link")])
            target = np.array([self.cmd[f"right_{k}"] for k in ["x", "y", "z"]])
            reach = float(np.linalg.norm(target - shoulder))
            if reach > self.arm_reach:
                target = shoulder + (target - shoulder) * (self.arm_reach / reach)
                for k, v in zip(["x", "y", "z"], target):
                    self.cmd[f"right_{k}"] = float(v)
            R = rpy_to_mat(self.cmd["right_roll"], self.cmd["right_pitch"], self.cmd["right_yaw"]) @ self._R0["right"]
            self._hand_goal["right"] = (target, R)
        if gripper_action == "open":
            self.cmd["right_hand"] = 1.0
        elif gripper_action == "close":
            self.cmd["right_hand"] = 0.0

        if body_action == "crouch":
            self.cmd["height"] = BOUNDS["height"][0]
            self._height_dirty = True
        elif body_action == "stand":
            self.cmd["height"] = 0.0
            self._height_dirty = True
        elif body_action in {"base_forward", "base_back", "base_left", "base_right"}:
            vx, vy = {"base_forward": (0.22, 0.0), "base_back": (-0.18, 0.0),
                      "base_left": (0.0, 0.18), "base_right": (0.0, -0.18)}[body_action]
            base_cmd, base_active = (vx, vy, wrap(yaw + self.amo_yaw_off), 0.0), True
        elif body_action in {"turn_left", "turn_right"}:
            sign = 1.0 if body_action == "turn_left" else -1.0
            base_cmd, base_active = (0.0, 0.0, wrap(yaw + sign * 0.35 + self.amo_yaw_off), 1.0), True

        start_step, start_events = self.step_i, len(self.grasp_events)
        for _ in range(max(2, round(duration_s / CTRL_DT))):
            self._advance_hands()
            self._step(base_cmd, base_active)
            if self.fallen:
                break
        return {"body_action": body_action, "right_axes": right_axes, "gripper_action": gripper_action,
                "duration_s": duration_s, "control_steps": self.step_i - start_step, "fallen": self.fallen,
                "box_on_floor": self.obj_on_floor, "blocked": dict(self._blocked),
                "grasp_events": self.grasp_events[start_events:], "stage": self.max_stage,
                "task_success": bool(self.task.check_success(self.env._get_info(), mujoco_env=self.mj)),
                "sim_time": self.step_i * CTRL_DT}

    # ---------------- camera geometry (MuJoCo camera of the same name/intrinsics as the Isaac one)
    def _cam_Rt(self):
        """Camera-to-world rotation in OpenCV axes (+z forward, +x right, +y down) and position."""
        R_mj = self.d.cam_xmat[self.cam_id].reshape(3, 3)  # MuJoCo camera: looks along -z, +y up
        return R_mj @ np.diag([1.0, -1.0, -1.0]), self.d.cam_xpos[self.cam_id].copy()

    def _project(self, P):
        R, t = self._cam_Rt()
        pc = (np.asarray(P) - t) @ R
        z = np.maximum(pc[:, 2], 1e-6)
        u, v = self.K[0, 0] * pc[:, 0] / z + self.K[0, 2], self.K[1, 1] * pc[:, 1] / z + self.K[1, 2]
        ok = (pc[:, 2] > 0.03) & (u >= 0) & (u < self.img_w - 1) & (v >= 0) & (v < self.img_h - 1)
        return u, v, ok

    def _annotate(self, img_bgr):
        for s in ["left", "right"]:
            pts = np.stack([self.grasp_point(s)] + [self._body(n)[0] for n in FINGERTIPS[s]])
            u, v, ok = self._project(pts)
            col = MARK_COLOR[s]
            for i in range(1, len(pts)):
                if ok[i]:
                    cv2.circle(img_bgr, (int(u[i]), int(v[i])), 3, col, -1)
            if ok[0]:
                cv2.circle(img_bgr, (int(u[0]), int(v[0])), 8, col, 2)
                cv2.putText(img_bgr, s[0].upper(), (int(u[0]) + 10, int(v[0]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        return img_bgr

    def measure(self, camera, x, y, label=""):
        if camera != "head":
            raise ValueError(f"unknown camera {camera}; use head")
        x, y = int(round(float(x))), int(round(float(y)))
        if not (0 <= x < self.img_w and 0 <= y < self.img_h):
            raise ValueError(f"pixel ({x}, {y}) outside the {self.img_w}x{self.img_h} image")
        self.depth_renderer.update_scene(self.d, camera=LLM_CAM)
        depth = self.depth_renderer.render()
        win = depth[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        win = win[np.isfinite(win) & (win > 0.02) & (win < 9.0)]
        if win.size == 0:
            return {"text": f"measure({camera}, {x}, {y}): nothing there (background)", "point": None}
        z = float(np.median(win))
        R, t = self._cam_Rt()
        pc = np.array([(x - self.K[0, 2]) * z / self.K[0, 0], (y - self.K[1, 2]) * z / self.K[1, 1], z])
        pt = self.to_pelvis(R @ pc + t)
        parts = [f"measure({camera}, {x}, {y}){' ' + label if label else ''}: point at pelvis-frame x={pt[0]:.2f} y={pt[1]:.2f} z={pt[2]:.2f} "
                 f"({z:.2f} m from the camera)"]
        for s in ["left", "right"]:
            d = pt - self.to_pelvis(self.grasp_point(s))
            parts.append(f"{s} grasp point is {abs(d[0]) * 100:.0f} cm {'short of' if d[0] > 0 else 'beyond'} it, "
                         f"{abs(d[1]) * 100:.0f} cm to the {'right' if d[1] > 0 else 'left'} of it, "
                         f"{abs(d[2]) * 100:.0f} cm {'below' if d[2] > 0 else 'above'} it (straight-line {np.linalg.norm(d) * 100:.0f} cm)")
        return {"text": "; ".join(parts), "point": pt.tolist()}

    def _touching_text(self):
        """What a torque-sensing arm can tell: that there is external contact on an arm, not what it touches."""
        con = self.hand_contacts()
        arms = [s for s, c in con.items() if any(v > CONTACT_REPORT_N for v in c.values())]
        return ("external contact: " + ", ".join(f"{s} arm" for s in arms)) if arms else ""

    def get_observation(self, include_third_person=False, marks=True, body_map=False):
        fr = self._render()
        images = {}
        for name, key in [("head", LLM_CAM)] + ([("third_person", THIRD_CAM)] if include_third_person else []):
            img = cv2.cvtColor(fr[key], cv2.COLOR_RGB2BGR)
            if marks and name == "head":
                img = self._annotate(img)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            images[name] = base64.b64encode(buf.tobytes()).decode()
        bx, by, byaw = self.base_odom()
        text = {}
        for s in ["left", "right"]:
            p, rpy = self.hand_in_pelvis(s)
            text.update({f"{s}_x": p[0], f"{s}_y": p[1], f"{s}_z": p[2], f"{s}_roll": rpy[0], f"{s}_pitch": rpy[1], f"{s}_yaw": rpy[2],
                         f"{s}_hand": self.openness_measured(s)})
            gp = self.to_pelvis(self.grasp_point(s))
            text.update({f"{s}_grasp_x": gp[0], f"{s}_grasp_y": gp[1], f"{s}_grasp_z": gp[2]})
        text.update({"base_x": bx, "base_y": by, "base_yaw": byaw, "height": self.pelvis()[0][2] - self.pelvis_z0})
        touching = self._touching_text()
        return {"images": images, "image_size": [self.img_w, self.img_h],
                "state_text": " ".join(f"{k}={v:.2f}" for k, v in text.items()) + (f" | {touching}" if touching else ""),
                "state": text, "sim_time": self.step_i * CTRL_DT, "instruction": self.instruction}

    def get_state(self):
        tp = self.target_pos()
        pp, Rp = self.pelvis()
        contacts = self.hand_contacts()
        target_name = self._bname(self.target.id).split("/")[-1]
        right_target_contact = any(key.split(":")[-1] == target_name for key in contacts["right"])
        Rt = np.zeros(9)
        mujoco.mju_quat2Mat(Rt, np.array(self.target.xquat))
        Rrel = Rp.T @ Rt.reshape(3, 3)
        st = {"sim_time": self.step_i * CTRL_DT, "step": self.step_i, "instruction": self.instruction,
              "box": {"pos": tp.tolist(), "quat": list(map(float, self.target.xquat)), "pelvis": self.to_pelvis(tp).tolist(),
                      "yaw_pelvis": math.atan2(Rrel[1, 0], Rrel[0, 0]), "lift": float(tp[2] - self.target_z0)},
              "waist": dict(zip(WAIST_JOINTS, [float(self.d.qpos[a]) for a in self.ik_qadr[:3]])),
              "pelvis": {"pos": pp.tolist(), "odom": list(self.base_odom()), "height": float(pp[2] - self.pelvis_z0)},
              "hands": {}, "commanded": dict(self.cmd), "attached": {s: self.attached[s] is not None for s in self.attached},
              "grasp_events": self.grasp_events, "sticky_grasp": self.sticky_grasp,
              "right_target_contact": right_target_contact, "right_contact_forces": contacts["right"],
              "stage": self.stage, "max_stage": self.max_stage, "fallen": self.fallen, "box_on_floor": self.obj_on_floor,
              "task_success": bool(self.task.check_success(self.env._get_info(), mujoco_env=self.mj))}
        for s in ["left", "right"]:
            gp = self.grasp_point(s)
            p, rpy = self.hand_in_pelvis(s)
            st["hands"][s] = {"wrist_pelvis_pos": p.tolist(), "wrist_pelvis_rpy": rpy, "grasp_point_pelvis": self.to_pelvis(gp).tolist(),
                              "box_surface_dist": self.dist_to_target(gp), "openness_measured": self.openness_measured(s),
                              "fingertips_pelvis": {n.split("_hand_")[1]: self.to_pelvis(self._body(n)[0]).tolist() for n in FINGERTIPS[s]}}
        return st

    # ---------------- debug-only (privileged; scripted grasp calibration, never exposed to the LLM)
    def _body_id(self, n):
        for i in range(self.m.nbody):
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if nm == n or nm.endswith("/" + n):
                return i
        raise KeyError(n)

    def debug_hand_geometry(self, side="right", roll=0.0, pitch=0.0, yaw=0.0, openness=(1.0, 0.75, 0.5, 0.25, 0.0)):
        """Fingertip and palm positions relative to the hand base, in pelvis axes, for a hand commanded to (roll, pitch, yaw),
        at several openness levels (forward kinematics on a scratch MjData; the robot does not move)."""
        d = mujoco.MjData(self.m)
        d.qpos[:] = self.d.qpos
        Rcmd = rpy_to_mat(roll, pitch, yaw) @ self._R0[side]  # hand-base orientation in the pelvis frame
        names = {"thumb": FINGERTIPS[side][0], "index": FINGERTIPS[side][1], "middle": FINGERTIPS[side][2],
                 "thumb_base": f"{side}_hand_thumb_0_link", "index_base": f"{side}_hand_index_0_link"}
        ids = {k: self._body_id(v) for k, v in names.items()}
        hb = self.bid[HAND_BASE[side]]
        out = {}
        for o in openness:
            for j, v in self.close_qpos[side].items():
                d.qpos[self.hand_qadr[side][j]] = (1.0 - o) * v
            mujoco.mj_kinematics(self.m, d)
            Rh, ph = d.xmat[hb].reshape(3, 3), d.xpos[hb]
            out[f"{o:.2f}"] = {k: (Rcmd @ (Rh.T @ (d.xpos[i] - ph))).round(4).tolist() for k, i in ids.items()}
        return out

    def debug_contacts(self):
        """Active contacts between the robot and anything else (feet excluded), with the total normal force per pair."""
        out = {}
        names = lambda b: mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or str(b)  # noqa: E731
        f = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            n1, n2 = names(b1), names(b2)
            robot = [n for n in (n1, n2) if any(k in n for k in ("hand", "wrist", "elbow", "shoulder", "torso", "waist", "pelvis", "hip", "knee"))]
            if not robot or ("ankle" in n1 or "ankle" in n2):
                continue
            mujoco.mj_contactForce(self.m, self.d, i, f)
            key = f"{n1} <-> {n2}"
            out[key] = round(out.get(key, 0.0) + float(abs(f[0])), 2)
        return out

    def hand_geometry(self, side="right", roll=0.0, pitch=0.0, yaw=0.0):
        """LLM tool: where the fingertips and the grasp point sit relative to the hand-base position for a commanded orientation."""
        side = side if side in ("left", "right") else "right"
        g = self.debug_hand_geometry(side, float(roll), float(pitch), float(yaw), openness=(1.0, 0.0))
        Rcmd = rpy_to_mat(float(roll), float(pitch), float(yaw)) @ self._R0[side]
        gp = Rcmd @ GRASP_CENTRE[side]
        fmt = lambda v: f"({v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f})"  # noqa: E731
        o, c = g["1.00"], g["0.00"]
        lowest = min([("thumb", o["thumb"][2]), ("index", o["index"][2]), ("middle", o["middle"][2])], key=lambda t: t[1])
        text = (f"{side} hand at roll={float(roll):.2f} pitch={float(pitch):.2f} yaw={float(yaw):.2f}; offsets from the hand-base position "
                f"(pelvis axes, metres): grasp point {fmt(gp)}; open fingertips thumb {fmt(o['thumb'])}, index {fmt(o['index'])}, "
                f"middle {fmt(o['middle'])}; closed fingertips thumb {fmt(c['thumb'])}, index {fmt(c['index'])}, middle {fmt(c['middle'])}. "
                f"Lowest open fingertip: {lowest[0]} at {lowest[1]:+.2f} m ({(gp[2] - lowest[1]) * 100:.0f} cm below the grasp point).")
        return {"text": text, "grasp_point_offset": gp.round(3).tolist()}

    def _hand_text(self):
        """Embodiment facts about the hand, from forward kinematics of the right hand at the reset orientation. No advice."""
        g = self.debug_hand_geometry("right", 0.0, 0.0, 0.0, openness=(1.0, 0.0))
        o, c = g["1.00"], g["0.00"]
        gp = self._R0["right"] @ GRASP_CENTRE["right"]  # hand frame -> pelvis axes, like the fingertip offsets
        cm = lambda v: f"({v[0] * 100:+.0f}, {v[1] * 100:+.0f}, {v[2] * 100:+.0f})"  # noqa: E731
        ahead = (o["index"][0] - gp[0]) * 100
        below = (gp[2] - min(o["thumb"][2], o["index"][2], o["middle"][2])) * 100
        span = float(np.linalg.norm(np.array(o["thumb"]) - (np.array(o["index"]) + np.array(o["middle"])) / 2)) * 100
        return [
            "Hand geometry (three-finger hand; numbers are for the RIGHT hand at roll=pitch=yaw=0 as offsets in cm from the hand-base",
            f"position you command, pelvis axes +x forward +y left +z up; the left hand mirrors y): grasp point {cm(gp)} = where the",
            f"thumb tip and the index/middle tips meet when the hand closes. Open fingertips: thumb {cm(o['thumb'])} (on the inner side),",
            f"index {cm(o['index'])}, middle {cm(o['middle'])}; closed: thumb {cm(c['thumb'])}, index {cm(c['index'])}, middle {cm(c['middle'])}.",
            "Closing moves the thumb forward from behind/inside and the two fingers back from in front, meeting at the grasp point.",
            f"The open fingers reach {ahead:.0f} cm beyond the grasp point, the lowest open fingertip is {below:.0f} cm below it, and the open",
            f"thumb tip is {span:.0f} cm from the fingertips. hand_geometry(side, roll, pitch, yaw) gives the same offsets for any other hand",
            "orientation. An arm whose contact force exceeds the controller's limit is stopped and backed off; move_to says so.",
            "The state line gives each hand's current grasp point (left_grasp_*/right_grasp_*) in the pelvis frame.",
        ] + self._reach_text()

    def _reach_text(self):
        """Arm reach from the model: shoulder position (pelvis frame) and the stretched length of the whole arm chain."""
        try:
            sh = self.to_pelvis(self.d.xpos[self._body_id("right_shoulder_pitch_link")])
        except KeyError:
            return []
        return [f"Arm reach: each shoulder sits at about ({sh[0]:+.2f}, {abs(sh[1]):.2f} to its side, {sh[2]:+.2f}) in the pelvis frame and the wrist can get at most "
                f"~{self.arm_reach:.2f} m from it (including the automatic waist bend). A hand target further away is moved to the nearest "
                "reachable point and move_to says so."]

    def debug_objects(self):
        """Every free-floating object in the scene (target + distractors): pelvis-frame centre of its AABB and its size."""
        out = {}
        for key, obj in self.mj.mj_objects.items():
            try:
                b = obj.id
            except AttributeError:
                continue
            if self.m.body_jntnum[b] < 1 or self.m.jnt_type[self.m.body_jntadr[b]] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            pts = []
            for g in range(self.m.ngeom):
                if self.m.geom_bodyid[g] != b:
                    continue
                c, h = self.m.geom_aabb[g][:3], self.m.geom_aabb[g][3:]
                R, p = self.d.geom_xmat[g].reshape(3, 3), self.d.geom_xpos[g]
                pts += [p + R @ (c + h * np.array(s)) for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)]
            if not pts:
                continue
            P = np.array([self.to_pelvis(q) for q in pts])
            lo, hi = P.min(0), P.max(0)
            out[key] = {"center_pelvis": ((lo + hi) / 2).round(4).tolist(), "size": (hi - lo).round(4).tolist(),
                        "z_world": float(self.d.xpos[b][2]), "name": mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b)}
        return out

    def describe(self, marks=False, body_map=False):
        lines = [
            "Frames: +x forward, +y left, +z up. Hand targets (left_*/right_*) are the hand base pose in the PELVIS frame (origin at",
            "the pelvis, moves with the robot); roll/pitch/yaw in radians relative to the hand orientation at reset (0,0,0 = as at",
            "reset, fingers pointing forward-down). base_x/base_y/base_yaw are in the ODOMETRY frame fixed at the reset pose.",
            f"height: pelvis height change in metres (0 = standing, negative = crouch, down to {BOUNDS['height'][0]:.2f}).",
            f"The torso bends forward automatically when a hand target needs it (waist pitch up to {math.degrees(self.waist_pitch_max):.0f} deg).",
            "Hands: left_hand/right_hand scalar, 0 = closed, 1 = open (three-finger Dex3 hand: thumb opposes index+middle).",
            f"Hand motion speed cap {HAND_SPEED} m/s; walking ~0.3 m/s; a move_to call times out after {MOVE_TIMEOUT_S}s sim.",
            "Bounds: " + ", ".join(f"{n} [{lo:.2f}, {hi:.2f}]" for n, (lo, hi) in BOUNDS.items()),
        ] + ([] if self.legacy_close else self._hand_text())
        extra_tools = [] if self.legacy_close else [{"toolSpec": {"name": "hand_geometry", "description": (
            "Preview a hand orientation: returns the grasp point and the open/closed fingertip offsets from the hand-base position "
            "for the given roll/pitch/yaw (radians, relative to the reset orientation), in pelvis axes. No motion, costs one call."),
            "inputSchema": {"json": {"type": "object", "properties": {
                "side": {"type": "string", "enum": ["left", "right"]},
                "roll": {"type": "number"}, "pitch": {"type": "number"}, "yaw": {"type": "number"}},
                "required": ["side", "roll", "pitch", "yaw"]}}}}]
        if marks:
            lines += ["The head image is annotated from the robot's own kinematics: a circle labelled L (yellow) or R (magenta) marks each",
                      "hand's grasp point (where the thumb and fingertips meet when the hand closes); dots are fingertips."]
        return {"text": "\n".join(lines), "bounds": {k: list(v) for k, v in BOUNDS.items()}, "sticky_grasp": self.sticky_grasp,
                "cameras": ["head"] if self.isaac is not None else [], "image_size": [self.img_w, self.img_h], "instruction": self.instruction,
                "extra_tools": extra_tools}


# ----------------------------------------------------------------------------------------------- server
def make_env_and_adapter(args):
    if args.floor_box:
        from simple.dr.types import Box as DRBox
        import importlib
        _mod = importlib.import_module(f"simple.tasks.{args.task}")  # task modules are named after their uid
        _T = next(c for c in vars(_mod).values() if isinstance(c, type) and getattr(c, "uid", None) == args.task)
        _size = [float(v) for v in args.floor_box.split(",")]
        _T.dr_cfgs["target"].asset_id, _T.dr_cfgs["target"].size = "primitive:cube", _size
        args.target_object = args.target_object or "primitive:cube"  # the task __init__ re-applies its target_object kwarg
        _sp = _T.dr_cfgs["spatial"]
        _sp.obj_surface_map = {"target": "ground"}
        _sp.robot_region = DRBox(low=[-1.3, 0.0, 0.0], high=[-1.3, 0.0, 0.0])
        _sp.target_region = DRBox(low=[-0.82, -0.03], high=[-0.78, 0.03])  # ~0.5 m ahead of the robot, clear of the table edge
        _sp.target_rotate_z = DRBox(low=0.0, high=0.0)
        if "distractors" in _T.dr_cfgs:
            _T.dr_cfgs["distractors"].number_of_distractors = 0
        _T.dr_cfgs["scene"].table_position = DRBox(low=[1.3, 0.0], high=[1.3, 0.0])  # push the table back: floor space in front
        log(f"floor box {_size} m, {args.box_mass} kg replaces the task target")
    log(f"creating env {args.env_id} ...")
    t0 = time.time()
    env = gym.make(args.env_id, task=args.task, robot_uid="g1_wholebody", sim_mode=args.sim_mode, headless=True, render_hz=50,
                   max_episode_steps=10 ** 7, **({"scene_uid": args.scene_uid} if args.scene_uid else {}),
                   **({"target_object": args.target_object} if args.target_object else {})).unwrapped
    adapter = Adapter(env, record_every=args.record_every, sticky_grasp=args.sticky_grasp, legacy_close=args.legacy_close,
                      instruction=args.instruction, box_mass=args.box_mass if args.floor_box else None)
    adapter.reset(seed=0)
    log(f"env + adapter ready in {time.time() - t0:.1f}s; serving on 127.0.0.1:{args.port}")
    return env, adapter


def build_methods(adapter):
    return {"reset": adapter.reset, "move_to": adapter.move_to, "step_axes": adapter.step_axes,
            "get_observation": adapter.get_observation,
            "get_state": adapter.get_state, "describe": adapter.describe, "measure": adapter.measure,
            "debug_hand_geometry": adapter.debug_hand_geometry, "debug_objects": adapter.debug_objects,
            "debug_contacts": adapter.debug_contacts, "debug_trace": adapter.debug_trace, "hand_geometry": adapter.hand_geometry}


METHODS, adapter, args, env = {}, None, None, None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"ok": True, "backend": "simple", "sim_time": adapter.step_i * CTRL_DT, "stage": adapter.max_stage})

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


def main():
    global args, env, adapter, METHODS
    args = build_parser().parse_args()
    env, adapter = make_env_and_adapter(args)
    METHODS = build_methods(adapter)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
