"""SIMPLE backend driven by the decoupled whole-body controller (GR00T-WholeBodyControl ONNX lower body from
third_party/decoupled_wbc) instead of AMO. Same RPC and same Adapter as sim_server_simple.py; only actuation differs:
our IK's waist + arm targets and the finger targets go to the WBC as `target_upper_body_pose`, the base controller
sends `navigate_cmd` = [vx, vy, turning_flag, absolute target yaw], and `height` becomes `base_height_command`
(0.74 m standing, down to 0.24 m). Default task: the carry-box scene (0.23 x 0.38 x 0.43 m box on the floor, table).

  cd ~/SIMPLE && MUJOCO_GL=egl OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python ~/astra/sim_server_wbc.py --port 8765"""

import math
import os
import sys
import time
from http.server import HTTPServer

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402

import sim_server_simple as base  # noqa: E402
from sim_server_simple import BASE_STOP_XY, CTRL_DT, HAND_BASE, KP_XY, V_MAX, YAW_STOP, Adapter, log, rpy_to_mat, wrap  # noqa: E402

from simple.core.action import ActionCmd  # noqa: E402

STAND_HEIGHT = 0.74  # WBC base_height_command when standing; teleop clips it to [0.2, 0.74]
HEIGHT_SPEED = 0.15  # m/s: body height changes are ramped like hand targets, so the arms can follow
base.BOUNDS["height"] = (-0.5, 0.02)
AMO_MJCF = os.path.expanduser("~/SIMPLE/data/robots/g1/g1_29dof_wholebody_dex3.xml")  # has the *_finger_tip bodies
TIP_LINK = {"thumb": "thumb_2_link", "index": "index_1_link", "middle": "middle_1_link"}
base.FINGERTIPS = {s: [f"{s}_hand_{TIP_LINK[f]}" for f in ["thumb", "index", "middle"]] for s in ["left", "right"]}


def tip_offsets():
    """Fingertip position in each distal finger link's frame, taken from the AMO model (constant: fixed bodies)."""
    m = mujoco.MjModel.from_xml_path(AMO_MJCF)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    out = {}
    for s in ["left", "right"]:
        for f, link in TIP_LINK.items():
            a, b = m.body(f"{s}_hand_{link}").id, m.body(f"{s}_hand_{f}_finger_tip").id
            out[f"{s}_hand_{link}"] = d.xmat[a].reshape(3, 3).T @ (d.xpos[b] - d.xpos[a])
    return out


class WbcAdapter(Adapter):
    def __init__(self, env, sonic_config, **kw):
        super().__init__(env, **kw)
        from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
        self.sonic_config = sonic_config
        self.agent = SonicDecoupledWbcAgent(self.robot, sonic_config)
        rm = self.agent._dwbc_robot_model
        idx = rm.get_joint_group_indices("upper_body")
        self.upper_names = [n for n, i in rm.joint_to_dof_index.items() if i in idx]
        self.tip_off = tip_offsets()
        self._wbc_cmd = None
        self._height_cur = 0.0
        log(f"WBC upper-body joints ({len(self.upper_names)}): {self.upper_names[:3]} ... {self.upper_names[-1]}")

    # fingertips = distal link + fixed offset (the Sonic MJCF has no tip bodies)
    def _body(self, name):
        p, R = super()._body(name)
        off = self.tip_off.get(name)
        return (p + R @ off, R) if off is not None else (p, R)

    def debug_hand_geometry(self, side="right", roll=0.0, pitch=0.0, yaw=0.0, openness=(1.0, 0.75, 0.5, 0.25, 0.0)):
        d = mujoco.MjData(self.m)
        d.qpos[:] = self.d.qpos
        Rcmd = rpy_to_mat(roll, pitch, yaw) @ self._R0[side]
        names = {"thumb": base.FINGERTIPS[side][0], "index": base.FINGERTIPS[side][1], "middle": base.FINGERTIPS[side][2],
                 "thumb_base": f"{side}_hand_thumb_0_link", "index_base": f"{side}_hand_index_0_link"}
        ids = {k: self._body_id(v) for k, v in names.items()}
        hb = self.bid[HAND_BASE[side]]
        out = {}
        for o in openness:
            for j, v in self.close_qpos[side].items():
                d.qpos[self.hand_qadr[side][j]] = (1.0 - o) * v
            mujoco.mj_kinematics(self.m, d)
            Rh, ph = d.xmat[hb].reshape(3, 3), d.xpos[hb]
            res = {}
            for k, i in ids.items():
                p = d.xpos[i] + d.xmat[i].reshape(3, 3) @ self.tip_off.get(names[k], np.zeros(3))
                res[k] = (Rcmd @ (Rh.T @ (p - ph))).round(4).tolist()
            out[f"{o:.2f}"] = res
        return out

    def _bind(self):
        super()._bind()
        self.ik_range[:3] = [[-0.8, 0.8], [-0.3, 0.3], [-0.2, 1.0]]  # waist is ours under the WBC: allow a real bend
        self.agent.reset()
        self.agent._wbc_policy.lower_body_policy.use_policy_action = True
        self._height_cur = 0.0

    def _drop_box_on_floor(self, ahead=0.55, side=None):
        if "carry_box" in self.task.uid:
            return  # that task already spawns its box on the floor
        super()._drop_box_on_floor(ahead, side)

    # ---------------- actuation through the WBC
    def _base_cmd(self):
        x, y, yaw = self.base_odom()
        ex, ey = self.cmd["base_x"] - x, self.cmd["base_y"] - y
        eyaw = wrap(self.cmd["base_yaw"] - yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        exb, eyb = c * ex + s * ey, -s * ex + c * ey
        dist = math.hypot(ex, ey)
        yaw_world = wrap(self.odom_yaw0 + self.cmd["base_yaw"])  # the WBC servos an absolute yaw from the base quaternion
        turning = 1.0 if abs(eyaw) > YAW_STOP else 0.0
        if dist > BASE_STOP_XY:
            mag = min(V_MAX, max(0.15, KP_XY * dist))
            return (exb / dist * mag, eyb / dist * mag, yaw_world, turning), False
        if turning:
            return (0.0, 0.0, yaw_world, 1.0), False
        return (0.0, 0.0, yaw_world, 0.0), True

    def _advance_height(self):
        goal = self.cmd.get("height", 0.0)
        step = HEIGHT_SPEED * CTRL_DT
        self._height_cur = goal if abs(goal - self._height_cur) <= step else self._height_cur + math.copysign(step, goal - self._height_cur)
        return self._height_cur == goal

    def _run(self, max_steps, base_active, min_steps=0):
        reached = not base_active
        for i in range(max_steps):
            cmd = (0.0, 0.0, wrap(self.cmd["base_yaw"] + self.amo_yaw_off), 0.0)
            if base_active:
                cmd, reached = self._base_cmd()
            hands_done = self._advance_hands()
            height_done = self._advance_height()
            self._step(cmd, base_active)
            if hands_done and height_done and reached and i + 1 >= min_steps:
                return i + 1, reached
            if self.fallen:
                return i + 1, reached
        return max_steps, reached

    def _apply(self, base_cmd, base_active):
        target = self._ik()  # waist + both arms (17 joints)
        for s in ["left", "right"]:
            o = self.cmd.get(f"{s}_hand", 1.0)
            target.update({j: (1.0 - o) * v for j, v in self.close_qpos[s].items()})
        vx, vy, yaw_t, turning = base_cmd
        if not base_active:
            vx, vy, turning = 0.0, 0.0, 0.0
        t_now = time.monotonic()
        obs = self.agent._build_wbc_observation(self.robot.prepare_obs())
        pol = self.agent._wbc_policy
        pol.set_observation(obs)
        goal = {"target_upper_body_pose": np.array([target[n] for n in self.upper_names], dtype=np.float32),
                "navigate_cmd": np.array([vx, vy, turning, yaw_t], dtype=np.float32),
                "base_height_command": np.array([STAND_HEIGHT + self._height_cur], dtype=np.float32),
                "target_time": t_now + CTRL_DT, "interpolation_garbage_collection_time": t_now - 2 * CTRL_DT, "timestamp": t_now}
        pol.set_goal(goal)
        act = pol.get_action(time=t_now)
        rm = self.agent._dwbc_robot_model
        self._wbc_cmd = ActionCmd("decoupled_wbc", target_q=rm.get_body_actuated_joints(act["q"]),
                                  left_hand_q=rm.get_hand_actuated_joints(act["q"], side="left"),
                                  right_hand_q=rm.get_hand_actuated_joints(act["q"], side="right"))

    def _step(self, base_cmd, base_active):
        self._apply(base_cmd, base_active)
        for _ in range(int(round(CTRL_DT / self.sonic_config["SIMULATE_DT"]))):
            self.mj.apply_action(self._wbc_cmd)
            self.mj.step(render=False)
        self.step_i += 1
        self._check_contact_stop()
        self._record_trace()
        if self.sticky_grasp:
            self._update_sticky()
        self._update_stage()
        if self.pelvis()[1][2, 2] < 0.4 or self.pelvis()[0][2] < self.pelvis_z0 - 0.6:
            self.fallen = True
        if self.target_pos()[2] < self.target_z0 - 0.15:
            self.obj_on_floor = True
        if self.record_dir and self.step_i % self.record_every == 0:
            fr = self._render()
            for cam, key in base.RECORD_CAMS.items():
                if key not in fr:
                    continue
                import cv2
                cv2.imwrite(os.path.join(self.record_dir, cam, f"frame_{self.frame_i:05d}.jpg"),
                            cv2.cvtColor(fr[key], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.frame_i += 1

    def reset(self, seed, record_dir=None):
        out = super().reset(seed, record_dir)
        # extra settling: the WBC's velocity-based stabilisation, then re-take the references
        n = 0
        while not self.robot.stabilized and n < 150:
            self._step((0.0, 0.0, self.odom_yaw0, 0.0), False)
            n += 1
        self._capture_hands()
        self._set_odom_origin()
        self.amo_yaw_off = 0.0
        self.pelvis_z0 = self.pelvis()[0][2]
        self.target_z0 = self.target_pos()[2]
        self.fallen = self.obj_on_floor = False
        self.max_stage = self.stage = self._hold_steps = 0
        tp = self.to_pelvis(self.target_pos())
        log(f"WBC stabilised after {n} extra steps; target(pelvis)=({tp[0]:.2f},{tp[1]:.2f},{tp[2]:.2f})")
        out["target_pelvis"] = tp.tolist()
        return out


def main():
    p = base.build_parser()
    p.set_defaults(env_id="simple/G1WholebodyXMoveBendCarryBoxSonic-v0", task="g1_wholebody_xmove_bend_carry_box_sonic")
    p.add_argument("--table-x", type=float, default=0.9, help="coffee table x (task default 0.3 leaves the box against its edge)")
    p.add_argument("--box-x", type=float, default=-0.55, help="box spawn x (robot at -1.2)")
    p.add_argument("--playground", action="store_true", help="kitchen tabletop scene under the WBC: 6 random objects on the table "
                   "and a --floor-box (default 0.25,0.30,0.35) on the floor to the left; no target, no score")
    args = p.parse_args()
    if args.playground:
        args.env_id, args.task = "simple/G1WholebodyXMoveBendCarryBoxSonic-v0", "g1_wholebody_tabletop_grasp_mp"
        args.floor_box = args.floor_box or "0.25,0.30,0.35"
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    sonic_config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[]).load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    from simple.dr.types import Box as DRBox
    # third-person + side cameras for the recording (the Sonic task only has the head stereo pair)
    from simple.tasks.g1_wholebody_tabletop_grasp_mp import G1WholebodyTabletopGraspMP
    from simple.tasks.g1_wholebody_xmove_bend_carry_box_sonic import G1WholebodyXMoveBendCarryBoxTaskSonic as T
    for cam in ["front_stereo", "side_left"]:
        T.sensor_cfgs[cam] = G1WholebodyTabletopGraspMP.sensor_cfgs[cam]
    T.dr_cfgs["scene"].table_position = DRBox(low=[args.table_x, 0.0], high=[args.table_x, 0.0])
    T.dr_cfgs["spatial"].target_region = DRBox(low=[args.box_x - 0.12, -0.18], high=[args.box_x + 0.12, 0.18])  # varies per seed
    T.dr_cfgs["spatial"].target_rotate_z = DRBox(low=-0.4, high=0.4)
    make_kwargs = {}
    if args.playground:
        import importlib
        from simple.robots.registry import RobotRegistry
        _mod = importlib.import_module(f"simple.tasks.{args.task}")
        _K = next(c for c in vars(_mod).values() if isinstance(c, type) and getattr(c, "uid", None) == args.task)
        _size = [float(v) for v in args.floor_box.split(",")]
        _K.dr_cfgs["target"].asset_id, _K.dr_cfgs["target"].size = "primitive:cube", _size
        _K.dr_cfgs["distractors"].number_of_distractors = 6
        _sp = _K.dr_cfgs["spatial"]
        _sp.robot_region = DRBox(low=[-1.3, 0.0, 0.0], high=[-1.3, 0.0, 0.0])
        _sp.target_region = DRBox(low=[-0.82, 0.55], high=[-0.78, 0.65])  # placeholder; dropped onto the floor at reset
        _sp.target_rotate_z = DRBox(low=0.0, high=0.0)
        _sp.obj_surface_map = {"target": "ground"}
        _K.metadata["physics_dt"], _K.metadata["render_hz"] = sonic_config["SIMULATE_DT"], 50
        _K.dr_cfgs["scene"].table_height = DRBox(low=0.75, high=0.75)  # under the Sonic env z=0 is the floor, not the tabletop
        _orig_make = RobotRegistry.make

        def _make(**kw):  # the tabletop task builds its robot without kwargs; G1Sonic needs the sonic config
            if kw.get("uid") == "g1_sonic" and "sonic_config" not in kw:
                kw = {**kw, "sonic_config": sonic_config}
            return _orig_make(**kw)
        RobotRegistry.make = staticmethod(_make) if isinstance(RobotRegistry.__dict__.get("make"), staticmethod) else _make
        from simple.robots.g1_sonic import G1Sonic
        _orig_reset = G1Sonic.reset

        def _reset(self, **kw):  # the tabletop task resets its robot without a spawn pose (only the elastic-band anchor uses it)
            if "spawn_pose" not in kw:
                lay = getattr(_holder.get("task"), "layout", None)
                kw["spawn_pose"] = lay.robot.pose if lay is not None else type("P", (), {"position": [-1.3, 0.0, 0.8]})()
            return _orig_reset(self, **kw)
        _holder = {}
        G1Sonic.reset = _reset
        make_kwargs = {"task": args.task, "robot_uid": "g1_sonic", "target_object": "primitive:cube"}
        log(f"playground: {args.task} under the WBC, 6 distractors, floor box {_size}")
    log(f"creating env {args.env_id} (decoupled WBC) ...")
    t0 = time.time()
    env = gym.make(args.env_id, sonic_config=sonic_config, sim_mode="mujoco_isaac", headless=True, render_hz=50,
                   max_episode_steps=10 ** 7, **make_kwargs).unwrapped
    if args.playground:
        _holder["task"] = env.task
    adapter = WbcAdapter(env, sonic_config, record_every=args.record_every, sticky_grasp=args.sticky_grasp,
                         legacy_close=args.legacy_close, instruction=args.instruction, box_mass=args.box_mass)
    if args.playground:
        adapter.drop_side = 0.6
    adapter.reset(seed=0)
    log(f"env + adapter ready in {time.time() - t0:.1f}s; serving on 127.0.0.1:{args.port}")
    base.args, base.adapter, base.METHODS = args, adapter, base.build_methods(adapter)
    HTTPServer(("127.0.0.1", args.port), base.Handler).serve_forever()


if __name__ == "__main__":
    main()
