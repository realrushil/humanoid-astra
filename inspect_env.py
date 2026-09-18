"""Step 0: load Isaac-PickPlace-Locomanipulation-G1-Abs-v0 headless and report its layout.

Run on the GPU box:  python inspect_env.py  (always headless, cameras enabled)
"""

import argparse
import time

# gotcha: pinocchio (Pink IK) must be imported BEFORE the Isaac Sim app starts, otherwise its
# boost.python registry collides with the one Isaac Sim loads ("No Python class registered ...").
import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument("--steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
args.enable_pinocchio = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# everything below must come after the app is up
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402
import isaaclab_tasks  # noqa: E402, F401

# gotcha: isaaclab_tasks blacklists "pick_place" packages from auto-import, so register explicitly
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def sec(title):
    print(f"\n===== {title} =====", flush=True)


def fmt(t):
    return "[" + ", ".join(f"{v:+.3f}" for v in t.flatten().tolist()) + "]"


t0 = time.time()
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env = gym.make(args.task, cfg=env_cfg).unwrapped
print(f"\nenv created in {time.time() - t0:.1f}s", flush=True)
obs, _ = env.reset()
robot = env.scene["robot"]
am = env.action_manager

sec("CONTROL RATE")
print(f"sim.dt={env.cfg.sim.dt} s  decimation={env.cfg.decimation}  -> env step dt={env.step_dt} s ({1 / env.step_dt:.0f} Hz)")
print(f"render_interval={env.cfg.sim.render_interval} physics steps  episode_length_s={env.cfg.episode_length_s}")

sec("ACTION SPACE")
print(f"action_space={env.action_space}  total_action_dim={am.total_action_dim}")
idx = 0
for name in am.active_terms:
    term = am.get_term(name)
    d = term.action_dim
    print(f"  [{idx}:{idx + d}] term '{name}' ({type(term).__name__}) dim={d}")
    idx += d

ik = am.get_term("upper_body_ik")
ctrl = ik.cfg.controller
print("\nupper_body_ik layout (traced from PinkInverseKinematicsAction):")
base = 0
for i, task in enumerate(ctrl.variable_input_tasks):
    if hasattr(task, "frame"):
        print(f"  [{base}:{base + 3}] {task.frame} position xyz")
        print(f"  [{base + 3}:{base + 7}] {task.frame} quaternion (w,x,y,z)")
        base += 7
print(f"  [{base}:{base + ik.hand_joint_dim}] hand joint position targets (rad), order:")
for j, n in enumerate(ik._hand_joint_names):
    print(f"      [{base + j}] {n}")
print(f"  IK-controlled arm/waist joints: {ik._isaaclab_controlled_joint_names}")
lb = am.get_term("lower_body_joint_pos")
print("\nlower_body_joint_pos layout: [vx, vy, wz, hip_height] -> AGILE RL policy -> leg joint targets")
print(f"  leg joints driven: {lb._joint_names}")
print(f"  policy_output_scale={lb.cfg.policy_output_scale}  policy={lb.cfg.policy_path}")

sec("IK TARGET FRAME")
print(f"controller.base_link_name={ctrl.base_link_name}")
for task in ctrl.variable_input_tasks:
    if hasattr(task, "frame"):
        print(f"  task frame={task.frame}  base_link_frame_name={getattr(task, 'base_link_frame_name', None)}")
print(
    "Action poses are interpreted in the ENV-LOCAL WORLD frame (world pose minus env origin);\n"
    "PinkInverseKinematicsAction.process_actions transforms them into the pelvis frame each step\n"
    "using the live pelvis pose, then LocalFrameTask tracks them relative to the pelvis."
)

sec("ROBOT RESET STATE")
root_pos = robot.data.root_pos_w[0] - env.scene.env_origins[0]
root_quat = robot.data.root_quat_w[0]
print(f"root pos (env-local)={fmt(root_pos)}  root quat wxyz={fmt(root_quat)}")
r, p, y = mu.euler_xyz_from_quat(root_quat.unsqueeze(0))
print(f"root rpy (rad)=({r.item():+.3f}, {p.item():+.3f}, {y.item():+.3f})  -> yaw {torch.rad2deg(y).item():+.1f} deg")
fwd = mu.quat_apply(root_quat.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=env.device))[0]
print(f"pelvis +x axis in world = {fmt(fwd)}   (robot 'forward')")
names = robot.data.body_names
print(f"num bodies={len(names)}  num joints={robot.num_joints}")
print(f"body names: {names}")
for link in ["pelvis", "torso_link", "head_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]:
    if link in names:
        i = names.index(link)
        pw = robot.data.body_pos_w[0, i] - env.scene.env_origins[0]
        qw = robot.data.body_quat_w[0, i]
        pp, qp = mu.subtract_frame_transforms(root_pos.unsqueeze(0), root_quat.unsqueeze(0), pw.unsqueeze(0), qw.unsqueeze(0))
        print(f"  {link:22s} env-local pos={fmt(pw)} quat={fmt(qw)} | in pelvis frame pos={fmt(pp)} quat={fmt(qp)}")
hand_links = [n for n in names if "hand" in n or "palm" in n]
print(f"hand/finger links: {hand_links}")

sec("HAND JOINTS")
lim = robot.data.joint_pos_limits[0]
for j, n in enumerate(ik._hand_joint_names):
    jid = robot.data.joint_names.index(n)
    print(f"  {n:28s} limits=[{lim[jid, 0]:+.3f}, {lim[jid, 1]:+.3f}]  default={robot.data.default_joint_pos[0, jid]:+.3f}")
print("teleop retargeter convention (g1_upper_body_motion_ctrl_retargeter): open=0 rad; closed: index/middle *_0,*_1 -> +1.0,")
print("  thumb_1 -> -0.4, thumb_2 -> -0.7, thumb_0 -> +/-0.5 (sign flips for right hand)")

sec("OBSERVATIONS")
for group, val in obs.items():
    if isinstance(val, dict):
        print(f"group '{group}' (dict):")
        for k, v in val.items():
            print(f"    {k:20s} shape={tuple(v.shape)}")
    else:
        print(f"group '{group}' tensor shape={tuple(val.shape)}")
        print(f"    terms: {env.observation_manager.active_terms[group]}")
        print(f"    dims : {env.observation_manager.group_obs_term_dim[group]}")

sec("CAMERAS")
print(f"scene sensors: {list(env.scene.sensors.keys())}")
stage = env.sim.stage
cams = [p.GetPath().pathString for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
print(f"USD camera prims in stage: {cams}")
print("-> stock env defines NO camera sensors; cameras must be added (CameraCfg/TiledCameraCfg) for the adapter.")

sec("SCENE OBJECTS (env-local)")
for key in ["packing_table", "object"]:
    asset = env.scene[key]
    if hasattr(asset, "data") and hasattr(asset.data, "root_pos_w"):
        print(f"  {key}: pos={fmt(asset.data.root_pos_w[0] - env.scene.env_origins[0])} quat={fmt(asset.data.root_quat_w[0])}")
    else:
        print(f"  {key}: cfg pos={env_cfg.scene.__dict__[key].init_state.pos}")
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
for prim_path in ["/World/envs/env_0/PackingTable", "/World/envs/env_0/Object"]:
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        print(f"  {prim_path} world AABB min=({mn[0]:+.3f},{mn[1]:+.3f},{mn[2]:+.3f}) max=({mx[0]:+.3f},{mx[1]:+.3f},{mx[2]:+.3f})")

sec("LOCOMOTION COMMAND RANGES")
print("Not stored in the env or the TorchScript policy. Ranges used by the stock teleop retargeters that feed this")
print("policy (g1_motion_controller_locomotion.py): vx,vy in [-0.5, 0.5] m/s (movement_scale), wz in [-1, 1] rad/s")
print("(raw thumbstick, unscaled), hip_height clamped to [0.4, 1.0] m, default 0.72 m (g1_lower_body_standing.py).")
print("Standing retargeter sends [0,0,0,0.72]. Empirical limits to be measured in milestone 3.")

sec(f"STEPPING {args.steps} STEPS WITH A HOLD ACTION (cameras enabled)")
# hold action: current wrist poses in env-local world frame, hands open, base standing at 0.72
li, ri = names.index("left_wrist_yaw_link"), names.index("right_wrist_yaw_link")
act = torch.zeros(1, am.total_action_dim, device=env.device)
act[0, 0:3] = robot.data.body_pos_w[0, li] - env.scene.env_origins[0]
act[0, 3:7] = robot.data.body_quat_w[0, li]
act[0, 7:10] = robot.data.body_pos_w[0, ri] - env.scene.env_origins[0]
act[0, 10:14] = robot.data.body_quat_w[0, ri]
act[0, 28:32] = torch.tensor([0.0, 0.0, 0.0, 0.72], device=env.device)
print(f"action={fmt(act)}")
t0 = time.time()
for i in range(args.steps):
    obs, rew, terminated, truncated, info = env.step(act)
    if i % 10 == 0 or i == args.steps - 1:
        pz = (robot.data.root_pos_w[0] - env.scene.env_origins[0])[2].item()
        lw = robot.data.body_pos_w[0, li] - env.scene.env_origins[0]
        print(f"  step {i:3d}: pelvis z={pz:.3f}  left wrist={fmt(lw)}  terminated={terminated.item()} truncated={truncated.item()}")
dt = (time.time() - t0) / args.steps
print(f"wall time per env step: {dt * 1000:.0f} ms  ({env.step_dt / dt:.2f}x realtime)")

print("\nINSPECT_DONE", flush=True)
env.close()
simulation_app.close()
