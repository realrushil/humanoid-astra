"""Probe a SIMPLE G1 whole-body env headless: timings, obs/info layout, camera frames.

Run on the box inside the SIMPLE venv:
  cd ~/SIMPLE && MUJOCO_GL=egl .venv/bin/python ~/astra/simple_probe.py --steps 100
"""
import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--env-id", default="simple/G1WholebodyBendPickMP-v0")
p.add_argument("--task", default="g1_wholebody_bend_pick_mp")
p.add_argument("--robot-uid", default="g1_wholebody")
p.add_argument("--sim-mode", default="mujoco_isaac")
p.add_argument("--steps", type=int, default=100)
p.add_argument("--out", default=os.path.expanduser("~/simple_probe_out"))
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

import gymnasium as gym
import simple.envs  # noqa: F401  registers env ids

t0 = time.time()
env = gym.make(
    args.env_id,
    task=args.task,
    robot_uid=args.robot_uid,
    sim_mode=args.sim_mode,
    headless=True,
    render_hz=50,
    max_episode_steps=args.steps + 10,
)
print(f"[probe] make: {time.time() - t0:.1f}s", flush=True)

t0 = time.time()
obs, info = env.reset(seed=0)
print(f"[probe] reset: {time.time() - t0:.1f}s", flush=True)

task = env.unwrapped.task
robot = task.robot
print("[probe] instruction:", getattr(task, "instruction", None), flush=True)
print("[probe] obs keys:", {k: (getattr(v, "shape", None), getattr(v, "dtype", None)) for k, v in obs.items()}, flush=True)
print("[probe] info keys:", list(info.keys())[:40], flush=True)
print("[probe] n joints:", len(robot.joint_names), flush=True)
print("[probe] joint names:", robot.joint_names, flush=True)
print("[probe] hand names:", getattr(robot, "hand_names", None), flush=True)
print("[probe] metadata:", task.metadata, flush=True)
print("[probe] sensor cfgs:", {k: (type(v).__name__, getattr(v, "mount", None), getattr(v, "width", None), getattr(v, "height", None)) for k, v in task.sensor_cfgs.items()}, flush=True)


def write_png(path, rgb):
    """Dependency-free PNG writer (Isaac Sim's bundled Pillow has a broken C extension)."""
    import struct
    import zlib
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def save_frames(tag):
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] in (3, 4):
            write_png(os.path.join(args.out, f"{tag}_{k}.png"), np.ascontiguousarray(v[..., :3]).astype(np.uint8))


save_frames("reset")

# stand still: loco_command with zero velocity, keep default height
from simple.core.action import ActionCmd

stand = ActionCmd("loco_command", command=[0.0, 0.0, 0.0, 0.0, 0.0, -0.15, 0.0, 0.0])
times = []
try:
  for i in range(args.steps):
    t0 = time.time()
    obs, reward, terminated, truncated, info = env.step(stand)
    times.append(time.time() - t0)
    if i in (0, 9, 49):
        save_frames(f"step{i + 1}")
    if terminated or truncated:
        print(f"[probe] episode ended at step {i + 1}", flush=True)
        break
except Exception as e:  # keep going to the timing report
    print(f"[probe] step loop error at step {len(times)}: {e!r}", flush=True)
times = np.array(times) if times else np.array([np.nan])
print(f"[probe] step time: mean {times.mean() * 1000:.0f} ms, median {np.median(times) * 1000:.0f} ms, "
      f"first {times[0] * 1000:.0f} ms -> {1 / np.median(times):.1f} steps/s ({len(times)} steps)", flush=True)
print("[probe] target pose (info['target']):", np.round(info.get("target", np.zeros(7)), 3).tolist(), flush=True)
print("[probe] robot pose:", robot.get_robot_pose() if hasattr(robot, "get_robot_pose") else None, flush=True)
json.dump({"step_ms_median": float(np.median(times) * 1000), "steps": int(len(times))},
          open(os.path.join(args.out, "timing.json"), "w"))
env.close()
print("[probe] PROBE_DONE", flush=True)
