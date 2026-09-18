"""Milestone 3: exercise the adapter through the RPC. Turn 90 deg, walk 1 m forward, move each hand, then repeat
the walk while holding a hand target and check the hand stays put in the pelvis frame. Records a video."""

import argparse
import json
import math
import os
import time

from sim_client import SimClient, make_episode_videos

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="runs/m3_test")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
c = SimClient()
log = open(os.path.join(args.out, "log.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")


def move(**targets):
    t0 = time.time()
    r = c.move_to(targets)
    say(f"\n> move_to({json.dumps(targets)})  [{time.time() - t0:.1f}s wall]")
    say("  " + r["text"])
    st = c.get_state()
    ox, oy, oyaw = st["pelvis"]["odom"]
    say(f"  odom: x={ox:+.3f} y={oy:+.3f} yaw={math.degrees(oyaw):+.1f}deg  pelvis z={st['pelvis']['pos'][2]:.3f}  fallen={st['fallen']}")
    assert not st["fallen"], "ROBOT FELL"
    return r, st


def hand_pose(st, side):
    p = st["hands"][side]["wrist_pelvis_pos"]
    rpy = st["hands"][side]["wrist_pelvis_rpy"]
    return p, rpy


say("health:", c.health())
say("reset:", c.reset(args.seed, record_dir=os.path.join(args.out, "frames")))

say("\n===== PART 1: turn 90 deg left, walk 1 m forward (new heading), then hand targets")
move(base_yaw=math.pi / 2)
move(base_x=0.0, base_y=1.0)  # 1 m forward along the robot's new heading (odom +y)
_, st = move(left_x=0.40, left_y=0.25, left_z=0.30, left_roll=0.0, left_pitch=0.0, left_yaw=0.0)
p, rpy = hand_pose(st, "left")
say(f"  left wrist in pelvis: pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) rpy=({math.degrees(rpy[0]):+.1f},{math.degrees(rpy[1]):+.1f},{math.degrees(rpy[2]):+.1f})deg")
_, st = move(right_x=0.40, right_y=-0.25, right_z=0.10, right_roll=0.0, right_pitch=-0.5, right_yaw=0.0)
p, rpy = hand_pose(st, "right")
say(f"  right wrist in pelvis: pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) rpy=({math.degrees(rpy[0]):+.1f},{math.degrees(rpy[1]):+.1f},{math.degrees(rpy[2]):+.1f})deg")
_, st = move(left_hand=0.0, right_hand=0.0)
say(f"  measured openness: left={st['hands']['left']['openness_measured']:.2f} right={st['hands']['right']['openness_measured']:.2f}")
_, st = move(left_hand=1.0, right_hand=1.0)
say(f"  measured openness: left={st['hands']['left']['openness_measured']:.2f} right={st['hands']['right']['openness_measured']:.2f}")

say("\n===== PART 2: reset, hold a left-hand target, walk the same path in 4 segments, check the hand in the pelvis frame")
c.reset(args.seed, record_dir=os.path.join(args.out, "frames2"))
hold = dict(left_x=0.35, left_y=0.20, left_z=0.25, left_roll=0.0, left_pitch=0.0, left_yaw=0.0)
_, st = move(**hold)
p0, _ = hand_pose(st, "left")
say(f"  hold pose reached: ({p0[0]:+.3f},{p0[1]:+.3f},{p0[2]:+.3f})")
max_dev = 0.0
for target in [dict(base_yaw=math.pi / 2), dict(base_x=0.0, base_y=0.33), dict(base_x=0.0, base_y=0.66), dict(base_x=0.0, base_y=1.0)]:
    _, st = move(**target)
    p, rpy = hand_pose(st, "left")
    dev = math.dist(p, [hold["left_x"], hold["left_y"], hold["left_z"]])
    max_dev = max(max_dev, dev)
    say(f"  left wrist in pelvis after segment: ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  deviation from commanded={dev * 100:.1f} cm")
say(f"\nmax hand deviation from commanded pelvis-frame pose while walking: {max_dev * 100:.1f} cm")
say("stage:", st["stage"], "fallen:", st["fallen"])

say("\n===== video")
for name, prefix in [("frames", "part1"), ("frames2", "part2")]:
    outs = make_episode_videos(os.path.join(args.out, name), os.path.join(args.out, prefix))
    say("wrote", ", ".join(outs.values()))
say("\nTEST_ADAPTER_DONE")
