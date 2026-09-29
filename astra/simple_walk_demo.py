"""Walk demo on the SIMPLE backend: back away from the table, turn, sidestep, come back. Records head + third-person
frames (server --record-every N) and encodes mp4s. Usage (on the box): ~/IsaacLab/.venv/bin/python simple_walk_demo.py"""
import argparse
import json
import math
import os
import time

from sim_client import SimClient, make_video

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/simple_walk_demo")
ap.add_argument("--fps", type=int, default=25)
args = ap.parse_args()
c = SimClient(timeout=900)
os.makedirs(args.out, exist_ok=True)
c.reset(0, record_dir=os.path.join(args.out, "frames"))
t0 = time.time()
plan = [
    (dict(base_x=-0.7, base_y=0.0, base_yaw=0.0), "walk backwards away from the table"),
    (dict(base_yaw=math.radians(90)), "turn left in place"),
    (dict(base_x=-0.7, base_y=0.7, base_yaw=math.radians(90)), "walk forward (odom +y)"),
    (dict(base_yaw=math.radians(180)), "turn to face away from the table"),
    (dict(base_x=-1.2, base_y=0.7, base_yaw=math.radians(180)), "walk forward again"),
    (dict(base_yaw=0.0), "turn around"),
    (dict(base_x=-0.3, base_y=0.2, base_yaw=0.0), "walk back toward the table"),
    (dict(left_x=0.35, left_y=0.2, left_z=0.3, right_x=0.35, right_y=-0.2, right_z=0.3), "raise both hands"),
]
for t, note in plan:
    r = c.move_to(t)
    st = c.get_state()
    print(f"{note}: {r['text'].split('|')[1].strip()}; odom={[round(v, 2) for v in st['pelvis']['odom']]} fallen={st['fallen']}", flush=True)
    if r.get("fallen"):
        break
st = c.get_state()
print(f"sim {st['sim_time']:.1f}s in {time.time() - t0:.0f}s wall", flush=True)
for cam in ["third_person", "head"]:
    d = os.path.join(args.out, "frames", cam)
    if os.path.isdir(d) and os.listdir(d):
        print(cam, make_video(d, os.path.join(args.out, f"walk_{cam}.mp4"), fps=args.fps), flush=True)
json.dump(st, open(os.path.join(args.out, "final_state.json"), "w"), indent=1)
print("WALK_DONE", flush=True)
