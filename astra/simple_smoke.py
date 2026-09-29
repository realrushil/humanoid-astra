"""Smoke test for sim_server_simple.py through the normal client: hands, crouch, turn, walk. Saves images to --out."""
import argparse
import base64
import json
import math
import os
import time

from sim_client import SimClient

p = argparse.ArgumentParser()
p.add_argument("--out", default="runs/simple_smoke")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--skip-base", action="store_true")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
c = SimClient(timeout=900)
print("health:", c.health(), flush=True)


def snap(tag):
    o = c.get_observation(include_third_person=True, marks=True, body_map=False)
    for k, b64 in o["images"].items():
        open(os.path.join(args.out, f"{tag}_{k}.jpg"), "wb").write(base64.b64decode(b64))
    print(f"[{tag}] {o['state_text']}", flush=True)
    return o


def move(tag, **t):
    t0 = time.time()
    r = c.move_to(t)
    print(f"[{tag}] move_to({json.dumps(t)}) -> {r['text']}  [{time.time() - t0:.1f}s wall]", flush=True)
    st = c.get_state()
    print(f"    stage={st['max_stage']} fallen={st['fallen']} lift={st['box']['lift']:.3f} "
          f"right grasp->target {st['hands']['right']['box_surface_dist']:.3f} m", flush=True)
    return r, st


print(c.describe(marks=True)["text"], flush=True)
r = c.reset(args.seed, record_dir=None)
print("reset:", r, flush=True)
st = c.get_state()
print("target in pelvis frame:", [round(v, 3) for v in st["box"]["pelvis"]], "hands:",
      {s: [round(v, 3) for v in st["hands"][s]["wrist_pelvis_pos"]] for s in ["left", "right"]}, flush=True)
snap("00_reset")
# hands: raise the right hand forward, then open/close
rx, ry, rz = st["hands"]["right"]["wrist_pelvis_pos"]
move("01_right_fwd", right_x=rx + 0.15, right_y=ry, right_z=rz + 0.15, right_roll=0.0, right_pitch=0.0, right_yaw=0.0)
snap("01_right_fwd")
move("02_close", right_hand=0.0)
move("03_open", right_hand=1.0)
move("04_left_up", left_x=0.30, left_y=0.20, left_z=0.10)
snap("04_left_up")
# crouch and stand
move("05_crouch", height=-0.25)
snap("05_crouch")
move("06_stand", height=0.0)
if not args.skip_base:
    move("07_turn", base_yaw=math.radians(45))
    snap("07_turn")
    move("08_walk", base_x=0.4, base_y=0.0, base_yaw=math.radians(45))
    snap("08_walk")
    move("09_back", base_x=0.0, base_y=0.0, base_yaw=0.0)
    snap("09_back")
print("measure centre:", c.measure("head", 320, 200)["text"], flush=True)
print("SMOKE_DONE", flush=True)
