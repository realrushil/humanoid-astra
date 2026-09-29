"""Milestone 2 check: reset, fetch one observation, save a JPEG per camera under runs/m2_frames/."""

import argparse
import json
import os
import time

from sim_client import SimClient

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="runs/m2_frames")
args = parser.parse_args()

c = SimClient()
print("health:", c.health())
t0 = time.time()
print("reset:", c.reset(args.seed), f"({time.time() - t0:.1f}s)")
obs = c.get_observation(include_third_person=True)
os.makedirs(args.out, exist_ok=True)
for name, b64 in obs["images"].items():
    path = os.path.join(args.out, f"seed{args.seed}_{name}.jpg")
    with open(path, "wb") as f:
        f.write(SimClient.decode_image(b64))
    print("saved", path)
print("state_text:", obs["state_text"])
st = c.get_state()
print("state:", json.dumps({k: st[k] for k in ["box", "tables", "pelvis", "stage", "fallen", "box_on_floor"]}, indent=1))
print("hands:", json.dumps(st["hands"], indent=1))
print("describe:\n" + c.describe()["text"])
