"""Scripted two-hand floor-box lift + carry on the WBC backend (privileged; feasibility ceiling). python wbc_probe.py --seeds 0 1 2 3 4"""
import argparse
import json
import math

from sim_client import SimClient

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
a = ap.parse_args()
c = SimClient("http://127.0.0.1:8765", timeout=900)
for seed in a.seeds:
    c.reset(seed)
    box = c.call("debug_objects")["target"]; bx, by, bz = box["center_pelvis"]; sx, sy, sz = box["size"]
    yaw = math.atan2(by, bx)
    c.move_to({"base_yaw": yaw}); c.move_to({"base_x": 0.0, "base_y": 0.0})
    box = c.call("debug_objects")["target"]; bx, by, bz = box["center_pelvis"]; sx, sy, sz = box["size"]
    st = c.get_state()["pelvis"]["odom"]
    c.move_to({"base_x": st[0] + (bx - sx / 2 - 0.17) * math.cos(st[2]), "base_y": st[1] + (bx - sx / 2 - 0.17) * math.sin(st[2])})
    c.move_to({"left_hand": 0.0, "right_hand": 0.0, "left_z": 0.25, "right_z": 0.25, "left_y": 0.3, "right_y": -0.3})
    c.move_to({"height": -0.45})
    box = c.call("debug_objects")["target"]; bx, by, bz = box["center_pelvis"]; sx, sy, sz = box["size"]; gz = bz + 0.05
    c.move_to({"left_x": bx, "right_x": bx, "left_y": by + sy / 2 + 0.1, "right_y": by - sy / 2 - 0.1, "left_z": gz, "right_z": gz,
               "left_roll": 0, "left_pitch": 0, "left_yaw": 0, "right_roll": 0, "right_pitch": 0, "right_yaw": 0})
    yl, yr = by + sy / 2 + 0.1, by - sy / 2 - 0.1
    for i in range(10):
        con = c.call("debug_contacts"); fl = sum(v for k, v in con.items() if "left_hand" in k and "cube" in k); fr = sum(v for k, v in con.items() if "right_hand" in k and "cube" in k)
        if fl > 4 and fr > 4: break
        if fl <= 4: yl -= 0.01
        if fr <= 4: yr += 0.01
        c.move_to({"left_y": yl, "right_y": yr})
    c.move_to({"left_y": yl - 0.015, "right_y": yr + 0.015})
    c.move_to({"height": 0.0})
    s1 = c.get_state(); lift = s1["box"]["lift"]
    o = s1["pelvis"]["odom"]; c.move_to({"base_x": o[0] + 0.6 * math.cos(o[2]), "base_y": o[1] + 0.6 * math.sin(o[2])})
    s2 = c.get_state()
    print(json.dumps({"seed": seed, "lift_after_stand_cm": round(lift * 100, 1), "lift_after_walk_cm": round(s2["box"]["lift"] * 100, 1),
                      "stage": s2["max_stage"], "fallen": s2["fallen"]}), flush=True)
