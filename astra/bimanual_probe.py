"""Scripted two-hand box lift on the SIMPLE floor-box scene (privileged, feasibility only): walk up, crouch, closed hands pitched
down onto both side faces, squeeze until contact, lift. Usage: python bimanual_probe.py [--seed 0] [--record]"""
import argparse
import math

from sim_client import SimClient

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--record", action="store_true")
a = ap.parse_args()
c = SimClient("http://127.0.0.1:8765", timeout=900)
c.reset(a.seed, record_dir=f"runs/bimanual_probe_seed{a.seed}" if a.record else None)
def P(tag, r):
    t = r["text"]; err = t[t.find("final error"):t.find("| clipped")]
    print(f"{tag}: {t[:60]}... {err}", flush=True)
box = lambda: c.call("debug_objects")["target"]  # noqa: E731
b = box(); print("box (pelvis)", [round(v, 2) for v in b["center_pelvis"]], "size", [round(v, 2) for v in b["size"]])
near_face = b["center_pelvis"][0] - b["size"][0] / 2
P("walk", c.move_to({"base_x": near_face - 0.13, "base_y": b["center_pelvis"][1], "base_yaw": 0.0}))
P("close+raise", c.move_to({"left_hand": 0.0, "right_hand": 0.0, "left_z": 0.15, "right_z": 0.15}))
P("crouch", c.move_to({"height": -0.2}))
b = box(); bx, by, bz = b["center_pelvis"]; sx, sy, sz = b["size"]; top = bz + sz / 2
print("box after crouch (pelvis)", [round(v, 2) for v in (bx, by, bz)], "top", round(top, 2))
gap = 0.10  # fist pad ~7.5 cm inward of the wrist + link radius; start clear of the faces
pose = lambda side, y: {f"{side}_x": bx - 0.02, f"{side}_y": y, f"{side}_z": top + 0.09, f"{side}_roll": 0.0, f"{side}_pitch": 1.571, f"{side}_yaw": 0.0}  # noqa: E731
P("pre-squeeze", c.move_to({**pose("right", by - sy / 2 - gap), **pose("left", by + sy / 2 + gap)}))
yr, yl = by - sy / 2 - gap, by + sy / 2 + gap
for i in range(8):
    con = c.call("debug_contacts")
    fr = sum(v for k, v in con.items() if "right" in k and "cube" in k)
    fl = sum(v for k, v in con.items() if "left" in k and "cube" in k)
    print(f"  squeeze step {i}: right {fr:.1f} N, left {fl:.1f} N  (yr={yr:.3f}, yl={yl:.3f})", flush=True)
    if fr > 4 and fl > 4:
        break
    if fr <= 4:
        yr += 0.01
    if fl <= 4:
        yl -= 0.01
    c.move_to({"right_y": yr, "left_y": yl})
P("extra squeeze", c.move_to({"right_y": yr + 0.015, "left_y": yl - 0.015}))
print("contacts:", c.call("debug_contacts"))
P("lift", c.move_to({"right_z": top + 0.24, "left_z": top + 0.24}))
s = c.get_state()
print("lift_cm", round(s["box"]["lift"] * 100, 1), "stage", s["max_stage"], "fallen", s["fallen"], "contacts:", c.call("debug_contacts"))
P("hold", c.move_to({"right_z": top + 0.24}))
s = c.get_state(); print("after hold: lift_cm", round(s["box"]["lift"] * 100, 1), "stage", s["max_stage"], "fallen", s["fallen"])
