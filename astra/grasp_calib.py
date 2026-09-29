"""Scripted grasp calibration on the SIMPLE tabletop task (privileged object poses; never used by the LLM).
Checks that "put the grasp point on the object's centre, close, lift" works across object shapes and hand orientations.
Usage (sim server running, one target object per server start): python grasp_calib.py --seeds 0 1 --out runs/grasp_calib/apple.jsonl"""

import argparse
import json
import math
import os

from sim_client import SimClient

# hand roll/pitch/yaw (right hand; the left hand mirrors roll and yaw). top_*: the hand's opening faces straight down, so
# the object enters between thumb and fingers from above and only the fingertips come near the table. top_x pinches
# along x (thumb behind the object, fingers in front); top_y is the same turned 90 deg about the vertical (pinch along y).
STRATEGIES = {
    "upright": (0.0, 0.0, 0.0),
    "top_x": (-1.571, 0.524, 0.0),
    "top_y": (-1.571, 0.524, 1.571),
}


def run_trial(c, seed, strategy, log, record=None):
    c.reset(seed, record_dir=record)
    objs = c.call("debug_objects")
    t = objs["target"]
    cx, cy, cz = t["center_pelvis"]
    sx, sy, sz = t["size"]
    side = "right" if cy <= 0.03 else "left"
    roll, pitch, yaw = STRATEGIES[strategy]
    if side == "left":
        roll, yaw = -roll, -yaw
    odom0 = c.get_state()["pelvis"]["odom"]
    gz = max(cz, cz - sz / 2 + 0.04)  # the open hand's lowest finger is ~3.5 cm below the grasp point: keep it off the tabletop

    def place(gx, gy, gz_, hand):
        """closed loop on the reported grasp point: hand base target corrected until the grasp point is within 5 mm"""
        st = c.get_state()["hands"][side]
        base = [st["wrist_pelvis_pos"][i] + [gx, gy, gz_][i] - st["grasp_point_pelvis"][i] for i in range(3)]
        for _ in range(4):
            r = c.move_to({f"{side}_x": base[0], f"{side}_y": base[1], f"{side}_z": base[2], f"{side}_roll": roll,
                           f"{side}_pitch": pitch, f"{side}_yaw": yaw, f"{side}_hand": hand})
            gp = c.get_state()["hands"][side]["grasp_point_pelvis"]
            err = [[gx, gy, gz_][i] - gp[i] for i in range(3)]
            if math.dist(err, [0, 0, 0]) < 0.005:
                break
            base = [base[i] + err[i] for i in range(3)]
        return math.dist(err, [0, 0, 0]), r

    pre_err, r_pre = place(cx, cy, gz + 0.10, 1.0)
    obj_pre = c.call("debug_objects")["target"]["center_pelvis"]
    # descend (re-read the object in case the approach nudged it; a real policy would re-measure)
    ox, oy, _ = obj_pre
    err, r_desc = place(ox, oy, gz, 1.0)
    con_desc = c.call("debug_contacts")
    moved_before_close = math.dist(c.call("debug_objects")["target"]["center_pelvis"][:2], [cx, cy])
    st = c.get_state()["hands"][side]
    r_close = c.move_to({f"{side}_hand": 0.0})
    con_close = c.call("debug_contacts")
    c.move_to({f"{side}_z": st["wrist_pelvis_pos"][2] + 0.15})
    c.move_to({f"{side}_z": st["wrist_pelvis_pos"][2] + 0.15})  # hold ~1 s more
    s = c.get_state()
    lift = s["box"]["lift"]
    res = {"seed": seed, "object": t["name"], "size_cm": [round(v * 100, 1) for v in t["size"]], "strategy": strategy, "hand": side,
           "place_err_mm": round(err * 1000, 1), "pre_err_mm": round(pre_err * 1000, 1), "obj_moved_before_close_cm": round(moved_before_close * 100, 1),
           "lift_cm": round(lift * 100, 1), "success": lift > 0.10, "stage": s["max_stage"], "fallen": s["fallen"],
           "base_drift_cm": round(math.dist(s["pelvis"]["odom"][:2], odom0[:2]) * 100, 1),
           "stopped": [t for t in (r_pre["text"], r_desc["text"], r_close["text"]) if "STOPPED" in t][:1],
           "contacts_after_descend": con_desc, "contacts_after_close": con_close}
    print(json.dumps(res), flush=True)
    log.write(json.dumps(res) + "\n")
    log.flush()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--strategies", nargs="+", default=["upright"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", action="store_true", help="record frames on the box under <out dir>/rec_<seed>_<strategy>")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    c = SimClient("http://127.0.0.1:8765", timeout=900)
    with open(a.out, "a") as log:
        for seed in a.seeds:
            for strat in a.strategies:
                try:
                    rec = f"{os.path.dirname(a.out)}/rec_{os.path.basename(a.out)[:-6]}_{seed}_{strat}" if a.record else None
                    run_trial(c, seed, strat, log, rec)
                except Exception as e:  # noqa: BLE001
                    print(f"seed {seed} {strat}: ERROR {e}", flush=True)
