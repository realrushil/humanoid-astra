"""Scripted bend-and-pick through move_to only (SIMPLE backend), closed-loop on get_state(). Proves the task is solvable
through the LLM's interface. Usage (on the box): ~/IsaacLab/.venv/bin/python scripted_simple.py --seeds 0 1 2

Both hands are tucked, the robot crouches, then the hand comes down beside the box with the fingers pointing across it
and slides sideways so the box sits between thumb and fingers (a pinch across its thin side), closes, and lifts."""

import argparse
import json
import math
import os
import time

from sim_client import SimClient, make_video

# Right hand at yaw +90 deg (fingers pointing left, +y): relative to the hand base the thumb tip is at (-0.11, +0.07),
# the index/middle tips at (+0.01, +0.21) and the grasp point (fingertip centroid) at (-0.03, +0.16). So a box sitting at
# (base_x - 0.03, base_y + 0.16) has the thumb in front of it (-x) and the fingers behind it (+x): a side pinch across
# the box's thin dimension. Measured with explore_tmp3.py; mirror the y offsets for the left hand.
YAW = {"right": math.pi / 2, "left": -math.pi / 2}
GP_OFF = {"right": (-0.031, 0.164, -0.005), "left": (-0.031, -0.164, -0.005)}


PITCH_TOP = 1.05  # rad: fingers pointing down; grasp point relative to the hand base at this pitch (measured)
GP_TOP = {"right": (0.083, 0.029, -0.142), "left": (0.083, -0.029, -0.142)}


def scripted_plan(c, side="right", grasp="top"):
    print_ = lambda s: print("  " + s, flush=True)  # noqa: E731
    sgn = 1.0 if side == "right" else -1.0  # approach from the hand's own side

    def hand_top(dx, dy, dz):
        bx, by, bz = c.get_state()["box"]["pelvis"]
        ox, oy, oz = GP_TOP[side]
        return {f"{side}_x": bx + dx - ox, f"{side}_y": by + dy - oy, f"{side}_z": bz + dz - oz,
                f"{side}_roll": 0.0, f"{side}_pitch": PITCH_TOP, f"{side}_yaw": 0.0}

    def hand(dx, dy, dz):
        """hand-base target (sideways grip) that puts the grasp point at box centre + (dx, dy, dz)"""
        bx, by, bz = c.get_state()["box"]["pelvis"]
        ox, oy, oz = GP_OFF[side]
        return {f"{side}_x": bx + dx - ox, f"{side}_y": by + dy - oy, f"{side}_z": bz + dz - oz,
                f"{side}_roll": 0.0, f"{side}_pitch": 0.0, f"{side}_yaw": YAW[side]}

    # crouching drops the pelvis ~23 cm, which brings the outstretched fingers to the box's height: tuck both hands first
    st0 = c.get_state()
    bz0 = st0["box"]["pelvis"][2]
    if st0["box"]["pelvis"][0] > 0.42:  # beyond a comfortable reach: step the base forward first
        px, py, pyaw = st0["pelvis"]["odom"]
        step = st0["box"]["pelvis"][0] - 0.36
        yield dict(base_x=px + step * math.cos(pyaw), base_y=py + step * math.sin(pyaw), base_yaw=pyaw), \
            f"The object is {st0['box']['pelvis'][0] * 100:.0f} cm ahead, a stretch; stepping {step * 100:.0f} cm closer first."
    if bz0 < -0.15:  # low table: crouch (after raising the hands so the fingers clear the box on the way down)
        yield dict(left_x=0.22, left_y=0.16, left_z=0.26, right_x=0.22, right_y=-0.16, right_z=0.26), "Raising both hands so the fingers clear the box when I crouch."
        yield dict(height=-0.23), "The box is on a low table; crouching to bring it within reach."
    if grasp == "side":
        yield hand(0.01, -sgn * 0.12, 0.20), "Moving the hand beside the object, fingers pointing across it, above the table."
        yield hand(0.01, -sgn * 0.12, 0.01), "Lowering the open hand to the object's height, beside it."
        goal_dz = 0.01
    else:  # top-down: fingers pointing down (pitch 60 deg), thumb and fingers close around the object from above
        yield hand_top(0.0, 0.0, 0.16), "Moving the hand above the object with the fingers pointing down."
        goal_dz = 0.0

    def servo(hover_dz, tol_xy, tol_z, note, tries=5, off=(0.0, 0.0)):
        """closed loop on the MEASURED grasp point (the wrist cannot always reach the commanded orientation)"""
        for i in range(tries):
            st = c.get_state()
            h = st["hands"][side]
            g, w = h["grasp_point_pelvis"], h["wrist_pelvis_pos"]
            b = [st["box"]["pelvis"][0] + off[0], st["box"]["pelvis"][1] + off[1], st["box"]["pelvis"][2]]
            dxy = math.hypot(g[0] - b[0], g[1] - b[1])
            print_(f"servo {i}: grasp point {dxy * 100:.1f} cm (xy) from object centre, dz {(g[2] - b[2]) * 100:+.0f} cm (want {hover_dz * 100:+.0f}), stage {st['max_stage']}")
            if dxy < tol_xy and abs(g[2] - b[2] - hover_dz) < tol_z:
                return True
            if st["box"]["lift"] < -0.1:
                raise RuntimeError("object knocked off the table")
            yield {f"{side}_x": w[0] + (b[0] - g[0]), f"{side}_y": w[1] + (b[1] - g[1]), f"{side}_z": w[2] + (b[2] + hover_dz - g[2])}, note
        return False

    if grasp == "top":  # align above the object first (fingertips ~2 cm clear of it), then come down beside it: the two
        # fingertips sit 5 cm to the +x/-y side of the grasp point, so offset the target so they pass clear of the object
        yield from servo(0.09, 0.015, 0.03, "Centring the hand above the object before descending.", off=(0.02, -0.02 * sgn))
        yield from servo(0.035, 0.015, 0.015, "Lowering the hand so the object sits between the thumb and the fingers.", off=(0.02, -0.02 * sgn))
    else:
        yield from servo(goal_dz, 0.025, 0.02, "Lowering the hand so the object sits between the thumb and the fingers.")
    yield {f"{side}_hand": 0.0}, "Closing the hand on the object."
    st = c.get_state()
    p = st["hands"][side]["wrist_pelvis_pos"]
    print_(f"closed: openness {st['hands'][side]['openness_measured']:.2f}")
    yield {f"{side}_z": p[2] + 0.06}, "Lifting the object a little to check the grip."
    yield {f"{side}_z": p[2] + 0.16}, "Lifting the object."
    st = c.get_state()
    print_(f"after lift: stage={st['max_stage']} lift={st['box']['lift']:.3f} success={st['task_success']}")
    if st["box"]["lift"] < 0.05:
        raise RuntimeError(f"box did not lift (lift {st['box']['lift']:.3f} m)")
    yield {f"{side}_z": p[2] + 0.15, f"{side}_x": p[0] - 0.05}, "Holding the box up."
    return "Crouched, gripped the cracker box from the side and lifted it off the table."


def drive(plan, execute):
    r = None
    try:
        while True:
            targets, note = plan.send(r)
            r = execute(targets, note)
    except StopIteration as e:
        return e.value


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/simple_scripted")
    ap.add_argument("--side", default="right")
    ap.add_argument("--grasp", default="top", choices=["top", "side"])
    args = ap.parse_args()
    c = SimClient(timeout=900)
    results = []
    for seed in args.seeds:
        out_dir = os.path.join(args.out, f"seed{seed}")
        os.makedirs(out_dir, exist_ok=True)
        calls = []

        def move(t, note):
            r = c.move_to(t)
            calls.append({"targets": t, "note": note, "result": r["text"]})
            print(f"  move_to({json.dumps({k: round(v, 3) for k, v in t.items()})})\n     -> {r['text']}", flush=True)
            if r.get("fallen"):
                raise RuntimeError("robot fell")
            return r

        t0 = time.time()
        c.reset(seed, record_dir=os.path.join(out_dir, "frames"))
        st = c.get_state()
        print(f"seed {seed}: '{st['instruction']}' box in pelvis frame = {[round(v, 3) for v in st['box']['pelvis']]} yaw {math.degrees(st['box']['yaw_pelvis']):.0f} deg")
        try:
            summary = drive(scripted_plan(c, args.side, args.grasp), move)
            err = None
        except Exception as e:  # noqa: BLE001
            summary, err = None, str(e)
        st = c.get_state()
        res = {"seed": seed, "stage": st["max_stage"], "task_success": st["task_success"], "fallen": st["fallen"], "lift": st["box"]["lift"],
               "calls": len(calls), "sim_time": st["sim_time"], "wall_time": round(time.time() - t0, 1), "summary": summary, "error": err}
        json.dump({"result": res, "calls": calls}, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
        print(f"==> seed {seed}: stage {res['stage']} success={res['task_success']} {err or ''}", flush=True)
        results.append(res)
        for cam in ["head", "third_person"]:
            d = os.path.join(out_dir, "frames", cam)
            if os.path.isdir(d) and os.listdir(d):
                make_video(d, os.path.join(out_dir, f"episode_{cam}.mp4"))
    print("SUMMARY: " + " ".join(f"seed{r['seed']}=stage{r['stage']}" for r in results))
    print("SCRIPTED_DONE", flush=True)
