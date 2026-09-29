"""Milestone 4: hard-coded pick-and-place through move_to only, using get_state() for box/table positions.
Proves the task is solvable through the LLM's interface. Usage: python scripted_policy.py --seeds 0 1 2 3 4

The sequence lives in `scripted_plan(c)`, a generator that yields (targets, note) and receives each move_to
result, so run_llm.py --dry-run can replay the exact same closed-loop logic through the agent loop."""

import argparse
import json
import math
import os
import time

from sim_client import SimClient, make_episode_videos

PI = math.pi
REACH_X = 0.50  # wrist x beyond this is at the edge of the arm's reach
CARRY = dict(right_x=0.30, right_y=-0.12, right_z=0.20, right_roll=0.0, right_pitch=0.0, right_yaw=0.0)


def scripted_plan(c):
    """Generator of (targets, note) move_to calls; `.send()` it each result dict. Returns a summary string
    when finished, raises RuntimeError when it wants to give up."""
    print_ = lambda s: print("  " + s, flush=True)  # noqa: E731

    # ---- pick (right hand, side approach from behind: fingers pass the box's outer face, thumb behind it).
    # The locomotion policy steps back/yaws when the arm extends, so re-read the box in the pelvis frame and
    # re-target before every step (closed loop) instead of trusting the reset-time position.
    def box_rel(dx, dz):
        bx, by, bz = c.get_state()["box"]["pelvis"]
        return dict(right_x=bx + dx, right_y=by - 0.045, right_z=bz + 0.01 + dz, right_roll=0.0, right_pitch=0.0, right_yaw=0.0)

    yield box_rel(-0.25, 0.12), "I see the red box on the table ahead. Moving the right hand above and behind it."
    yield box_rel(-0.25, 0.0), "Descending to the box height while staying behind it."
    for i in range(5):
        st = c.get_state()
        d = st["hands"]["right"]["box_surface_dist"]
        print_(f"servo {i}: grasp point {d * 100:.1f} cm from box surface")
        if d < 0.02 and i > 0:
            break
        t = box_rel(-0.13, 0.0)
        if t["right_x"] > REACH_X + 0.03:  # too far for the arm: step the base forward by the excess (current heading)
            px, py, pyaw = st["pelvis"]["odom"]
            step = t["right_x"] - REACH_X + 0.06
            print_(f"box too far for the arm ({t['right_x']:.2f} > {REACH_X}); stepping base forward {step * 100:.0f} cm")
            yield (dict(base_x=px + step * math.cos(pyaw), base_y=py + step * math.sin(pyaw), base_yaw=pyaw),
                   "The box is slightly out of reach; stepping the base forward a little.")
            continue
        yield t, "Advancing the open hand so the box sits between the fingers and thumb."
    yield dict(right_hand=0.0), "The box is between the fingers; closing the right hand."
    st = c.get_state()
    if not st["attached"]["right"]:
        raise RuntimeError(f"grasp did not attach (surface dist {st['hands']['right']['box_surface_dist'] * 100:.1f} cm)")
    yield dict(right_z=st["hands"]["right"]["wrist_pelvis_pos"][2] + 0.15), "Lifting the box off the table."
    yield CARRY, "Bringing the box close to the body before walking."
    print_(f"after lift: stage={c.get_state()['max_stage']} attached={c.get_state()['attached']}")

    # ---- navigate to table B's long (+world-y) side and face it
    st = c.get_state()
    tb = st["tables"]["B"]["center_odom"]
    tyaw = st["tables"]["B"]["yaw_odom"]  # direction of the table's +x axis in odom
    side = tyaw + PI / 2  # the table's +y side direction
    standoff = 0.30 + 0.33
    gx, gy = tb[0] + standoff * math.cos(side), tb[1] + standoff * math.sin(side)
    face = side + PI  # look back toward the table centre
    heading = math.atan2(gy, gx)

    def move_base(x, y, yaw, note, tries=3):
        for _ in range(tries):
            yield dict(base_x=x, base_y=y, base_yaw=yaw), note
            px, py, pyaw = c.get_state()["pelvis"]["odom"]
            if math.hypot(x - px, y - py) < 0.10 and abs((yaw - pyaw + PI) % (2 * PI) - PI) < math.radians(10):
                return
            print_("base pose not reached; retrying")
            note = "The base did not reach its target; re-commanding the same pose."

    yield dict(base_yaw=heading), "Turning toward the second table behind me."
    yield from move_base(gx, gy, heading, "Walking to a spot beside the second table.")
    yield from move_base(gx, gy, face, "Turning to face the second table.")
    st = c.get_state()
    print_(f"arrived: odom={[round(v, 3) for v in st['pelvis']['odom']]} stage={st['max_stage']} attached={st['attached']}")

    # ---- place 0.15 m inside the table's near edge; box centre sits ~(0.13, +0.03) from the wrist in the pelvis frame
    drop_x, drop_y = tb[0] + 0.15 * math.cos(side), tb[1] + 0.15 * math.sin(side)  # odom

    def place_rel(dz):
        st = c.get_state()
        px, py, pyaw = st["pelvis"]["odom"]
        cx, sy = math.cos(-pyaw), math.sin(-pyaw)
        lx, ly = cx * (drop_x - px) - sy * (drop_y - py), sy * (drop_x - px) + cx * (drop_y - py)
        box_z = st["tables"]["B"]["top_z"] + 0.035 + 0.03 - st["pelvis"]["pos"][2]
        return dict(right_x=lx - 0.132, right_y=ly - 0.027, right_z=box_z + dz, right_roll=0.0, right_pitch=0.0, right_yaw=0.0)

    for _ in range(3):  # make sure the drop point is in front of the robot before reaching for it
        t = place_rel(0.12)
        if 0.28 <= t["right_x"] <= 0.52 and -0.20 <= t["right_y"] <= 0.10:
            break
        print_(f"drop point not in front ({t['right_x']:.2f}, {t['right_y']:.2f}); re-positioning base")
        yield from move_base(gx, gy, face, "The table edge is not squarely in front of me; re-positioning the base.")
    yield place_rel(0.12), "The table is in front of me. Moving the box above the near edge of the table."
    yield place_rel(0.12), "Re-adjusting the hover position after the base shifted."
    yield place_rel(0.0), "Lowering the box to just above the table top."
    yield dict(right_hand=1.0), "The box is on the table; opening the hand to release it."
    yield dict(right_x=0.25, right_y=-0.15, right_z=0.15), "Retracting the hand away from the box."
    yield dict(right_x=0.20, right_y=-0.15, right_z=0.10), "Returning the arm to a resting pose."
    return "Picked up the box from the first table, carried it to the second table and placed it there."


def drive(plan, execute):
    """Run a plan generator, calling execute(targets, note) -> result for each yielded call."""
    r = None
    try:
        while True:
            targets, note = plan.send(r)
            r = execute(targets, note)
    except StopIteration as e:
        return e.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default="runs/m4_scripted")
    parser.add_argument("--video", action="store_true", help="encode per-camera + mosaic videos for every seed")
    args = parser.parse_args()
    c = SimClient()

    def run_episode(seed, out_dir):
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
        bx, by, bz = c.get_state()["box"]["pelvis"]
        print(f"seed {seed}: box in pelvis frame = ({bx:.3f}, {by:.3f}, {bz:.3f})")
        summary = drive(scripted_plan(c), move)
        st = c.get_state()
        res = {"seed": seed, "stage": st["max_stage"], "fallen": st["fallen"], "box_on_floor": st["box_on_floor"],
               "grasp_events": st["grasp_events"], "calls": len(calls), "sim_time": st["sim_time"],
               "wall_time": time.time() - t0, "box_final": st["box"]["pos"], "summary": summary}
        json.dump({"result": res, "calls": calls}, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
        return res

    results = []
    for seed in args.seeds:
        out_dir = os.path.join(args.out, f"seed{seed}")
        try:
            r = run_episode(seed, out_dir)
        except Exception as e:  # noqa: BLE001
            r = {"seed": seed, "stage": c.get_state()["max_stage"], "error": str(e)}
        print(f"==> seed {seed}: stage {r['stage']}  {r.get('error', '')}", flush=True)
        results.append(r)
        if args.video or seed == args.seeds[0]:
            make_episode_videos(os.path.join(out_dir, "frames"), os.path.join(out_dir, "episode"))
    print("\nSUMMARY: " + " ".join(f"seed{r['seed']}=stage{r['stage']}" for r in results))
    print(f"success (stage 4): {sum(r['stage'] == 4 for r in results)}/{len(results)}")
    print("SCRIPTED_DONE")
