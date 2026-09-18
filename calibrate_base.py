"""Measure what the locomotion policy actually does for raw (vx, vy, wz) commands. Debug tool for the adapter."""

import sys

from sim_client import SimClient

c = SimClient()
tests = [(0, 0, 0.3), (0, 0, 0.6), (0, 0, 1.0), (0.15, 0, 0.6), (0.3, 0, 0.6), (0.1, 0, 0), (0.2, 0, 0), (0.3, 0, 0),
         (0.5, 0, 0), (-0.3, 0, 0), (0, 0.2, 0), (0, 0.4, 0), (0, -0.3, 0), (0.05, 0, 0.3)]
if len(sys.argv) > 1:
    tests = [tuple(float(v) for v in t.split(",")) for t in sys.argv[1:]]
for i, t in enumerate(tests):
    vx, vy, wz = t[:3]
    hip = t[3] if len(t) > 3 else 0.72
    secs = t[4] if len(t) > 4 else 3.0
    c.reset(0)
    c.call("calibrate", vx=-0.5, vy=0.0, wz=0.0, seconds=2.5)  # back away from table A first
    r = c.call("calibrate", vx=vx, vy=vy, wz=wz, seconds=secs, hip=hip)
    m = (r["vx_meas"], r["vy_meas"], r["wz_meas"])
    print(f"cmd vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} hip={hip:.2f} {secs:.0f}s -> meas vx={m[0]:+.3f} vy={m[1]:+.3f} wz={m[2]:+.3f} rad/s  coast: {r['coast_xy']*100:.1f}cm {r['coast_yaw']*57.3:+.1f}deg  fallen={r['fallen']}", flush=True)
