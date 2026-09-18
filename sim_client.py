"""sim_client.py: thin client for sim_server.py. No Isaac types cross this boundary (JSON + base64 JPEG)."""

import base64
import json
import urllib.error
import urllib.request


class SimClient:
    def __init__(self, url="http://127.0.0.1:8765", timeout=600):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def call(self, method, **params):
        data = json.dumps({"method": method, "params": params}).encode()
        req = urllib.request.Request(self.url + "/", data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            resp = json.loads(e.read())
        if not resp.get("ok"):
            raise RuntimeError(f"sim_server {method} failed: {resp.get('error')}")
        return resp["result"]

    def health(self):
        with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
            return json.loads(r.read())

    def reset(self, seed, record_dir=None):
        return self.call("reset", seed=seed, record_dir=record_dir)

    def move_to(self, targets):
        return self.call("move_to", targets=targets)

    def get_observation(self, include_third_person=False, marks=True, body_map=True):
        return self.call("get_observation", include_third_person=include_third_person, marks=marks, body_map=body_map)

    def get_state(self):
        return self.call("get_state")

    def measure(self, camera, x, y, label=""):
        return self.call("measure", camera=camera, x=x, y=y, label=label)

    def describe(self, marks=False, body_map=False):
        return self.call("describe", marks=marks, body_map=body_map)

    @staticmethod
    def decode_image(b64):
        return base64.b64decode(b64)


def make_video(frames_dir, out_path, fps=10):
    """Encode frame_%05d.jpg files (written by the server's record_dir) into an mp4 with ffmpeg."""
    import subprocess

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{frames_dir}/frame_%05d.jpg",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out_path]
    subprocess.run(cmd, check=True)
    return out_path


def make_episode_videos(record_dir, out_prefix, fps=10):
    """One mp4 per recorded camera plus a 2x2 mosaic (head | third-person / left wrist | right wrist)."""
    import os
    import subprocess

    cams = ["head", "third_person", "left_wrist", "right_wrist"]
    outs = {}
    for cam in cams:
        d = os.path.join(record_dir, cam)
        if os.path.isdir(d) and os.listdir(d):
            outs[cam] = make_video(d, f"{out_prefix}_{cam}.mp4", fps)
    if len(outs) == 4:
        inputs = []
        for cam in cams:
            inputs += ["-framerate", str(fps), "-i", f"{record_dir}/{cam}/frame_%05d.jpg"]
        filt = ";".join(f"[{i}:v]scale=480:360,drawtext=text='{cam}':x=8:y=8:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.5[v{i}]"
                        for i, cam in enumerate(cams))
        filt += ";[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]"
        mosaic = f"{out_prefix}_mosaic.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", filt, "-map", "[out]",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", mosaic], check=True)
        outs["mosaic"] = mosaic
    return outs
