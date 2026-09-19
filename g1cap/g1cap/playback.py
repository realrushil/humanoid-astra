"""Render a SONIC JSONL recording offline as a labelled MP4."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess


def goal_markers(episode):
    """Convert saved task goals to world coordinates for display only."""
    episode = Path(episode)
    task = json.loads((episode/'task.json').read_text())
    frame = json.loads((episode/'manifest.json').read_text())['episode_frame']
    x, y = frame['origin_xy']
    c, s = math.cos(frame['origin_yaw']), math.sin(frame['origin_yaw'])
    targets = task.get('waypoints') or [task['target_position'][:2]]
    return [(x+c*px-s*py, y+s*px+c*py, .025) for px,py in targets]


def read_recording(path, start_time=0.0):
    """Return metadata and monotonically ordered frames at or after start_time."""
    metadata = {}
    frames = []
    previous = None
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            if item.get('record_type') == 'metadata':
                metadata.update(item)
                continue
            if item.get('record_type') != 'frame' or float(item['sim_time']) < start_time:
                continue
            item['sim_time'] = float(item['sim_time'])
            item['qpos'] = [float(value) for value in item['qpos']]
            if previous is not None and item['sim_time'] < previous:
                raise ValueError('recording frame times are not monotonic')
            previous = item['sim_time']
            frames.append(item)
    return metadata, frames


def render(repo, poses, out, start_time=0.0, episode=None):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    import numpy as np
    metadata, frames = read_recording(poses, start_time)
    if not frames:
        raise ValueError('recording contains no frames at or after start time')
    goals = goal_markers(episode) if episode else []
    model_path = Path(metadata.get('model_path', ''))
    if not model_path.is_file():
        candidate = Path(repo) / metadata.get('model_path', '')
        model_path = candidate if candidate.is_file() else Path(repo) / 'gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'
    if not model_path.is_file():
        raise RuntimeError('recording model_path is unavailable; provide the upstream --repo')
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.0, 0.0, 0.8]
    camera.distance = 3.5
    camera.azimuth = 135
    camera.elevation = -20
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0  # Hide the upstream decorative red COM site.
    label = 'Recorded simulation playback'
    if goals:
        label += ' | gold=final goal, blue=earlier waypoint'
    ffmpeg = subprocess.Popen(['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                               '-s', '1280x720', '-r', '30', '-i', '-', '-an',
                               '-vf', f"drawtext=text='{label}':x=20:y=20:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5",
                               '-metadata', 'title=Recorded simulation playback',
                               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out)], stdin=subprocess.PIPE)
    try:
        next_frame = 0
        current = frames[0]
        end_time = frames[-1]['sim_time']
        target = current['sim_time']
        while target <= end_time + 1e-9:
            while next_frame + 1 < len(frames) and frames[next_frame + 1]['sim_time'] <= target:
                next_frame += 1
                current = frames[next_frame]
            frame = current
            if len(frame['qpos']) != model.nq:
                raise ValueError(f"qpos length {len(frame['qpos'])} does not match model.nq {model.nq}")
            data.qpos[:] = frame['qpos']
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=options)
            for index, position in enumerate(goals):
                scene = renderer.scene
                color = [.95,.65,.10,.85] if index == len(goals)-1 else [.15,.65,1.,.85]
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                                   [.10,.015,0.], position, np.eye(3).ravel(), color)
                scene.ngeom += 1
            ffmpeg.stdin.write(renderer.render().tobytes())
            target += 1.0 / 30.0
    finally:
        if ffmpeg.stdin:
            ffmpeg.stdin.close()
        if ffmpeg.wait() != 0:
            raise RuntimeError('ffmpeg failed while encoding playback')
        renderer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--poses', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--start-time', type=float, default=0.0)
    parser.add_argument('--episode', help='episode directory for task goal markers')
    args = parser.parse_args(argv)
    render(args.repo, args.poses, args.out, args.start_time, args.episode)


if __name__ == '__main__':
    raise SystemExit(main())
