"""Small, immutable camera packets. No simulator, model or scoring dependencies."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
import struct


def _frame_bytes(frame, directory):
    relative=Path(frame['file'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('camera file must stay inside its input directory')
    root=Path(directory).resolve();path=root/relative
    if any(p.is_symlink() for p in (path,*path.parents) if p!=root and root in p.parents):
        raise ValueError('camera symlinks are not accepted')
    if not path.resolve().is_relative_to(root) or not path.is_file() or path.stat().st_size>8*1024*1024:
        raise ValueError('missing or oversized camera file')
    content=path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=frame['encoded_sha256']:
        raise ValueError('camera bytes changed after capture')
    if (len(content)<33 or content[:8]!=b'\x89PNG\r\n\x1a\n' or content[12:16]!=b'IHDR'
            or struct.unpack('>II',content[16:24])!=(frame['width'],frame['height'])):
        raise ValueError('camera PNG dimensions do not match the manifest')
    return content


def validate_snapshot(snapshot, directory, session_id):
    """Validate one published pair against its exact structured-state capture."""
    try:
        if snapshot['version']!=1 or snapshot['session_id']!=session_id:
            raise ValueError('snapshot version or session mismatch')
        if not re.fullmatch('[0-9a-f]{32}',snapshot['snapshot_id']):
            raise ValueError('invalid snapshot identity')
        for key in ('step','physics_step'):
            if type(snapshot[key]) is not int or snapshot[key]<0:raise ValueError('invalid capture step')
        for key in ('sim_time_s','captured_at_unix_s'):
            value=snapshot[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('invalid capture time')
        obs=snapshot['observation']
        if (obs['session_id']!=session_id or obs['step']!=snapshot['step'] or
                obs['physics_step']!=snapshot['physics_step'] or obs['time']!=snapshot['sim_time_s']):
            raise ValueError('images and observation must describe the same step')
        expected=['head'] if obs.get('observation_mode')=='sensor_estimates_v1' else ['head','overview']
        if [f['camera'] for f in snapshot['frames']]!=expected:
            raise ValueError('snapshot camera set differs from observation mode')
        for frame in snapshot['frames']:
            if frame['source'] not in ('live_camera','decoded_recording'):raise ValueError('invalid image provenance')
            for key in ('encoded_sha256','rgb_sha256'):
                if not re.fullmatch('[0-9a-f]{64}',frame[key]):raise ValueError('invalid image hash')
            if any(type(frame[k]) is not int or not 0<frame[k]<=4096 for k in ('width','height')):
                raise ValueError('invalid image dimensions')
            _frame_bytes(frame,directory)
    except (KeyError,TypeError,OSError,struct.error) as error:
        raise ValueError('invalid or unreadable camera packet') from error


def stage_visual_observations(snapshots, directory, resources, session_id):
    """Copy at most previous/current pairs; return manifests relative to resources."""
    if not 1<=len(snapshots)<=2:raise ValueError('provide one or two camera snapshots')
    for packet in snapshots:validate_snapshot(packet,directory,session_id)
    if len(snapshots)==2 and (snapshots[0]['step']>=snapshots[1]['step'] or
                             snapshots[0]['sim_time_s']>=snapshots[1]['sim_time_s']):
        raise ValueError('previous camera observation must precede current')
    copied=deepcopy(snapshots)
    destination=Path(resources)/'observations';destination.mkdir(exist_ok=False)
    for index,packet in enumerate(copied):
        for frame in packet['frames']:
            content=_frame_bytes(frame,directory)
            relative=Path('observations')/f'{index}-{frame["camera"]}.png'
            (Path(resources)/relative).write_bytes(content)
            frame['file']=relative.as_posix()
    return copied
