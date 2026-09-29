"""Onboard RGB only: calibrated simulated mounts and bounded worker thumbnails.

Wrist mounts/rate are simulation assumptions, not measured hardware calibration.
No scene geometry, depth buffer or simulator link pose is accepted here.
"""
import base64
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from PIL import Image
import numpy as np

CAMERAS=('head','left_wrist','right_wrist')


def public_calibration(calibration):
    """Static intrinsics and robot-relative mount only; never a scene pose.

    Intrinsics refer to original pixels. Apply sample_stride_xy when using
    worker thumbnails. Partial calibration remains explicitly partial.
    """
    result={}
    for key,shape in (('intrinsic',(3,3)),('offset_position_m',(3,)),
                      ('offset_quaternion_xyzw',(4,))):
        if key in calibration:
            value=np.asarray(calibration[key],dtype=float)
            if value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('invalid camera calibration '+key)
            result[key]=value.tolist()
    for key in ('width','height'):
        if key in calibration:
            value=calibration[key]
            if type(value) is not int or not 0<value<=4096:raise ValueError('invalid calibration dimensions')
            result[key]=value
    for key,allowed in (('parent_frame',('head_link','left_wrist_yaw_link','right_wrist_yaw_link')),
                        ('offset_convention',('world','ros','opengl'))):
        if key in calibration:
            if calibration[key] not in allowed:raise ValueError('invalid camera calibration '+key)
            result[key]=calibration[key]
    return result


def configure_wrist_cameras(scene):
    """Mount above/behind each wrist; world convention is x-forward/z-up.

    Optical direction pitches 35 degrees down relative to the wrist. Copying
    head configuration preserves native sensor class/render settings, not pose.
    Cameras provide RGB only when read; recorder delivery uses head cadence.
    Zero native period avoids float32 elapsed-period misses (observed at16 s
    with a20 ms period). Strict native-buffer freshness checks remain in force.
    """
    for side in ('left','right'):
        camera=deepcopy(scene.robot_head_cam)
        camera.prim_path='{ENV_REGEX_NS}/Robot/'+side+'_wrist_yaw_link/rgb_camera'
        camera.width,camera.height=320,240
        camera.data_types=['rgb'];camera.update_period=0.
        camera.offset.pos=(-.06,0.,.09)
        angle=math.radians(35)/2
        # Installed CameraCfg.OffsetCfg uses XYZW (see native-sensor-api.json).
        camera.offset.rot=(0.,math.sin(angle),0.,math.cos(angle))
        camera.offset.convention='world'
        camera.spawn.focal_length=12.
        camera.spawn.clipping_range=(.02,3.)
        setattr(scene,side+'_wrist_cam',camera)


def rgb_packet(rgb,camera,step,time_s,calibration):
    """Nearest-neighbor sample <=48x32 RGB8, small enough for one worker reply.

    Pixel (u,v) samples original (stride_x*u,stride_y*v). Full-resolution RGB
    remains in the sensor archive and generation-time camera snapshots.
    """
    image=np.asarray(rgb)
    if camera not in CAMERAS or image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
        raise ValueError('expected permitted RGB8 camera')
    if not min(image.shape[:2]) or type(step) is not int or step<0 or not math.isfinite(time_s) or time_s<0:
        raise ValueError('invalid camera dimensions or capture time')
    sy,sx=math.ceil(image.shape[0]/32),math.ceil(image.shape[1]/48)
    sampled=image[::sy,::sx].copy()
    calibration=public_calibration(calibration)
    calibration_id=hashlib.sha256(json.dumps(calibration,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return dict(camera=camera,modality='rgb',step=step,time_s=time_s,
        width=sampled.shape[1],height=sampled.shape[0],original_width=image.shape[1],original_height=image.shape[0],
        sample_stride_xy=[sx,sy],encoding='base64_rgb8_row_major',
        rgb_base64=base64.b64encode(sampled.tobytes()).decode('ascii'),
        rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),calibration_id=calibration_id,
        source='live_camera',calibration=deepcopy(calibration))


def public_rgb(packet,now_s):
    if packet is None:return dict(status='unavailable',reason='camera_not_captured')
    age=now_s-packet['time_s']
    if not math.isfinite(age) or not 0<=age<=.150001:
        return dict(status='unavailable',reason='camera_stale',camera=packet['camera'])
    keys=('camera','modality','step','time_s','width','height','original_width','original_height',
          'sample_stride_xy','encoding','rgb_base64','rgb_sha256','calibration_id','source')
    return dict({k:deepcopy(packet[k]) for k in keys},status='available',age_s=age,
                calibration=public_calibration(packet['calibration']))


class RGBStream:
    """Record native RGB and clocks before handing immutable copies to workers."""
    def __init__(self,output,calibrations,physics_tick_s=.005):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.calibrations=deepcopy(calibrations)
        self.latest={};self.images={};self.clocks={}
        self.physics_tick_s=physics_tick_s
        (self.output/'calibration.json').write_text(json.dumps(calibrations,indent=2)+'\n')
        self.log=(self.output/'frames.jsonl').open('w')

    def capture(self,camera,image,step,time_s,clock):
        self.latest.pop(camera,None);self.images.pop(camera,None)
        sample=dict(clock,time_s=time_s)
        if not all(math.isfinite(v) for v in sample.values()):raise ValueError('nonfinite camera clock')
        if abs(clock['native_time_s']-clock['updated_at_s'])>1e-7:
            raise ValueError('native camera buffer stale')
        previous=self.clocks.get(camera)
        if previous is not None:
            dt=time_s-previous['time_s']
            ticks=math.ceil(max(0.,dt)/self.physics_tick_s)
            largest=max(abs(clock['native_time_s']),abs(previous['native_time_s']),self.physics_tick_s)
            tolerance=max(1e-7,(ticks+2)*float(np.spacing(np.float32(largest))))
            if (dt<=0 or clock['frame']<=previous['frame'] or
                    abs(clock['native_time_s']-previous['native_time_s']-dt)>tolerance):
                raise ValueError('native camera did not advance with physics')
        packet=rgb_packet(image,camera,step,time_s,self.calibrations[camera])
        path=self.output/f'{step:06d}-{camera}.png'
        Image.fromarray(image).save(path)
        self.log.write(json.dumps(dict(camera=camera,step=step,**sample,file=path.name,
            encoded_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rgb_sha256=packet['rgb_sha256'],
            calibration_id=packet['calibration_id']))+'\n');self.log.flush()
        self.latest[camera]=packet;self.images[camera]=image.copy();self.clocks[camera]=sample

    def close(self):self.log.close()
