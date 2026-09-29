"""Strict RGB-D observation of one finite destination-surface candidate.

The caller supplies a semantic/color candidate mask from the onboard image. This
module validates only the measured RGB-D geometry: one connected candidate,
finite depth, a single visible plane and a finite two-dimensional extent. It
does not infer a world pose, support/contact, destination route, or release
success. Optical coordinates use metres with ``x`` right, ``y`` down and ``z``
forward.
"""

import math
from copy import deepcopy

import numpy as np
from scipy.ndimage import label

from .rgbd_geometry import depth_points, fit_planes


def green_destination_mask(rgb):
    """Return a conservative semantic mask for the declared green destination.

    This is a color cue only; it does not encode a world pose or table bounds.
    The thresholds are deliberately strict and operate on uint8 RGB pixels.
    """
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('RGB must be uint8 HxWx3')
    red, green, blue = [image[..., i].astype(np.int16) for i in range(3)]
    return (green >= 45) & (green >= red + 12) & (green >= blue + 8)


def destination_direction_segment(candidate, camera_to_body, body_to_segment, *, camera_position_body_m=(0.,0.,0.)):
    """Pelvis-to-candidate direction in local-segment axes, at capture time.

    ``camera_to_body`` and ``body_to_segment`` are calibrated 3x3 rotations;
    they are static/encoder-derived transforms, never simulator object poses.
    Camera translation is in body metres; zero is for coincident-origin callers.
    A near-vertical displacement or malformed matrix is rejected rather than projected
    into navigation.  The returned direction has no range or world position.
    """
    if candidate.get('status') != 'observed_destination_candidate':
        raise ValueError('destination_observation_unavailable')
    ray = np.asarray(candidate.get('centroid_camera_m'), float)
    first = np.asarray(camera_to_body, float)
    second = np.asarray(body_to_segment, float)
    translation=np.asarray(camera_position_body_m,float)
    if ray.shape != (3,) or translation.shape!=(3,) or first.shape != (3, 3) or second.shape != (3, 3):
        raise ValueError('destination_direction_invalid')
    if not all(np.isfinite(a).all() for a in (ray,first,second,translation)):
        raise ValueError('destination_direction_invalid')
    for rotation in (first,second):
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6):
            raise ValueError('destination_direction_invalid_rotation')
    displacement=second@(first@ray+translation)
    norm = np.linalg.norm(displacement)
    if norm <= 1e-6:
        raise ValueError('destination_direction_invalid')
    direction = displacement/norm
    planar = direction[:2]
    planar_norm = np.linalg.norm(planar)
    if planar_norm < .5:
        raise ValueError('destination_direction_near_vertical')
    planar /= planar_norm
    return planar.tolist()


def observe_destination_surface(time_s, rgb, candidate_mask, depth_m, intrinsic):
    """Return one conservative finite surface candidate or ``unavailable``."""
    if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s) or time_s < 0:
        raise ValueError('invalid observation time')
    image = np.asarray(rgb)
    mask = np.asarray(candidate_mask)
    depth = np.asarray(depth_m, float)
    k = np.asarray(intrinsic, float)
    if image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2] or mask.dtype != np.bool_:
        raise ValueError('RGB and candidate mask shape/type mismatch')
    if depth.shape != mask.shape or k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError('invalid depth or camera calibration')
    if not np.isfinite(image).all():
        raise ValueError('RGB contains nonfinite values')
    if not mask.any():
        return dict(status='unavailable', reason='destination_candidate_missing')
    components, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count != 1:
        return dict(status='unavailable', reason='destination_candidate_ambiguity', components=int(count))
    valid = mask & np.isfinite(depth) & (depth > 0)
    valid_fraction = float(valid.sum() / mask.sum())
    if valid.sum() < 200 or valid_fraction < .8:
        return dict(status='unavailable', reason='destination_depth_incomplete',
                    valid_points=int(valid.sum()), valid_fraction=valid_fraction)
    points = depth_points(depth, k)[valid]
    planes = fit_planes(points, threshold_m=.003, max_planes=1, min_points=200)
    if len(planes) != 1 or planes[0]['rms_m'] > .003:
        return dict(status='unavailable', reason='destination_plane_unavailable')
    plane = planes[0]
    plane_fraction=len(plane['points'])/len(points)
    # Membership is still an ideal-depth development assumption. A small fit
    # cannot lend its normal to a large unmodeled candidate or its outliers.
    if plane_fraction<.9:
        return dict(status='unavailable',reason='destination_plane_coverage',plane_fraction=plane_fraction)
    points=plane['points']
    normal = np.asarray(plane['normal'], float)
    # Build deterministic tangent axes from the camera optical basis.
    ref = np.array([1., 0., 0.]) if abs(normal[0]) < .9 else np.array([0., 1., 0.])
    tangent = np.cross(normal, ref)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    coordinates = points @ np.stack([tangent, bitangent], axis=1)
    spans = np.ptp(coordinates, axis=0)
    if np.any(spans < .15):
        return dict(status='unavailable', reason='destination_extent_insufficient',
                    observed_spans_m=spans.tolist())
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    centroid = points.mean(axis=0)
    return dict(status='observed_destination_candidate', observed_at_s=float(time_s),
                centroid_camera_m=centroid.tolist(),
                range_camera_m=float(np.linalg.norm(centroid)),
                normal_camera=normal.tolist(), offset_m=float(plane['offset_m']),
                tangent_camera=tangent.tolist(), bitangent_camera=bitangent.tolist(),
                extent_min_m=minimum.tolist(), extent_max_m=maximum.tolist(),
                observed_spans_m=spans.tolist(), complete_footprint_observed=False,
                extent_kind='visible_plane_patch_only',plane_fraction=plane_fraction,
                valid_points=int(valid.sum()), valid_fraction=valid_fraction,
                plane_rms_m=float(plane['rms_m']),
                association='single_connected_candidate_mask',
                frame='camera_optical_at_observation_time')


def observe_level_destination_surface(time_s, rgb, depth_m, intrinsic, up_camera):
    """Select one level patch of a green object, excluding its vertical sides.

    Up is a unit vector from synchronized onboard attitude and camera
    calibration. Five degrees and 3 mm are ideal-sensor assumptions, not
    hardware calibration. The selected patch has no complete-footprint or
    free-space meaning; multiple level planes remain ambiguous.
    """
    up=np.asarray(up_camera,float)
    if up.shape!=(3,) or not np.isfinite(up).all() or abs(np.linalg.norm(up)-1)>1e-6:
        raise ValueError('unit camera-time up vector required')
    mask=green_destination_mask(rgb)
    # Keep original input/depth/semantic-component validation before selection.
    original=observe_destination_surface(time_s,rgb,mask,depth_m,intrinsic)
    if original.get('reason') in ('destination_candidate_missing',
            'destination_candidate_ambiguity','destination_depth_incomplete'):
        return original
    depth=np.asarray(depth_m,float)
    valid=mask & np.isfinite(depth) & (depth>0)
    cloud=depth_points(depth,intrinsic)
    points=cloud[valid]
    planes=fit_planes(points,threshold_m=.003,max_planes=4,min_points=200)
    # Exhausting the extraction budget is not proof that no second level
    # surface exists. Unexplained geometry large enough for another candidate
    # makes selection unavailable, even when the first four fits look usable.
    explained=np.zeros(len(points),bool)
    for plane in planes:
        explained|=abs(points@plane['normal']+plane['offset_m'])<=.003
    unresolved=int((~explained).sum())
    if unresolved>=200:
        return dict(status='unavailable',reason='destination_unresolved_geometry',
                    unresolved_points=unresolved)
    level=[p for p in planes if abs(p['normal']@up)>=np.cos(np.deg2rad(5.))]
    if len(level)!=1:
        return dict(status='unavailable',reason='destination_level_plane_ambiguity',
                    level_candidates=len(level))
    plane=level[0]
    selected=valid & (abs(cloud@plane['normal']+plane['offset_m'])<=.003)
    result=observe_destination_surface(time_s,rgb,selected,depth,intrinsic)
    if result['status']=='observed_destination_candidate':
        angle=float(np.degrees(np.arccos(np.clip(abs(np.asarray(result['normal_camera'])@up),0.,1.))))
        if angle>5.:
            return dict(status='unavailable',reason='destination_level_refit_changed')
        result.update(selection='unique_observed_level_color_plane',
                      color_points=int(valid.sum()),level_points=int(selected.sum()),
                      level_angle_deg=angle)
    return result


class DestinationSurfaceStream:
    """Keep one candidate only within a declared camera/view epoch."""

    def __init__(self, *, max_gap_s=.150001):
        if not math.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError('invalid destination stream gap')
        self.max_gap_s = float(max_gap_s)
        self.epoch = None
        self.last_time = None
        self.invalid = False

    def initialize(self, sample, *, view_epoch):
        if sample.get('status') != 'observed_destination_candidate':
            raise ValueError('destination_candidate_unavailable')
        if isinstance(view_epoch, bool) or not isinstance(view_epoch, int):
            raise ValueError('invalid destination view epoch')
        t=sample.get('observed_at_s')
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t<0:
            raise ValueError('invalid destination observation time')
        self.epoch = view_epoch
        self.last_time = float(sample['observed_at_s'])
        self.invalid = False
        return dict(sample, stream_status='initialized', view_epoch=view_epoch)

    def submit(self, sample, *, view_epoch):
        if self.invalid or self.epoch is None:
            raise ValueError('destination_stream_uninitialized')
        if view_epoch != self.epoch:
            self.invalid = True
            raise ValueError('destination_view_epoch_changed')
        if sample.get('status') != 'observed_destination_candidate':
            self.invalid = True
            raise ValueError('destination_candidate_unavailable')
        time_s = sample.get('observed_at_s')
        if (not isinstance(time_s, (int, float)) or isinstance(time_s, bool) or
                not math.isfinite(time_s) or not 0 < time_s - self.last_time <= self.max_gap_s):
            self.invalid = True
            raise ValueError('destination_observation_gap')
        self.last_time = float(time_s)
        return dict(sample, stream_status='tracked', view_epoch=view_epoch)


class DestinationObservation:
    """Fit once per image; preserve optical geometry when localization is absent.

    Epochs mark discontinuity, not proven semantic identity. Reacquisition after
    loss/segment change starts a new epoch explicitly; consumers cannot continue
    a motion objective across it. This observation alone never admits navigation.
    """
    def __init__(self):
        self.stream=DestinationSurfaceStream()
        self.last_step=None;self.segment=None;self.epoch=0
        self.latest=dict(status='unavailable',reason='destination_not_observed')

    def update(self,camera,packet,camera_pose,motion,*,up_body=None):
        step=camera['step'];t=camera['time_s']
        if step==self.last_step:return
        if self.last_step is not None and step<self.last_step:
            self.stream.invalid=True
            self.latest=dict(status='unavailable',reason='destination_time_reversed')
            return
        self.last_step=step
        if up_body is None:
            sample=observe_destination_surface(t,camera['rgb'],green_destination_mask(camera['rgb']),
                                               camera['depth_m'],camera['calibration']['intrinsic'])
        else:
            if packet['step']!=step or abs(packet['time_s']-t)>1e-8:
                self.stream.invalid=True
                self.latest=dict(status='unavailable',reason='destination_attitude_time_mismatch')
                return
            from .scene_motion import _rigid
            pose=_rigid(camera_pose)
            up=np.asarray(up_body,float)
            if up.shape!=(3,):raise ValueError('three-dimensional camera-time up required')
            sample=observe_level_destination_surface(t,camera['rgb'],camera['depth_m'],
                camera['calibration']['intrinsic'],pose[:3,:3].T@up)
        if sample['status']!='observed_destination_candidate':
            self.stream.invalid=True;self.latest=sample;return
        segment=motion.get('segment') if motion is not None and motion.get('status')=='tracked_local_segment' else None
        continuous=(not self.stream.invalid and self.stream.epoch is not None and self.segment==segment
                    and 0<t-self.stream.last_time<=self.stream.max_gap_s)
        if continuous:sample=self.stream.submit(sample,view_epoch=self.epoch)
        else:
            self.epoch+=1
            sample=self.stream.initialize(sample,view_epoch=self.epoch)
        self.segment=segment
        sample['direction_status']='unavailable'
        synchronized=(packet['step']==step and abs(packet['time_s']-t)<=1e-8 and
                      motion is not None and motion.get('status')=='tracked_local_segment' and
                      abs(motion.get('time_s',float('nan'))-t)<=1e-8)
        if synchronized:
            try:
                pose=np.asarray(camera_pose,float);body=np.asarray(motion['body_in_segment'],float)
                if pose.shape!=(4,4) or body.shape!=(4,4):raise ValueError('invalid transform')
                sample['direction_segment_xy']=destination_direction_segment(sample,pose[:3,:3],body[:3,:3],
                    camera_position_body_m=pose[:3,3])
                sample.update(direction_status='available',segment_id=segment,
                    direction_frame='local_stance_segment_at_observation_time',
                    direction_origin='pelvis_at_observation_time')
            except (ValueError,TypeError,KeyError):pass
        self.latest=sample

    def observe(self,now_s):
        if self.latest['status']!='observed_destination_candidate':return deepcopy(self.latest)
        age=now_s-self.latest['observed_at_s']
        if not math.isfinite(age) or not 0<=age<=self.stream.max_gap_s:
            return dict(status='unavailable',reason='destination_observation_stale')
        return dict(deepcopy(self.latest),age_s=age)
