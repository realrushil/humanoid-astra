"""Measured placement screen with explicit required and preferred clearance in world metres; not a collision certificate."""
import math
from g1cap.arena_surfaces import transformed_bounds,contained,separation


def placement_retraction(obs,geometry,supports,surface_id='destination',*,arm_margin_m=.005,preferred_arm_margin_m=None,arm_clearance=None):
    if not math.isfinite(arm_margin_m) or not .005<=arm_margin_m<=.02:
        raise ValueError('arm margin outside5–20mm planning range')
    preferred=arm_margin_m if preferred_arm_margin_m is None else preferred_arm_margin_m
    if not math.isfinite(preferred) or not arm_margin_m<=preferred<=.02:
        raise ValueError('preferred margin must be between the hard margin and 20 mm')
    best=None
    table=obs['surfaces'][surface_id]['bounds'];box=obs['box_bounds']
    drop=obs['surfaces'][surface_id]['clearance_m']+.002
    if not 0<drop<=.142:raise ValueError('outside_lowering_screen')
    axis=[obs['root_pos'][i]-obs['box_pos'][i] for i in range(2)]
    length=math.hypot(*axis)
    if length<.1:raise ValueError('root_box_direction_too_short')
    axis=[v/length for v in axis]
    arms=[transformed_bounds(shape,obs['body_poses'][name])
          for name,shapes in geometry['robot'].items()
          if any(w in name for w in ('shoulder','elbow','wrist','hand')) for shape in shapes]
    if not arms:raise ValueError('missing_arm_collision_geometry')
    parts=[b for ps in supports.values() for b in ps]
    # Native worlds supply cached source-hull/oriented-primitive distances.
    # A bounds-only caller keeps the conservative dependency-light screen.
    refined_gap=arm_clearance.for_pose(obs,drop) if arm_clearance is not None else None
    # Prefer the smallest displacement attaining the desired reserve. If none
    # does, maximize clearance among candidates satisfying both hard margins.
    # Placement has an 8 cm qualified development ceiling. The 0.5 mm search
    # grid contains no scenario coordinates or saved offsets.
    for i in range(161):
        distance=i*.0005;delta=[distance*axis[0],distance*axis[1],0.]
        def shift(bound,z):
            return {key:[v+delta[j]-(z if j==2 else 0.) for j,v in enumerate(bound[key])] for key in ('min','max')}
        moved=shift(box,0.)
        if not contained(moved,table,.01):continue
        # Lowering proxy shifts measured arm bounds down by box clearance.
        # Actual leg/arm deflection and balance are qualified in physics, not here.
        gap=(refined_gap(delta) if refined_gap is not None else
             min(separation(shift(arm,drop),part) for arm in arms for part in parts))
        if gap>=arm_margin_m:
            margin=min(min(moved['min'][j]-table['min'][j],table['max'][j]-moved['max'][j]) for j in range(2))
            candidate=dict(distance_m=distance,translation_world=delta,predicted_arm_clearance_m=gap,predicted_box_margin_m=margin)
            if gap>=preferred:return candidate
            # Maximize attainable reserve when the preference is unreachable.
            # Numerical ties (1e-12m) retain the earlier, smaller displacement.
            if best is None or gap>best['predicted_arm_clearance_m']+1e-12:
                best=candidate
    return best
