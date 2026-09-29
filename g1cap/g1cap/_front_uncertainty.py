"""Conditional front-plane sensitivity; neither identity nor motion permission."""
import numpy as np

def uncertainty(line,point_error_m=.003,up_error_rad=np.deg2rad(.5),plane_band_m=.003):
    """Bound one connected fitted fragment under explicit boundary assumptions.

    Metres/radians. Assumes each selected raw sample is within point_error of
    the same true boundary, and that boundary is perpendicular to true source
    up within up_error of measured up. Its oriented normal must lie in the
    nominal normal's hemisphere; association must establish this separately.

    Let x be fitted-line tangent, y nominal normal, z measured source up.
    Endpoint residual bounds imply |n.x| <= 2*epsilon/span; up uncertainty
    implies |n.z| <= sin(up_error). Thus n.y >= sqrt(1-x^2-z^2).
    RMS*sqrt(N) bounds maximum line residual. Projecting raw samples onto the
    measured source plane adds at most plane_band*sin(up_error) normal error.
    Quantile endpoints are convex interpolations of fitted sample projections.
    Their midpoint therefore has signed true-plane error <=epsilon.
    """
    span=float(line['observed_span_m']);count=line['points'];rms=float(line['rms_m'])
    values=np.r_[span,count,rms,point_error_m,up_error_rad,plane_band_m]
    if (not np.isfinite(values).all() or span<=0 or isinstance(count,bool)
            or int(count)!=count or count<2 or min(rms,point_error_m,up_error_rad,plane_band_m)<0
            or up_error_rad>=np.pi/2):
        raise ValueError('invalid front uncertainty inputs')
    epsilon=float(point_error_m+np.sqrt(count)*rms+plane_band_m*np.sin(up_error_rad))
    squared=(2*epsilon/span)**2+np.sin(up_error_rad)**2
    if squared>=1:raise ValueError('front orientation uninformative')
    angle=float(np.arcsin(np.sqrt(squared)))
    return dict(anchor_error_m=epsilon,normal_angle_rad=angle,
        normal_difference=float(2*np.sin(angle/2)),span_m=span,
        assumed_point_error_m=float(point_error_m),assumed_up_error_rad=float(up_error_rad),
        assumed_plane_band_m=float(plane_band_m),identity_qualified=False)
