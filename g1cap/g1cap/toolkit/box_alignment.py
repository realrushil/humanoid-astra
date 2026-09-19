"""Common rigid rotation about measured box center; world metres/radians."""
import math
import numpy as np


class AlignmentPath:
    def __init__(self,center,rotation):
        self.center=np.asarray(center,dtype=float);R=np.asarray(rotation,dtype=float)
        if self.center.shape!=(3,) or R.shape!=(3,3) or not np.isfinite(self.center).all() or not np.isfinite(R).all():
            raise ValueError('finite center and rotation required')
        if not np.allclose(R.T@R,np.eye(3),atol=1e-6) or abs(np.linalg.det(R)-1)>1e-6:
            raise ValueError('orthonormal rotation required')
        k=int(np.argmax(abs(R[2])));normal=R[:,k]*np.sign(R[2,k])
        self.angle=math.acos(float(np.clip(normal[2],-1,1)))
        if self.angle>math.radians(45):raise ValueError('alignment exceeds45-degree alignment envelope')
        axis=np.cross(normal,[0.,0.,1.]);axis/=max(np.linalg.norm(axis),1e-12)
        x,y,z=axis;self.skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

    def delta(self,fraction):
        if not math.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('fraction outside0..1')
        a=self.angle*fraction
        return np.eye(3)+math.sin(a)*self.skew+(1-math.cos(a))*(self.skew@self.skew)
