"""Local paired wrist-pose IK. A kinematic plan is not a grasp or balance proof."""
import math
from .arm_planner import ArmPlanner, ARM_SUFFIXES, UPPER_BODY
from ..models import finite_number

ARMS=tuple(f'{side}_{s}_joint' for side in ('left','right') for s in ARM_SUFFIXES)


def paired_reference(sample,positions):
    """14 left-then-right arm radians into SONIC's interleaved 17-joint order."""
    if len(positions)!=14 or not all(finite_number(v) for v in positions):
        raise ValueError('14 finite paired arm positions required')
    joints=dict(zip(sample['body_joint_names'],sample['body_joint_positions']))
    joints.update(zip(ARMS,positions))
    return [joints[name] for name in UPPER_BODY]


def validate_goals(goals):
    """Exactly two world wrist poses, metres and unit WXYZ quaternions."""
    if not isinstance(goals,dict) or set(goals)!={'left','right'}:
        raise ValueError('left and right pose goals required')
    for goal in goals.values():
        if not isinstance(goal,dict):raise ValueError('pose goal must be a dictionary')
        p,q=goal.get('position'),goal.get('quaternion')
        if (not isinstance(p,(list,tuple)) or len(p)!=3 or not isinstance(q,(list,tuple)) or len(q)!=4
                or not all(finite_number(v) for v in (*p,*q)) or abs(math.hypot(*q)-1.)>.01):
            raise ValueError('finite world position and unit WXYZ quaternion required')


def hand_errors(sample,goals):
    """Measured wrist errors, using the shorter quaternion rotation angle."""
    errors={}
    for side,goal in goals.items():
        p=sample.get(side+'_wrist_position');q=sample.get(side+'_wrist_quaternion_wxyz')
        velocity=sample.get(side+'_wrist_velocity_world')
        for value,size in ((p,3),(q,4),(velocity,3)):
            if not isinstance(value,(list,tuple)) or len(value)!=size or not all(finite_number(v) for v in value):
                raise ValueError('invalid_hand_state')
        if abs(math.hypot(*q)-1.)>.01:raise ValueError('invalid_hand_state')
        target=goal['quaternion']
        errors[side]=dict(position_m=math.dist(p,goal['position']),
                          orientation_rad=quaternion_error(q,target),speed_m_s=math.hypot(*velocity))
    return errors


def quaternion_error(q,target):
    """Shortest rotation distance in radians; quaternion signs are equivalent."""
    cosine=abs(sum(a*b for a,b in zip(q,target)))/(math.hypot(*q)*math.hypot(*target))
    return 2*math.acos(min(1.,cosine))


def hands_at_goal(sample,goals):
    return all(e['position_m']<=.02 and e['orientation_rad']<=.15 and e['speed_m_s']<=.05
               for e in hand_errors(sample,goals).values())


class BimanualPlanner(ArmPlanner):
    """Freeze measured root/waist/fingers/objects; solve both wrist origins + orientation.

    Positions are world metres. Quaternions are unit WXYZ body-to-world.
    Orientation residuals are world rotation vectors, weighted at 0.12 m/rad.
    Uses the same declared collision geometry and bounded step checking as the
    right-arm planner. No intentional contact is permitted by this planner yet.
    """
    def __init__(self,model_path):
        super().__init__(model_path)
        self.joints=[self.model.joint(n).id for n in ARMS]
        self.q_indices=self.model.jnt_qposadr[self.joints]
        self.v_indices=self.model.jnt_dofadr[self.joints]
        if not all(self.model.jnt_limited[j] for j in self.joints):
            raise ValueError('both arms require explicit joint limits')
        self.lower=self.model.jnt_range[self.joints,0]+.005
        self.upper=self.model.jnt_range[self.joints,1]-.005
        self.wrists={s:self.model.body(s+'_wrist_yaw_link').id for s in ('left','right')}

    def plan(self,sample,goals):
        from scipy.spatial.transform import Rotation
        np,mj=self.np,self.mj
        validate_goals(goals)
        targets={}
        for side,goal in goals.items():
            p,q=goal['position'],goal['quaternion']
            if len(p)!=3 or len(q)!=4 or not all(finite_number(v) for v in (*p,*q)) or abs(math.hypot(*q)-1.)>.01:
                raise ValueError('finite world position and unit WXYZ quaternion required')
            targets[side]=(np.asarray(p),Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix())
        self._load(sample)
        if any(np.linalg.norm(targets[s][0]-self.data.xpos[b])>.4 for s,b in self.wrists.items()):
            return dict(status='rejected',reason='target_outside_local_envelope',joint_path=[])
        collision=self._collision()
        if collision: return dict(status='rejected',reason='initial_model_collision',joint_path=[],collision=collision)
        q=self.data.qpos[self.q_indices].copy()
        if np.any(q<self.lower) or np.any(q>self.upper):
            return dict(status='rejected',reason='initial_joint_limit',joint_path=[])
        path=[q.tolist()]
        for _ in range(320):
            rows=[]; residual=[]; errors={}
            for side,body in self.wrists.items():
                p,rotation=targets[side]
                ep=p-self.data.xpos[body]
                er=Rotation.from_matrix(rotation@self.data.xmat[body].reshape(3,3).T).as_rotvec()
                errors[side]=dict(position_m=float(np.linalg.norm(ep)),orientation_rad=float(np.linalg.norm(er)))
                jp,jr=np.zeros((3,self.model.nv)),np.zeros((3,self.model.nv))
                mj.mj_jacBody(self.model,self.data,jp,jr,body)
                rows.extend((jp[:,self.v_indices],.12*jr[:,self.v_indices]))
                residual.extend((ep,.12*er))
            if all(e['position_m']<=.005 and e['orientation_rad']<=.03 for e in errors.values()):
                return dict(status='planned',reason='paired_local_ik',joint_names=list(ARMS),joint_path=path,errors=errors)
            jac=np.vstack(rows);error=np.concatenate(residual)
            delta=jac.T@np.linalg.solve(jac@jac.T+.0001*np.eye(12),error)
            candidate,collision,_=self._clear_step(q,delta)
            if candidate is None:
                return dict(status='rejected',reason='path_model_collision',joint_path=[],errors=errors,collision=collision)
            q=candidate;self.data.qpos[self.q_indices]=q;mj.mj_forward(self.model,self.data)
            path.append(q.tolist())
        return dict(status='rejected',reason='paired_ik_not_converged',joint_path=[],errors=errors)
