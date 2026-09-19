"""Keep scalar robot joints separate from scene free bodies. No dynamics here."""


def robot_joint_ids(model):
    """Scalar joints in the pelvis subtree; object free joints are never motors."""
    root=model.body('pelvis').id
    return [j for j in range(model.njnt)
            if model.body_rootid[model.jnt_bodyid[j]]==root and model.jnt_type[j] in (2,3)]


def object_states(model,data):
    """Named free-body world poses and spatial velocities from simulator truth."""
    import mujoco as mj
    import numpy as np
    root=model.body('pelvis').id
    objects={}
    for j in range(model.njnt):
        body=int(model.jnt_bodyid[j])
        if model.jnt_type[j]!=mj.mjtJoint.mjJNT_FREE or body==root: continue
        velocity=np.zeros(6)
        mj.mj_objectVelocity(model,data,mj.mjtObj.mjOBJ_BODY,body,velocity,0)
        objects[model.body(body).name]=dict(position_world=data.xpos[body].tolist(),
            quaternion_wxyz=data.xquat[body].tolist(),linear_velocity_world=velocity[3:].tolist(),
            angular_velocity_world=velocity[:3].tolist())
    return objects


def geom_metadata(model,geom_id):
    """Identify the actual collision surface; several geoms can share one body."""
    import mujoco as mj
    geom_id=int(geom_id)
    mesh_id=int(model.geom_dataid[geom_id])
    return dict(id=geom_id,name=model.geom(geom_id).name or f'geom_{geom_id}',
                body=model.body(int(model.geom_bodyid[geom_id])).name,
                mesh=model.mesh(mesh_id).name if model.geom_type[geom_id]==mj.mjtGeom.mjGEOM_MESH else None)
