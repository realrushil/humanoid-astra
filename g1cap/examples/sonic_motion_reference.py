"""Small handwritten tool-contract probe, not a workstation task solution."""
def run(robot, task):
    robot.stop(timeout=8.)
    state=robot.observe()
    if state['episode_status']!='running' or state['settled_for']<.5:
        return
    joints=dict(zip(state['body_joint_names'],state['body_joint_positions']))
    elbow=joints['right_elbow_joint']
    yaw=state['pelvis_yaw']
    result=robot.sonic_motion([
        {'mode':'idle','duration':1.,'right_arm':{'right_elbow_joint':elbow+.12}},
        {'mode':'walk','duration':1.5,'velocity_world':[.08,0.],
         'facing_world':yaw+.10},
        {'mode':'idle','duration':1.,'facing_world':yaw,
         'right_arm':{'right_elbow_joint':elbow}},
    ])
    print('motion',result['status'],result['reason'],result.get('completion'))
    if robot.observe()['episode_status']=='running':
        result=robot.stop(timeout=8.)
        print('settle',result['status'],result['reason'])
