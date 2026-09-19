import importlib.util
from pathlib import Path
import tempfile
import unittest


class ObjectContactTests(unittest.TestCase):
    def test_object_support_is_not_a_robot_fault_but_robot_box_contact_is(self):
        from g1cap.sonic_sim import summarize_contacts
        support=dict(body1='table',body2='box',distance=-.003,normal_force=10.,robot_contact=False)
        hand=dict(body1='right_wrist_yaw_link',body2='box',distance=-.003,normal_force=10.,robot_contact=True)
        self.assertEqual(summarize_contacts([support])['forbidden_contacts'],[])
        self.assertEqual(summarize_contacts([hand])['forbidden_contacts'],[hand])


@unittest.skipUnless(importlib.util.find_spec('mujoco'),'MuJoCo installed on Linux')
class MovableBoxTests(unittest.TestCase):
    def test_free_box_preserves_robot_interface_and_full_pose(self):
        import mujoco as mj
        from g1cap.scene import box_scene,write_scene_model
        from g1cap.sim_state import robot_joint_ids,object_states
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'robot.xml'
            source.write_text('''<mujoco><worldbody>
            <body name="pelvis" pos="-2 0 1"><freejoint/><geom type="sphere" size=".1"/>
            <body name="arm"><joint name="elbow"/><geom type="sphere" size=".05"/></body>
            </body></worldbody></mujoco>''')
            plain=mj.MjModel.from_xml_path(str(source))
            model=mj.MjModel.from_xml_path(str(write_scene_model(source,Path(folder)/'scene.xml',box_scene())))
            data=mj.MjData(model); mj.mj_forward(model,data)
            self.assertEqual(model.nq,plain.nq+7)
            self.assertEqual(model.nv,plain.nv+6)
            self.assertEqual(model.nu,plain.nu)
            self.assertEqual([model.joint(j).name for j in robot_joint_ids(model)],['elbow'])
            self.assertEqual(model.joint('elbow').qposadr,plain.joint('elbow').qposadr)
            objects=object_states(model,data)
            self.assertEqual(list(objects),['parcel'])
            self.assertEqual(len(objects['parcel']['quaternion_wxyz']),4)
            self.assertGreater(model.body('parcel').mass[0],0.)


    def test_contact_geometry_distinguishes_palm_from_wrist_on_same_body(self):
        import mujoco as mj
        from g1cap.sim_state import geom_metadata
        vertices='0 0 0  .02 0 0  0 .02 0  0 0 .02'
        model=mj.MjModel.from_xml_string(f'''<mujoco><asset>
        <mesh name="wrist_mesh" vertex="{vertices}"/><mesh name="palm_mesh" vertex="{vertices}"/>
        </asset><worldbody><body name="wrist"><geom name="wrist_surface" type="mesh" mesh="wrist_mesh"/>
        <geom name="palm_surface" type="mesh" mesh="palm_mesh" pos=".1 0 0"/></body>
        <geom name="parcel" type="box" size=".1 .1 .1" pos="1 0 0"/></worldbody></mujoco>''')
        wrist=geom_metadata(model,model.geom('wrist_surface').id)
        palm=geom_metadata(model,model.geom('palm_surface').id)
        self.assertEqual(wrist['body'],palm['body'])
        self.assertNotEqual(wrist['mesh'],palm['mesh'])
        self.assertEqual(palm['mesh'],'palm_mesh')
        self.assertIsNone(geom_metadata(model,model.geom('parcel').id)['mesh'])
