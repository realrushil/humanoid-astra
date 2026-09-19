import copy, json, math, unittest
from pathlib import Path
from g1cap import arena_scene
from g1cap.arena_scene import TABLES, table_layout

def transfer_scene():
    tables=table_layout('two_empty_equal_height_tables_v1')
    return dict(tables=tables,box=dict(size_m=[.2,.2,.2],mass_kg=.1,
                center_xy=[.60,.18],xy_jitter_m=.01,yaw_offset_rad=0.))

class SceneTests(unittest.TestCase):
    def test_all_declared_transfer_cases_have_valid_nonoverhanging_scenes(self):
        directory=Path(__file__).resolve().parents[1]/'examples/box-transfer'
        suite=json.loads((directory/'suite.json').read_text())
        for case in suite['cases']:
            with self.subTest(recipe=case['recipe']):
                recipe=json.loads((directory/case['recipe']).read_text())
                tables=table_layout(recipe['fixture'],scene=recipe['scene'])
                self.assertGreater(tables['source']['top_z'],arena_scene.FLOOR_Z)

    def test_mass_is_authored_on_the_rigid_child_not_the_usd_container(self):
        try:
            from pxr import Usd,UsdGeom,UsdPhysics
        except ImportError:
            self.skipTest('USD bindings are exercised on the Arena Linux runtime')
        stage=Usd.Stage.CreateInMemory()
        root=UsdGeom.Xform.Define(stage,'/Box').GetPrim()
        body=UsdGeom.Xform.Define(stage,'/Box/Geometry/body').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        arena_scene.author_box_mass(root,.5)
        self.assertFalse(root.HasAPI(UsdPhysics.MassAPI))
        self.assertEqual(UsdPhysics.MassAPI(body).GetMassAttr().Get(),.5)
        other=UsdGeom.Xform.Define(stage,'/Box/other').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(other)
        with self.assertRaises(ValueError):arena_scene.author_box_mass(root,.2)

    def test_explicit_scene_is_validated_and_copied(self):
        scene=transfer_scene();scene['tables']['destination']['top_z']=.02
        result=table_layout('box_transfer_v1',scene=scene)
        self.assertAlmostEqual(result['destination']['top_z'],.02)
        result['source']['center_xy']=[99,99]
        self.assertEqual(scene['tables']['source']['center_xy'],(.75,.18))

    def test_invalid_scene_rejects_overhang_overlap_and_bad_physics(self):
        invalid=[]
        scene=transfer_scene();scene['box']['size_m']=[.4,.2,.2];invalid.append(scene)
        scene=transfer_scene();scene['box']['mass_kg']=float('nan');invalid.append(scene)
        scene=transfer_scene();scene['box']['mass_kg']=True;invalid.append(scene)
        scene=transfer_scene();scene['tables']['destination']=copy.deepcopy(scene['tables']['source']);invalid.append(scene)
        scene=transfer_scene();scene['box']['center_xy']=[.45,.18];invalid.append(scene)
        scene=transfer_scene();scene['tables']['destination']['top_z']=-.9;invalid.append(scene)
        scene=transfer_scene();scene['box']['unrecognized_size']=[.1,.1,.1];invalid.append(scene)
        for scene in invalid:
            with self.subTest(scene=scene),self.assertRaises(ValueError):
                table_layout('box_transfer_v1',scene=scene)

    def test_rotated_footprint_and_reset_range_must_fit_source(self):
        scene=transfer_scene();scene['box']['yaw_offset_rad']=math.pi/4
        with self.assertRaises(ValueError):table_layout('box_transfer_v1',scene=scene)
        scene['box']['center_xy'][0]=.63
        self.assertIn('source',table_layout('box_transfer_v1',scene=scene))

    def test_original_bin_fixture_preserves_geometry(self):
        self.assertEqual(table_layout('native_bin'),TABLES)
    def test_empty_fixture_has_equal_heights_without_changing_source(self):
        layout=table_layout('two_empty_equal_height_tables_v1',(.02,-.03))
        self.assertEqual(layout['source'],TABLES['source'])
        self.assertEqual(layout['destination']['top_z'],layout['source']['top_z'])
        self.assertEqual(layout['destination']['size_xy'],TABLES['destination']['size_xy'])
        self.assertAlmostEqual(layout['destination']['center_xy'][0],-.225)
        self.assertAlmostEqual(layout['destination']['center_xy'][1],-1.6572)
        self.assertEqual(TABLES['destination']['top_z'],-.265)
    def test_unsupported_fixture_or_offsets_rejected_before_build(self):
        for name,offset in [('unknown',(0,0)),('native_bin',(.01,0)),
                            ('two_empty_equal_height_tables_v1',(.051,0)),
                            ('two_empty_equal_height_tables_v1',(float('nan'),0)),
                            ('two_empty_equal_height_tables_v1',(True,0))]:
            with self.subTest(name=name,offset=offset),self.assertRaises(ValueError):table_layout(name,offset)

if __name__=='__main__':unittest.main()
