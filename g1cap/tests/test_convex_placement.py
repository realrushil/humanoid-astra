"""Native distance refinement must avoid false bound collisions, not weaken margins."""
import importlib.util
import math
import unittest

from g1cap.arena_placement_geometry import placement_retraction


def fixture():
    # A sloped solid clears a table edge; its enclosing box intersects it.
    vertices=[[-.02,-.01,-.01],[-.02,.01,-.01],[-.02,0.,.03],[.02,0.,.03]]
    shape=dict(min=[-.02,-.01,-.01],max=[.02,.01,.03],convex_vertices=vertices)
    geometry={'robot':{'right_hand_palm_link':[shape]}}
    table=dict(min=[0.,-.5,-.1],max=[1.,.5,0.])
    obs=dict(root_pos=[-.5,0.,.6],box_pos=[.2,0.,.2],
        box_bounds=dict(min=[.1,-.1,.1],max=[.3,.1,.3]),
        surfaces={'destination':dict(bounds=table,clearance_m=.1)},
        body_poses={'right_hand_palm_link':dict(pos=[-.01,0.,.102],xyzw=[0.,0.,0.,1.])})
    return obs,geometry,{'destination':[table]}


@unittest.skipUnless(importlib.util.find_spec('coal'), 'native Coal dependency unavailable')
class ConvexPlacementTests(unittest.TestCase):
    def plan(self, obs, geometry, supports):
        from g1cap.toolkit.convex_clearance import ConvexClearance
        return placement_retraction(obs,geometry,supports,
            arm_clearance=ConvexClearance(geometry,supports),preferred_arm_margin_m=.008)

    def test_sloped_grasp_fits_despite_its_enclosing_bound(self):
        args=fixture()
        self.assertGreater(placement_retraction(*args)['distance_m'],0.)
        result=self.plan(*args)
        self.assertEqual(result['distance_m'],0.)
        self.assertGreater(result['predicted_arm_clearance_m'],.008)

    def test_world_origin_does_not_change_clearance(self):
        obs,g,s=fixture();expected=self.plan(obs,g,s);delta=[17.,-21.,9.]
        for bounds in (obs['box_bounds'],s['destination'][0]):
            for side in ('min','max'):bounds[side]=[a+b for a,b in zip(bounds[side],delta)]
        for key in ('root_pos','box_pos'):obs[key]=[a+b for a,b in zip(obs[key],delta)]
        for pose in obs['body_poses'].values():pose['pos']=[a+b for a,b in zip(pose['pos'],delta)]
        actual=self.plan(obs,g,s)
        for key in ('distance_m','predicted_arm_clearance_m','predicted_box_margin_m'):
            self.assertAlmostEqual(actual[key],expected[key],places=10)

    def test_quarter_turn_preserves_asymmetric_hull_and_primitive_clearance(self):
        for mesh in (True,False):
            with self.subTest(mesh=mesh):
                obs,g,s=fixture()
                if not mesh:del g['robot']['right_hand_palm_link'][0]['convex_vertices']
                expected=self.plan(obs,g,s)
                # Rotate the whole scene +90 degrees about world Z. Table and
                # object remain axis-aligned, so their new bounds are exact.
                obs['root_pos']=[0.,-.5,.6];obs['box_pos']=[0.,.2,.2]
                obs['box_bounds']=dict(min=[-.1,.1,.1],max=[.1,.3,.3])
                s['destination'][0].update(min=[-.5,0.,-.1],max=[.5,1.,0.])
                obs['body_poses']['right_hand_palm_link']=dict(pos=[0.,-.01,.102],
                    xyzw=[0.,0.,math.sqrt(.5),math.sqrt(.5)])
                actual=self.plan(obs,g,s)
                for key in ('distance_m','predicted_arm_clearance_m','predicted_box_margin_m'):
                    self.assertAlmostEqual(actual[key],expected[key],places=10)

    def test_outside_source_bounds_is_rejected(self):
        args=fixture();args[1]['robot']['right_hand_palm_link'][0]['convex_vertices'][0][0]=-.05
        with self.assertRaisesRegex(ValueError,'source_hull_leaves'):
            self.plan(*args)

    def test_true_obstruction_and_containment_still_reject(self):
        obs,g,s=fixture();obs['body_poses']['right_hand_palm_link']['pos'][0]=.2
        obs['box_bounds']['min'][0]=.011
        self.assertIsNone(self.plan(obs,g,s))

    def test_missing_mesh_uses_conservative_oriented_bounds(self):
        args=fixture();del args[1]['robot']['right_hand_palm_link'][0]['convex_vertices']
        self.assertGreater(self.plan(*args)['distance_m'],0.)

    @unittest.skipUnless(importlib.util.find_spec('pxr'), 'native USD dependency unavailable')
    def test_source_mesh_transform_is_link_local_not_world(self):
        from pxr import Usd,UsdGeom,Gf
        from g1cap.toolkit.arena_geometry import source_hull_vertices
        stage=Usd.Stage.CreateInMemory()
        body=UsdGeom.Xform.Define(stage,'/body')
        body.AddTranslateOp().Set(Gf.Vec3d(17.,-21.,9.))
        mesh=UsdGeom.Mesh.Define(stage,'/body/collision')
        mesh.CreatePointsAttr([(0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)])
        mesh.AddTranslateOp().Set(Gf.Vec3d(.1,.2,.3))
        mesh.AddRotateZOp().Set(90.)
        bounds=dict(min=[-.9,.2,.3],max=[.1,1.2,1.3])
        points=source_hull_vertices(mesh.GetPrim(),body.GetPrim(),bounds,UsdGeom.XformCache())
        actual=sorted(tuple(round(v,6) for v in point) for point in points)
        self.assertEqual(actual,[(-.9,.2,.3),(.1,.2,.3),(.1,.2,1.3),(.1,1.2,.3)])


if __name__=='__main__':unittest.main()
