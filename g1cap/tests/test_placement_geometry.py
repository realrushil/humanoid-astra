import copy
import json
from pathlib import Path
import unittest
from g1cap.arena_placement_geometry import placement_retraction

ROOT=Path(__file__).resolve().parents[1]


class RetractionGeometryTests(unittest.TestCase):
    def sample(self):
        p=ROOT/'tests/fixtures/placement-clearance.json'
        d=json.loads(p.read_text())
        return d['aligned'],d['geometry'],d['supports']

    def test_recorded_obstruction_has_bounded_geometry_candidate(self):
        r,g,s=self.sample();plan=placement_retraction(r,g,s)
        self.assertIsNotNone(plan)
        self.assertGreater(plan['distance_m'],.025)
        self.assertLess(plan['distance_m'],.04)
        self.assertGreaterEqual(plan['predicted_arm_clearance_m'],.005)
        self.assertGreaterEqual(plan['predicted_box_margin_m'],.01)
        self.assertEqual(plan['translation_world'][2],0.)

    def test_world_origin_does_not_change_relative_retraction(self):
        r,g,s=self.sample();plan=placement_retraction(r,g,s)
        self.assertIsNotNone(plan)
        moved=copy.deepcopy(r);parts=copy.deepcopy(s);delta=[3.,-2.,1.]
        for key in ('root_pos','box_pos'):moved[key]=[v+d for v,d in zip(moved[key],delta)]
        for pose in moved['body_poses'].values():pose['pos']=[v+d for v,d in zip(pose['pos'],delta)]
        for b in [moved['box_bounds'],*[v['bounds'] for v in moved['surfaces'].values()],*[b for ps in parts.values() for b in ps]]:
            for key in ('min','max'):b[key]=[v+d for v,d in zip(b[key],delta)]
        result=placement_retraction(moved,g,parts)
        self.assertIsNotNone(result)
        for a,b in zip(plan['translation_world'],result['translation_world']):self.assertAlmostEqual(a,b,places=8)

    def test_refuses_retraction_that_would_move_box_off_table(self):
        r,g,s=self.sample()
        # Remove the available margin on the robot-facing edge without changing
        # the grasp; a planner must not return the old successful displacement.
        r['surfaces']['destination']['bounds']['max'][1]=r['box_bounds']['max'][1]+.01
        s['destination'][0]['max'][1]=r['surfaces']['destination']['bounds']['max'][1]
        self.assertIsNone(placement_retraction(r,g,s))

    def workspace_fixture(self):
        # The lowered arm clears the table after about 7.5 cm of retraction.
        # Root, box and table positions are arbitrary; no benchmark route.
        table=dict(min=[0.,-.5,-.1],max=[1.,.5,0.])
        shape=dict(min=[-.01,-.01,-.01],max=[.01,.01,.01])
        obs=dict(root_pos=[-.5,0.,.6],box_pos=[.2,0.,.2],
            box_bounds=dict(min=[.1,-.1,.1],max=[.3,.1,.3]),
            surfaces={'destination':dict(bounds=table,clearance_m=.1)},
            body_poses={'right_hand_palm_link':dict(pos=[.057,0.,.102],xyzw=[0.,0.,0.,1.])})
        return obs,{'robot':{'right_hand_palm_link':[shape]}},{'destination':[table]}

    def test_placement_workspace_can_clear_a_grasp_beyond_six_cm(self):
        result=placement_retraction(*self.workspace_fixture(),preferred_arm_margin_m=.008)
        self.assertIsNotNone(result)
        self.assertGreater(result['distance_m'],.06)
        self.assertLessEqual(result['distance_m'],.08)
        self.assertGreaterEqual(result['predicted_arm_clearance_m'],.008)
        self.assertGreaterEqual(result['predicted_box_margin_m'],.01)

    def test_workspace_extension_preserves_box_edge_margin(self):
        obs,g,s=self.workspace_fixture();obs['box_bounds']['min'][0]=.071
        self.assertIsNone(placement_retraction(obs,g,s))

    def test_workspace_extension_still_rejects_unreachable_clearance(self):
        obs,g,s=self.workspace_fixture();obs['body_poses']['right_hand_palm_link']['pos'][0]=.08
        self.assertIsNone(placement_retraction(obs,g,s))


if __name__=='__main__':unittest.main()
