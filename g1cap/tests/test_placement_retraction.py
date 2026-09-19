import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from g1cap.arena_retraction import PlacementRetraction

ROOT=Path(__file__).resolve().parents[1]


class RetractionControlTests(unittest.TestCase):
    def test_source_selection_uses_the_same_geometry_and_verification(self):
        c,row,saved=self.make()
        selected=copy.deepcopy(row)
        selected['surfaces']['source'],selected['surfaces']['destination']=selected['surfaces']['destination'],selected['surfaces']['source']
        source=PlacementRetraction(selected,saved,'source',lambda *a:c.motion,c.geometry,c.supports)
        self.assertEqual(source.plan,c.plan)
        moved=self.shifted(selected,source.plan['translation_world'],selected['time']+4.)
        self.assertIsNone(source.update(moved));self.assertEqual(source.phase,'place_retract_hold')

    def test_planning_clearance_has_reserve_above_final_requirement(self):
        c,row,saved=self.make()
        self.assertGreaterEqual(c.plan['predicted_arm_clearance_m'],.008)

    def test_recorded_one_mm_short_state_does_not_enter_hold(self):
        c,row,saved=self.make()
        boundary=json.loads((ROOT/'tests/fixtures/placement-clearance-boundary.json').read_text())
        self.assertIsNone(c.update(boundary))
        self.assertEqual(c.phase,'place_retract')

    def make(self):
        d=json.loads((ROOT/'tests/fixtures/placement-clearance.json').read_text())
        row=d['aligned'];saved=[.1]*43+[0.,0.,0.,.75,0.,0.,0.]
        def forbidden(*args):raise AssertionError('IK must stop during hold or after failure')
        motion=SimpleNamespace(command=forbidden,targets={n:SimpleNamespace(translation=p['pos']) for n,p in row['wrists_world'].items()})
        world=SimpleNamespace(raw=row,geometry=d['geometry'],support_parts=d['supports'],
            control=SimpleNamespace(last_action=saved),wrist_motion=lambda *a:motion)
        return PlacementRetraction(row,saved,'destination',world.wrist_motion,d['geometry'],d['supports']),row,saved

    def shifted(self,row,delta,time):
        r=copy.deepcopy(row);r['time']=time
        r['box_pos']=[v+d for v,d in zip(r['box_pos'],delta)]
        for k in ('min','max'):r['box_bounds'][k]=[v+d for v,d in zip(r['box_bounds'][k],delta)]
        for name,pose in r['body_poses'].items():
            if any(w in name for w in ('shoulder','elbow','wrist','hand')):pose['pos']=[v+d for v,d in zip(pose['pos'],delta)]
        r['wrists_world']={n:r['body_poses'][n] for n in r['wrists_world']}
        return r

    def test_verified_geometry_enters_hold_and_keeps_loaded_references(self):
        c,row,saved=self.make();self.assertEqual(c.phase,'place_retract')
        r=self.shifted(row,c.plan['translation_world'],row['time']+4.)
        self.assertIsNone(c.update(r));self.assertEqual(c.phase,'place_retract_hold')
        self.assertEqual(c.command(r),saved)
        for i in range(1,100):
            r['time']+=.02;self.assertIsNone(c.update(r))
        r['time']+=.02
        self.assertEqual(c.update(r),('completed','placement_clearance_and_hold'))

    def test_force_limit_stops_and_latches_before_any_placement(self):
        c,row,saved=self.make();self.assertEqual(c.phase,'place_retract')
        row['loaded_contacts']['maximum_hand_N']=25.1
        self.assertEqual(c.update(row),('failed','preparation_force_limit'))
        row['loaded_contacts']['maximum_hand_N']=10.
        self.assertEqual(c.update(row),('failed','preparation_force_limit'))
        self.assertEqual(c.command(row),saved)

    def test_eight_cm_request_has_time_to_track_before_bounded_timeout(self):
        c,row,saved=self.make()
        # A valid retained partial move after 8.5 s must still have the
        # placement-only 2 s settling allowance beyond its 8 s target ramp.
        axis=[v/c.plan['distance_m'] for v in c.plan['translation_world']]
        c.plan=dict(c.plan,distance_m=.08,translation_world=[v*.08 for v in axis])
        moved=self.shifted(row,[v*.068 for v in axis],row['time']+8.5)
        self.assertIsNone(c.update(moved))
        moved['time']=row['time']+10.
        self.assertEqual(c.update(moved),('failed','retraction_timeout'))


if __name__=='__main__':unittest.main()
