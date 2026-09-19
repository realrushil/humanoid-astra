import copy,unittest
from g1cap.arena_task import ArenaTransferTask
from test_arena_retreat import state

BOUNDS={'min':[-.8,-2.,-.07],'max':[.3,-1.2,-.03]}
def row(step,placed=False):
    r=state(step);r['box_floor_contact_peak_N']=0.
    r['surfaces']={
        'source':dict(robot_contact_peak_N=0.,box_upward_N=0.,clearance_m=r['clearance'],bounds={'min':[.45,-.22,-.07],'max':[1.05,.58,-.03]}),
        'destination':dict(robot_contact_peak_N=0.,box_upward_N=.981 if placed else 0.,clearance_m=0. if placed else .1,bounds=BOUNDS)}
    if placed:
        r['box_pos']=[-.25,-1.6,.07];r['hand_forces_N']={'left':0.,'right':0.};r['bilateral']=False
    return r
class TransferTaskTests(unittest.TestCase):
    def test_heavy_box_needs_corresponding_support_not_a_cube_constant(self):
        score=ArenaTransferTask('destination',box_size_m=[.2,.2,.2],box_mass_kg=.5)
        for i in range(130):
            r=row(i,i>=51);r.update(box_size_m=[.2,.2,.2],box_mass_kg=.5)
            score.update(r)
        self.assertFalse(score.metrics()['success'])
        for i in range(130,190):
            r=row(i,True);r.update(box_size_m=[.2,.2,.2],box_mass_kg=.5)
            r['surfaces']['destination']['box_upward_N']=4.905
            score.update(r)
        self.assertTrue(score.metrics()['success'])

    def test_rectangular_object_overhang_cannot_pass_as_a_smaller_cube(self):
        score=ArenaTransferTask('destination',box_size_m=[.4,.2,.2],box_mass_kg=.1)
        for i in range(130):
            r=row(i,i>=51);r.update(box_size_m=[.4,.2,.2],box_mass_kg=.1)
            if i>=51:r['box_pos'][0]=.15  # +X edge .35, beyond table edge .30.
            score.update(r)
        self.assertFalse(score.metrics()['success'])

    def test_physical_properties_cannot_change_during_an_episode(self):
        score=ArenaTransferTask('destination',box_size_m=[.2,.2,.2],box_mass_kg=.5)
        r=row(0);r.update(box_size_m=[.2,.2,.2],box_mass_kg=.1)
        score.update(r)
        self.assertEqual(score.metrics()['failure'],'invalid_transfer_observation')

    def test_initial_or_wrong_support_cannot_satisfy_transfer(self):
        score=ArenaTransferTask('destination')
        for i in range(100):score.update(row(i,True))
        self.assertFalse(score.metrics()['success'])
        score=ArenaTransferTask('destination')
        for i in range(51):score.update(row(i))
        for i in range(51,130):
            r=row(i,True);r['surfaces']['destination']['box_upward_N']=0.;r['surfaces']['source']['box_upward_N']=.981
            score.update(r)
        self.assertFalse(score.metrics()['success'])
    def test_lift_then_contained_stable_release_passes_without_tool_status(self):
        score=ArenaTransferTask('destination')
        for i in range(51):score.update(row(i))
        # A conservative per-tool contact stop is not itself a dropped object.
        r=row(51);r['loaded_contacts']['minimum_hand_N']=0.;score.update(r)
        for i in range(52,130):score.update(row(i,True))
        self.assertTrue(score.metrics()['success']);self.assertGreater(score.metrics()['box_displacement_m'],1.)
        r=row(130,True);r['box_pos'][0]=.28;score.update(r)
        self.assertFalse(score.metrics()['success'])
    def test_floor_or_destination_collision_is_latched_even_after_release(self):
        for field in ('floor','destination'):
            score=ArenaTransferTask('destination')
            for i in range(51):score.update(row(i))
            r=row(51)
            if field=='floor':r['box_floor_contact_peak_N']=1.
            else:r['surfaces']['destination']['robot_contact_peak_N']=6.
            score.update(r)
            for i in range(52,130):score.update(row(i,True))
            self.assertFalse(score.metrics()['success']);self.assertIsNotNone(score.metrics()['failure'])
    def test_missing_or_reordered_evidence_fails_closed(self):
        score=ArenaTransferTask('destination');r=row(0);del r['box_floor_contact_peak_N'];score.update(r)
        self.assertEqual(score.metrics()['failure'],'invalid_transfer_observation')
        score=ArenaTransferTask('destination');score.update(row(0));score.update(row(2))
        self.assertEqual(score.metrics()['failure'],'missing_or_reordered_observation')
