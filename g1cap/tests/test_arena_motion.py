import copy,math,unittest
from g1cap.arena_motion import LoadedTranslation,LoadedTurn
from test_arena_retreat import state
ACTION=[0.]*46+[.75,0.,0.,0.]

def turn_state(step,angle=0):
    row=state(step);row['root_quat']=[math.cos(angle/2),0,0,math.sin(angle/2)]
    row['turn_clearance_m']=.3
    return row

class MotionTests(unittest.TestCase):
    def test_forward_distance_is_along_entry_heading(self):
        anchor=turn_state(50,-math.pi/2)
        controller=LoadedTranslation(anchor,ACTION,.3)
        row=copy.deepcopy(anchor);row['root_pos'][1]-=.25;row['box_pos'][1]-=.24
        m=controller.measurements(row)
        self.assertAlmostEqual(m['root_forward_m'],.25);self.assertAlmostEqual(m['box_forward_m'],.24)
        self.assertAlmostEqual(m['lateral_error_m'],0.)
        command=controller.command(anchor)
        self.assertGreater(command[43],0);self.assertAlmostEqual(command[44],0.)
        self.assertEqual(command[:43],ACTION[:43]);self.assertEqual(command[46:],ACTION[46:])

    def test_forward_success_requires_object_progress_and_stable_stop(self):
        controller=LoadedTranslation(state(50),ACTION,.3)
        row=state(51);row['root_pos'][0]+=.3
        controller.update(row)
        result=None
        for step in range(52,110):
            row['time']=step*.02;result=controller.update(row)
            if result:break
        self.assertEqual(result,('failed','final_displacement_outside_goal'))

    def test_turn_owns_three_second_zero_navigation_handoff(self):
        c=LoadedTurn(turn_state(50),ACTION,-math.pi/2)
        row=turn_state(51,-math.pi/2)
        c.update(row)
        for step in range(52,105):c.update(turn_state(step,-math.pi/2))
        self.assertEqual(c.phase,'turn_hold')
        command=c.command(turn_state(105,-math.pi/2))
        self.assertEqual(command[43:46],[0.,0.,0.])
        self.assertEqual(command[:43],ACTION[:43]);self.assertEqual(command[46:],ACTION[46:])
        result=None
        for step in range(106,270):
            result=c.update(turn_state(step,-math.pi/2))
            if result:break
        self.assertEqual(result,('completed','turn_and_zero_navigation_hold'))

    def test_turn_region_and_post_turn_contact_faults_still_fail(self):
        c=LoadedTurn(turn_state(50),ACTION,-math.pi/2)
        for step in range(51,105):c.update(turn_state(step,-math.pi/2))
        row=turn_state(105,-math.pi/2);row['loaded_contacts']['minimum_hand_N']=0.
        self.assertEqual(c.update(row),('failed','substep_hand_contact_lost'))
        c=LoadedTurn(turn_state(50),ACTION,-math.pi/2)
        row=turn_state(51);row['root_pos'][0]+=.151
        self.assertEqual(c.update(row),('failed','turn_translation_limit'))

if __name__=='__main__':unittest.main()

import test_arena_retreat

class MotionDispatchTests(unittest.TestCase):
    def setup(self):
        c=test_arena_retreat.RetreatTests().setup_control()
        row=state(50);row['turn_clearance_m']=.3
        row['surfaces']={name:dict(lower_body_clearance_m=.1,robot_contact_peak_N=0.) for name in ('source','destination')}
        return c,row

    def test_new_methods_use_normal_controller_dispatch_and_existing_grasp(self):
        for method,args,phase in [('move_with_box',{'distance_m':.3},'loaded_forward'),
                                  ('turn_with_box',{'yaw_rad':-.5},'loaded_turn')]:
            c,row=self.setup();c.grasp_prepared=True
            result=c.start(method,args,row)
            self.assertEqual(result['status'],'running',result);self.assertEqual(c.phase,phase)
            self.assertIsNone(c.preparation)
            sent=c.command(row);self.assertEqual(sent[:43],ACTION[:43])
            c.cancel('test_stop',row);self.assertEqual(c.command(row)[43:46],[0.,0.,0.])

    def test_missing_geometry_or_insufficient_turn_region_rejected(self):
        c,row=self.setup();del row['surfaces']
        self.assertEqual(c.start('move_with_box',{'distance_m':.3},row)['reason'],'missing_support_geometry')
        c,row=self.setup();row['turn_clearance_m']=.17
        self.assertEqual(c.start('turn_with_box',{'yaw_rad':-.5},row)['reason'],'insufficient_turn_region')
        self.assertIsNone(c.preparation)

    def test_forward_collision_margin_remains_active_after_admission(self):
        c,row=self.setup();c.grasp_prepared=True;c.start('move_with_box',{'distance_m':.3},row)
        row=copy.deepcopy(row);row['time']=1.02;row['surfaces']['destination']['lower_body_clearance_m']=.02
        result=c.update(row)
        self.assertEqual(result['reason'],'insufficient_support_clearance')
        self.assertEqual(c.command(row)[43:46],[0.,0.,0.])
