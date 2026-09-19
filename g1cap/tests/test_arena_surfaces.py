import math
import unittest
from g1cap.arena_surfaces import box_bounds, contained, separation, support_observation

class SurfaceTests(unittest.TestCase):
    def test_rectangular_box_uses_recorded_full_dimensions(self):
        row=dict(box_quat=[1,0,0,0],box_pos=[0,0,.08],box_size_m=[.20,.24,.16])
        bounds=box_bounds(row)
        self.assertEqual(bounds,dict(min=[-.10,-.12,0.],max=[.10,.12,.16]))
        row['box_quat']=[math.sqrt(.5),0,0,math.sqrt(.5)]
        self.assertAlmostEqual(box_bounds(row)['max'][0],.12)
        self.assertAlmostEqual(box_bounds(row)['max'][1],.10)

    def test_support_scales_with_object_weight(self):
        box=dict(min=[-.1,-.1,0],max=[.1,.1,.2])
        surface=dict(min=[-.5,-.5,-.04],max=[.5,.5,0])
        self.assertFalse(support_observation(box,surface,1.,0.,.07,mass_kg=.5)['supported'])
        self.assertTrue(support_observation(box,surface,4.905,0.,.07,mass_kg=.5)['supported'])

    def test_invalid_dimensions_are_not_silently_treated_as_the_old_cube(self):
        for size in ([.2,0,.2],[.2,float('nan'),.2],[.2,.2],[True,.2,.2]):
            with self.subTest(size=size),self.assertRaises(ValueError):
                box_bounds(dict(box_quat=[1,0,0,0],box_pos=[0,0,0],box_size_m=size))

    def test_rotated_box_requires_its_full_footprint(self):
        row=dict(box_quat=[math.cos(math.pi/8),0,0,math.sin(math.pi/8)],box_pos=[0,0,.1])
        bounds=box_bounds(row)
        self.assertAlmostEqual(bounds['max'][0],math.sqrt(.02))
        self.assertFalse(contained(bounds,dict(min=[-.11,-.11,-.1],max=[.11,.11,0])))

    def test_separation_uses_height_as_well_as_planar_distance(self):
        self.assertAlmostEqual(separation(dict(min=[0,0,.02],max=[.1,.1,.1]),
                                         dict(min=[-.1,-.1,-.1],max=[.1,.1,0])),.02)

    def test_selected_support_requires_its_force_and_its_height(self):
        box=dict(min=[-.1,-.1,0],max=[.1,.1,.2])
        surface=dict(min=[-.5,-.5,-.04],max=[.5,.5,0])
        observation=support_observation(box,surface,1.,0.,.07)
        self.assertTrue(observation['supported']);self.assertTrue(observation['contained'])
        self.assertFalse(support_observation(box,surface,0.,0.,.07)['supported'])
        elevated=dict(min=[-.1,-.1,.02],max=[.1,.1,.22])
        self.assertFalse(support_observation(elevated,surface,1.,0.,.07)['supported'])
        self.assertEqual(observation['bounds'],surface)

if __name__=='__main__':unittest.main()
