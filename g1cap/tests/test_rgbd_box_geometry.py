import unittest
try:
    import numpy as np
except ImportError:
    np=None

@unittest.skipIf(np is None,'NumPy unavailable')
class VisibleSupportTests(unittest.TestCase):
    def test_gap_uses_observed_plane_and_gravity_not_tilted_box_up(self):
        from g1cap.rgbd_box_geometry import observed_support_gap
        a=np.radians(30);rx=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
        axes=np.diag([1.,-1.,-1.])@rx
        box=dict(status='accepted',center_camera_m=[0,0,.7],axes_camera=axes.tolist(),dimensions_m=[.2,.2,.2])
        result=observed_support_gap(np.ones((160,160),bool),np.ones((160,160)),
                 [[160,0,80],[0,160,80],[0,0,1]],box,[0,0,-1])
        self.assertEqual(result['status'],'observed_candidate')
        expected=.3-.1*(np.sin(a)+np.cos(a))
        self.assertAlmostEqual(result['gap_m'],expected,places=6)

    def test_no_plane_or_missing_box_is_unavailable(self):
        from g1cap.rgbd_box_geometry import observed_support_gap
        box=dict(status='accepted',center_camera_m=[0,0,.7],axes_camera=np.diag([1.,-1.,-1.]).tolist(),dimensions_m=[.2,.2,.2])
        mask=np.ones((40,40),bool);depth=np.full((40,40),float('nan'));k=[[100,0,20],[0,100,20],[0,0,1]]
        self.assertEqual(observed_support_gap(mask,depth,k,box,[0,0,-1])['status'],'unavailable')
        box['status']='unavailable'
        self.assertEqual(observed_support_gap(mask,depth,k,box,[0,0,-1])['reason'],'box_pose_unavailable')

    def test_initialization_never_fills_unobserved_shape(self):
        from g1cap.rgbd_box_geometry import initialize_box
        mask=np.zeros((40,40),bool);depth=np.ones((40,40));k=[[100,0,20],[0,100,20],[0,0,1]]
        result=initialize_box(mask,~mask,depth,k)
        self.assertEqual(result['status'],'unknown');self.assertIsNone(result['dimensions_m'])

if __name__=='__main__':unittest.main()
