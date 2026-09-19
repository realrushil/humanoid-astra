import unittest
try:
    import numpy as np
except ImportError:
    np=None


@unittest.skipIf(np is None,'NumPy unavailable')
class BoxTrackingTests(unittest.TestCase):
    def test_silhouette_depth_along_the_same_camera_ray_does_not_shift_the_pose(self):
        _,refine=self.api()
        x,y=np.meshgrid(np.linspace(-.08,.08,20),np.linspace(-.12,.12,20))
        faces=np.stack([x.ravel(),y.ravel(),np.full(x.size,.6)],axis=1)
        a=np.linspace(-1,1,60)
        edges=np.concatenate([np.stack([np.full(60,s*.1),a*.15,np.full(60,.6)],axis=1) for s in (-1,1)]+
                             [np.stack([a*.1,np.full(60,s*.15),np.full(60,.6)],axis=1) for s in (-1,1)])
        c0,r0,q0=refine(faces,edges,[0,0,.7],np.eye(3),[.2,.3,.2])
        c1,r1,q1=refine(faces,edges*.96,[0,0,.7],np.eye(3),[.2,.3,.2])
        np.testing.assert_allclose(c1,c0,atol=1e-8)
        np.testing.assert_allclose(r1,r0,atol=1e-8)
        self.assertEqual(q1['constraint_rank'],6)
        self.assertLess(q1['rms_m'],1e-6)

    def api(self):
        from g1cap.rgbd_box_tracking import box_image_points,refine_box_pose
        return box_image_points,refine_box_pose

    def test_flat_interior_reports_unobservable_directions(self):
        _,refine=self.api()
        x,y=np.meshgrid(np.linspace(-.04,.04,20),np.linspace(-.04,.04,20))
        p=np.stack([x.ravel(),y.ravel(),np.full(x.size,.6)],axis=1)
        center,rotation,info=refine(p,np.empty((0,3)),[0,0,.702],np.eye(3),[.2,.2,.2])
        self.assertEqual(info['constraint_rank'],3)
        self.assertEqual(info['status'],'partially_constrained')
        self.assertAlmostEqual(center[2],.7,places=5)

    def test_recovers_non_cubic_pose_from_three_observed_faces(self):
        _,refine=self.api();rng=np.random.default_rng(11)
        size=np.array([.16,.3,.12]);half=size/2
        def yaw(a):
            return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
        expected_r=yaw(.4);expected_c=np.array([.05,-.03,.8])
        faces=[]
        for axis in range(3):
            p=rng.uniform(-half+.01,half-.01,(250,3));p[:,axis]=-half[axis]
            faces.append(p)
        points=np.concatenate(faces)@expected_r.T+expected_c
        c,r,info=refine(points,np.empty((0,3)),expected_c+[.004,-.003,.005],
                        expected_r@yaw(.015),size)
        np.testing.assert_allclose(c,expected_c,atol=1e-5)
        np.testing.assert_allclose(r,expected_r,atol=1e-5)
        self.assertEqual(info['constraint_rank'],6)

    def test_foreground_occluder_is_not_object_outline(self):
        extract,_=self.api()
        mask=np.zeros((40,40),bool);mask[10:30,10:30]=True
        depth=np.ones((40,40));depth[mask]=.5
        k=[[50,0,20],[0,50,20],[0,0,1]]
        _,visible=extract(mask,depth,k)
        depth[~mask]=.3
        _,occluded=extract(mask,depth,k)
        self.assertGreater(len(visible),0)
        self.assertEqual(len(occluded),0)

    def test_lost_depth_and_distant_guess_do_not_invent_pose(self):
        _,refine=self.api()
        empty=np.empty((0,3));center=[0,0,.7]
        c,r,info=refine(empty,empty,center,np.eye(3),[.2,.2,.2])
        self.assertEqual(info['status'],'lost');np.testing.assert_equal(c,center)
        c,r,info=refine(np.ones((200,3))*5,empty,center,np.eye(3),[.2,.2,.2])
        self.assertEqual(info['status'],'lost');np.testing.assert_equal(c,center)

    def test_pose_rejects_reflection_and_invalid_dimensions(self):
        _,refine=self.api();empty=np.empty((0,3))
        for r,size in [(np.diag([-1,1,1]),[.2,.2,.2]),(np.eye(3),[.2,0,.2]),(np.eye(3),[.2,float('nan'),.2])]:
            with self.assertRaises(ValueError):refine(empty,empty,[0,0,.7],r,size)

    def test_mask_alignment_and_invalid_depth(self):
        extract,_=self.api();k=[[50,0,20],[0,50,20],[0,0,1]]
        with self.assertRaises(ValueError):extract(np.ones((5,5),bool),np.ones((6,5)),k)
        p,e=extract(np.ones((5,5),bool),np.full((5,5),float('nan')),k)
        self.assertEqual(len(p),0);self.assertEqual(len(e),0)

if __name__=='__main__':unittest.main()
