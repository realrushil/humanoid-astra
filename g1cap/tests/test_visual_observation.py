"""Snapshot identity and immutable image delivery, without simulator dependencies."""
from copy import deepcopy
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
import zlib


def snapshot_fixture(root, step=0):
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
    rgb=b'\xff\x00\x00'
    png=(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',1,1,8,2,0,0,0))+
         chunk(b'IDAT',zlib.compress(b'\x00'+rgb))+chunk(b'IEND',b''))
    frames=[]
    for camera in ('head','overview'):
        path=root/f'{step}-{camera}.png';path.write_bytes(png)
        frames.append(dict(camera=camera,file=path.name,width=1,height=1,
            encoded_sha256=hashlib.sha256(png).hexdigest(),rgb_sha256=hashlib.sha256(rgb).hexdigest(),source='live_camera'))
    return dict(version=1,session_id='session-a',snapshot_id=f'{step:032x}',step=step,physics_step=step*4,
                sim_time_s=step*.02,captured_at_unix_s=100.+step,
                observation={'session_id':'session-a','step':step,'physics_step':step*4,'time':step*.02},frames=frames)


class VisualObservationTests(unittest.TestCase):
    def module(self):
        from g1cap import visual_observation
        return visual_observation

    def test_delivery_preserves_order_and_bytes_and_rejects_changes(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);resources=root/'resources';resources.mkdir()
            before=snapshot_fixture(root);current=snapshot_fixture(root,10)
            staged=m.stage_visual_observations([before,current],root,resources,'session-a')
            self.assertEqual([p['step'] for p in staged],[0,10])
            for packet in staged:
                self.assertEqual([f['camera'] for f in packet['frames']],['head','overview'])
                self.assertEqual((resources/packet['frames'][0]['file']).read_bytes(),(root/'0-head.png').read_bytes())
            (root/'10-head.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):m.validate_snapshot(current,root,'session-a')

    def test_rejects_cross_session_mixed_state_and_invalid_frames(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);packet=snapshot_fixture(root)
            broken=[]
            for key,value in (('session_id','different'),('sim_time_s',float('nan')),('step',True)):
                p=deepcopy(packet);p[key]=value;broken.append(p)
            p=deepcopy(packet);p['observation']['step']=1;broken.append(p)
            p=deepcopy(packet);p['frames'][0]['width']=2;broken.append(p)
            p=deepcopy(packet);p['frames'].reverse();broken.append(p)
            p=deepcopy(packet);p['frames'][0]['file']='../outside.png';broken.append(p)
            (root/'link.png').symlink_to(root/'0-head.png')
            p=deepcopy(packet);p['frames'][0]['file']='link.png';broken.append(p)
            for p in broken:
                with self.subTest(packet=p),self.assertRaises(ValueError):m.validate_snapshot(p,root,'session-a')

    def test_rejects_reversed_history_and_more_than_four_images(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);resources=root/'resources';resources.mkdir()
            packets=[snapshot_fixture(root,i) for i in (0,10,20)]
            for history in ([],packets,list(reversed(packets[:2]))):
                with self.assertRaises(ValueError):m.stage_visual_observations(history,root,resources,'session-a')
