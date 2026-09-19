"""Two static tables on a floor, with the native movable box and destination bin.

World coordinates are metres, Z up. Arena places G1's initial root at z=0;
its floor is z=-0.795. These heights preserve the native working envelope.
The source surface leaves a 0.7 mm initial gap under the 20 cm box.
"""
from copy import deepcopy


def sensor_terminations(cfg):
    """Retain time limits, never end a sensor episode using true object state.

    A future native termination must be reviewed before entering this track.
    Exact drop/contact/success criteria remain in our independent scorer.
    """
    for name,term in vars(cfg).items():
        if not name.startswith('_') and term is not None and name not in ('time_out','success','object_dropped'):
            raise ValueError('unreviewed native termination: '+name)
    cfg.success=None
    cfg.object_dropped=None
import math

FLOOR_Z = -0.795
TABLES = {
    'source': {'center_xy': (0.75, 0.18), 'size_xy': (0.60, 0.80), 'top_z': -0.030},
    'destination': {'center_xy': (-0.245, -1.6272), 'size_xy': (1.10, 0.80), 'top_z': -0.265},
}

def author_box_mass(root,mass_kg):
    """Construction only: native box has a nested rigid body and no MassAPI.

    Author on that single physical body, not the asset's transform container.
    Never call after physics starts. The worker has no scene/asset access.
    """
    from pxr import Usd,UsdPhysics
    bodies=[p for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies)!=1:raise ValueError('transfer box must have exactly one rigid body')
    if isinstance(mass_kg,bool) or not isinstance(mass_kg,(int,float)) or not math.isfinite(mass_kg) or mass_kg<=0:
        raise ValueError('positive finite box mass required')
    mass=UsdPhysics.MassAPI.Apply(bodies[0])
    mass.CreateMassAttr(float(mass_kg));mass.CreateDensityAttr(0.)


def spawn_transfer_box(prim_path,cfg,translation=None,orientation=None,**kwargs):
    """Native textured/collidable USD, with mass on its actual rigid body."""
    import isaaclab.sim as sim
    native_cfg=deepcopy(cfg);native_cfg.mass_props=None
    root=sim.spawn_from_usd(prim_path,native_cfg,translation,orientation,**kwargs)
    author_box_mass(root,cfg.mass_props.mass)
    return root

def validate_transfer_scene(scene):
    """Validate construction assumptions, not grasp/reach/controller capability.

    Two axis-aligned desks and one cuboid. Full sizes and positions are metres;
    box yaw is an offset from the native asset's initial orientation. The box's
    entire randomized XY footprint must fit its source before physics starts.
    """
    def fields(value,expected):
        if not isinstance(value,dict) or set(value)!=set(expected):
            raise ValueError('scene fields must be '+', '.join(expected))
    def number(value,low,high):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError(f'scene value must be finite in [{low}, {high}]')
    def vector(value,count,low,high):
        if not isinstance(value,(list,tuple)) or len(value)!=count:raise ValueError('wrong scene vector length')
        for v in value:number(v,low,high)
    fields(scene,('tables','box'));fields(scene['tables'],('source','destination'))
    for table in scene['tables'].values():
        fields(table,('center_xy','size_xy','top_z'))
        vector(table['center_xy'],2,-4.5,4.5);vector(table['size_xy'],2,.4,1.5)
        number(table['top_z'],FLOOR_Z+.60,FLOOR_Z+.90)
    box=scene['box'];fields(box,('size_m','mass_kg','center_xy','xy_jitter_m','yaw_offset_rad'))
    vector(box['size_m'],3,.10,.35);number(box['mass_kg'],.05,1.)
    vector(box['center_xy'],2,-4.5,4.5);number(box['xy_jitter_m'],0.,.025)
    number(box['yaw_offset_rad'],-math.pi,math.pi)
    source,destination=(scene['tables'][n] for n in ('source','destination'))
    if source['center_xy'][0]-source['size_xy'][0]/2<.30:
        raise ValueError('source must remain in front of the initial standing robot')
    c,s=abs(math.cos(box['yaw_offset_rad'])),abs(math.sin(box['yaw_offset_rad']))
    sx,sy,_=box['size_m'];half=[(c*sx+s*sy)/2,(s*sx+c*sy)/2]
    if any(abs(box['center_xy'][i]-source['center_xy'][i])+half[i]+box['xy_jitter_m']+.001>source['size_xy'][i]/2 for i in range(2)):
        raise ValueError('box reset footprint overhangs source')
    if not any(abs(source['center_xy'][i]-destination['center_xy'][i])>
               (source['size_xy'][i]+destination['size_xy'][i])/2+.10 for i in range(2)):
        raise ValueError('tables overlap or lack separation')
    root=(0.,.18)
    if math.hypot(*(max(0.,abs(root[i]-destination['center_xy'][i])-destination['size_xy'][i]/2) for i in range(2)))<.35:
        raise ValueError('destination intersects initial standing region')
    return deepcopy(scene)


def table_layout(fixture='native_bin',destination_offset_xy_m=(0.,0.),*,scene=None):
    """Static fixture choice; bounded5cm destination variation is an assumption."""
    if fixture=='box_transfer_v1':
        if any(destination_offset_xy_m):raise ValueError('explicit scene cannot use legacy offsets')
        return validate_transfer_scene(scene)['tables']
    if scene is not None:raise ValueError('explicit scene requires box_transfer_v1')
    if fixture not in ('native_bin','two_empty_equal_height_tables_v1'):
        raise ValueError('unsupported Arena fixture')
    if (not isinstance(destination_offset_xy_m,(list,tuple)) or len(destination_offset_xy_m)!=2
        or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>.05
               for v in destination_offset_xy_m)):
        raise ValueError('destination offset must be two finite metres within5cm')
    if fixture=='native_bin' and any(destination_offset_xy_m):
        raise ValueError('native bin fixture does not support table-only offsets')
    tables=deepcopy(TABLES)
    if fixture=='two_empty_equal_height_tables_v1':
        tables['destination']['top_z']=tables['source']['top_z']
        tables['destination']['center_xy']=tuple(a+b for a,b in zip(tables['destination']['center_xy'],destination_offset_xy_m))
    return tables


def configure_task(builder,fixture):
    """Select task/assets before composing the native manager configuration.

    Empty-table completion belongs to our independent task scorer; keep native
    timeout/drop termination. This does not alter objects during an episode.
    """
    if fixture not in ('native_bin','two_empty_equal_height_tables_v1','box_transfer_v1'):
        raise ValueError('unsupported Arena fixture')
    if fixture=='native_bin':return
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.task_base import TaskBase

    class TableTransferTask(TaskBase):
        def __init__(self,old):
            super().__init__(episode_length_s=120,task_description='Move the brown box onto the green destination table.')
            self.scene_cfg=deepcopy(old.get_scene_cfg())
            self.scene_cfg.contact_sensor_brown_box.filter_prim_paths_expr=['{ENV_REGEX_NS}/destination_top/geometry/mesh']
            self.termination_cfg=deepcopy(old.get_termination_cfg())
            self.termination_cfg.success=None
        def get_scene_cfg(self):return self.scene_cfg
        def get_termination_cfg(self):return self.termination_cfg
        def get_events_cfg(self):return None
        def get_mimic_env_cfg(self,arm_mode):raise NotImplementedError('Mimic generation is not part of this fixture')
        def get_metrics(self):return []

    environment=builder.arena_env
    environment.task=TableTransferTask(environment.task)
    environment.scene=Scene([asset for name,asset in environment.scene.assets.items() if name!='blue_sorting_bin'])


def simplify_scene(cfg,fixture='native_bin',destination_offset_xy_m=(0.,0.),*,scene=None):
    tables=table_layout(fixture,destination_offset_xy_m,scene=scene)
    from isaaclab.assets import AssetBaseCfg
    import isaaclab.sim as sim
    cfg.scene.galileo_locomanip = None
    cfg.scene.floor = AssetBaseCfg(prim_path='/World/SimpleFloor',
        spawn=sim.CuboidCfg(size=(20.,20.,.10),
            collision_props=sim.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.32,.35,.40),roughness=.95)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.,0.,FLOOR_Z-.05)))
    cfg.scene.light = AssetBaseCfg(prim_path='/World/SimpleLight',
        spawn=sim.DomeLightCfg(intensity=650.,color=(.95,.97,1.)))
    def part(name,size,position,color):
        setattr(cfg.scene,name,AssetBaseCfg(prim_path='{ENV_REGEX_NS}/'+name,
            spawn=sim.CuboidCfg(size=size,collision_props=sim.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim.PreviewSurfaceCfg(diffuse_color=color,roughness=.8)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=position)))
    for name,table in tables.items():
        x,y=table['center_xy'];sx,sy=table['size_xy'];top=table['top_z'];thickness=.04
        color=(.30,.52,.34) if name=='destination' and fixture!='native_bin' else (.48,.51,.55)
        part(name+'_top',(sx,sy,thickness),(x,y,top-thickness/2),color)
        height=top-thickness-FLOOR_Z
        assert height>0
        for i,(dx,dy) in enumerate([(-1,-1),(-1,1),(1,-1),(1,1)]):
            part(name+'_leg_'+str(i),(.04,.04,height),
                (x+dx*(sx/2-.04),y+dy*(sy/2-.04),FLOOR_Z+height/2),(.27,.30,.34))
    return {'floor_z_m':FLOOR_Z,'tables':tables,'fixture':fixture,'destination_offset_xy_m':list(destination_offset_xy_m),'removed':'Galileo lab background, clutter, walls, shelves and original tables',
        'retained':('native G1, brown box, blue bin, object reset locations, task scoring and WBC' if fixture=='native_bin'
                    else 'native G1, brown box and WBC; explicit empty tables; application task scoring')}
