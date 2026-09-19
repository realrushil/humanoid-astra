"""Scene definitions shared by simulation, planning and scene observations.

Boxes are axis-aligned, dimensions are HALF sizes in metres. Markers are empty
space wrist goals, not contact surfaces. Positive box mass creates a free body;
zero denotes fixed scene geometry. Scene centres are INITIAL poses for free bodies.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import xml.etree.ElementTree as ET
from .models import finite_number


@dataclass(frozen=True)
class Box:
    name: str
    center_world: tuple
    half_size: tuple
    rgba: tuple
    mass: float = 0.  # kg; zero means a fixed body, not a massless moving body.
    friction: tuple = (1., .005, .0001)  # sliding, torsional, rolling; MuJoCo units.


@dataclass(frozen=True)
class Marker:
    name: str
    workstation: str
    position_world: tuple
    label: str


@dataclass(frozen=True)
class Scene:
    name: str
    boxes: tuple
    markers: tuple

    def __post_init__(self):
        names = [b.name for b in self.boxes] + [m.name for m in self.markers]
        if len(set(names)) != len(names) or not all(n and n.isidentifier() for n in names):
            raise ValueError('unique identifier names required')
        for box in self.boxes:
            for values, size in ((box.center_world,3), (box.half_size,3), (box.rgba,4)):
                if not isinstance(values,tuple) or len(values)!=size or not all(finite_number(v) for v in values):
                    raise ValueError('finite immutable geometry tuples required')
            if not all(v>0 for v in box.half_size) or not all(0<=v<=1 for v in box.rgba):
                raise ValueError('positive half sizes and RGBA in [0,1] required')
            if not finite_number(box.mass) or box.mass<0:
                raise ValueError('nonnegative box mass required (0 means fixed)')
            if not isinstance(box.friction,tuple) or len(box.friction)!=3 or not all(finite_number(v) and v>=0 for v in box.friction):
                raise ValueError('three nonnegative friction coefficients required')
        for marker in self.markers:
            p = marker.position_world
            if not isinstance(p,tuple) or len(p)!=3 or not all(finite_number(v) for v in p):
                raise ValueError('finite immutable marker position required')
            if marker.workstation not in {b.name for b in self.boxes}:
                raise ValueError('marker must belong to a workstation')

    def marker(self, name):
        for marker in self.markers:
            if marker.name == name:
                return marker
        raise ValueError(f'unknown marker: {name}')

    def to_dict(self):
        # Fresh JSON-compatible lists: a caller cannot modify the fixed scene.
        import json
        return json.loads(json.dumps(dict(asdict(self), frame='mujoco_world_z_up',
                          observation='privileged_static_simulator_geometry',
                          dimensions='box half sizes in metres; mass in kg; centres are initial poses',
                          movable_objects=any(b.mass>0 for b in self.boxes))))


def box_scene():
    """Development object fixture, not a qualified grasp benchmark.

    Parcel: 24 x 24 x 20 cm, 0.5 kg. Table tops at 0.65 m. Friction and
    mass are uncalibrated simulation assumptions. It starts 1 cm above support.
    """
    return Scene('box_transfer_fixture_v1', (
        Box('source_table',(.8,0.,.325),(.35,.45,.325),(.18,.35,.55,1.)),
        Box('destination_table',(2.3,0.,.325),(.35,.45,.325),(.15,.55,.35,1.)),
        Box('parcel',(.8,0.,.76),(.12,.12,.10),(.85,.50,.18,1.),mass=.5),
    ), ())


def workstation_scene(layout_offset_xy=(0., 0.)):
    """Translate the whole static layout in world XY metres, before startup.

    Robot startup is unchanged. An offset defines a task variant, not a
    controller correction; physical feasibility must be evaluated separately.
    """
    if (not isinstance(layout_offset_xy, (list, tuple)) or len(layout_offset_xy) != 2
            or not all(finite_number(v) for v in layout_offset_xy)):
        raise ValueError('layout_offset_xy must contain two finite metres')
    dx, dy = layout_offset_xy
    scene = Scene('workstation_v1', (
        Box('blue_workstation', (.96,-.24,.25), (.10,.16,.25), (.12,.35,.85,1.)),
        Box('yellow_workstation', (.96,.42,.25), (.10,.16,.25), (.90,.65,.10,1.)),
    ), (
        Marker('blue_lower', 'blue_workstation', (.78,-.24,.70), 'lower inspection target at the blue workstation'),
        Marker('blue_upper', 'blue_workstation', (.78,-.24,.86), 'upper inspection target at the blue workstation'),
        Marker('yellow_lower', 'yellow_workstation', (.78,.42,.70), 'lower inspection target at the yellow workstation'),
    ))
    def translated(p):
        return (p[0]+dx, p[1]+dy, p[2])
    return replace(scene,
                   boxes=tuple(replace(b, center_world=translated(b.center_world)) for b in scene.boxes),
                   markers=tuple(replace(m, position_world=translated(m.position_world)) for m in scene.markers))


def write_scene_model(source, destination, scene):
    """Expand MJCF includes, preserve assets/robot, append collision bodies.

    Called before physics startup, never from a policy. The returned XML is also
    the planner/playback model. Sites are nonphysical visual markers.
    """
    import mujoco
    source, destination = Path(source).resolve(), Path(destination).resolve()
    spec = mujoco.MjSpec.from_file(str(source))
    spec.meshdir = str((source.parent/spec.meshdir).resolve())
    spec.texturedir = str((source.parent/spec.texturedir).resolve())
    root = ET.fromstring(spec.to_xml())
    world = root.find('worldbody')
    for box in scene.boxes:
        body = ET.SubElement(world, 'body', name=box.name, pos=' '.join(map(str,box.center_world)))
        if box.mass>0:
            ET.SubElement(body,'freejoint',name=box.name+'_free')
        mass={'mass':str(box.mass)} if box.mass>0 else {}
        ET.SubElement(body, 'geom', name=box.name, type='box',
                      size=' '.join(map(str,box.half_size)), rgba=' '.join(map(str,box.rgba)),
                      contype='1', conaffinity='1', friction=' '.join(map(str,box.friction)),**mass)
    for marker in scene.markers:
        ET.SubElement(world, 'site', name=marker.name, type='sphere', size='.018',
                      pos=' '.join(map(str,marker.position_world)), rgba='0.2 1 0.4 0.8')
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding='unicode')
    return destination
