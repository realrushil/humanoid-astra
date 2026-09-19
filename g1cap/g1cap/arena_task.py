"""Independent source-return score from recorded simulator measurements.

Development fixture: 20 cm cube, 0.1 kg. These thresholds are assumptions,
not hardware calibration. A returned tool/program is never a task success.
"""
from collections import deque
import itertools
import math

from .arena_observation import assess_hold, angular_distance


def contained(obs, bounds):
    q=obs['box_quat'];norm=math.sqrt(sum(v*v for v in q))
    if norm<1e-8:return False
    w,x,y,z=[v/norm for v in q]
    rows=((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)),
          (2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)))
    return all(bounds['min'][i]<=obs['box_pos'][i]+sum(rows[i][j]*c[j] for j in range(3))<=bounds['max'][i]
               for c in itertools.product((-.1,.1),repeat=3) for i in range(2))


class ArenaBoxTask:
    def __init__(self,source_bounds):
        self.bounds=source_bounds
        self.history=deque(maxlen=51)
        self.held_at=None
        self.released_at=None
        self.failure=None
        self.peak_contact_N=0.

    def update(self,obs):
        self.history.append(obs)
        self.peak_contact_N=max(self.peak_contact_N,obs['robot_source_peak_N'])
        if self.peak_contact_N>5:self.failure='forbidden_source_contact'
        if obs['tilt']>math.radians(15):self.failure=self.failure or 'body_tilt_limit'
        samples=list(self.history)
        if self.held_at is None and assess_hold(samples)['ready'] and all(s['stance_clear'] for s in samples):
            self.held_at=obs['time']
        if self.held_at is None or len(samples)<51 or samples[0]['time']<=self.held_at:return
        pairs=list(zip(samples,samples[1:]))
        if any(abs(b['time']-a['time']-.02)>1e-6 for a,b in pairs):return
        supported=all(s['supported'] and s['stance_clear'] and contained(s,self.bounds)
                      and max(s['hand_forces_N'].values())<=.5 for s in samples)
        stable=all(math.dist(a['box_pos'],b['box_pos'])/.02<=.05 and
                   angular_distance(a['box_quat'],b['box_quat'])/.02<=.2 for a,b in pairs)
        if supported and stable and self.released_at is None:self.released_at=obs['time']

    def metrics(self):
        return dict(success=self.held_at is not None and self.released_at is not None and self.failure is None,
                    stable_hold_at_s=self.held_at,supported_release_at_s=self.released_at,
                    failure=self.failure,peak_robot_source_contact_N=self.peak_contact_N,
                    observation_source='simulator_ground_truth')


class ArenaRetreatTask:
    """Version2: score physical transport, not the controller's grasp-pose proxy.

    Keep the first stable-hold anchor for distance. Deliberate lifting may change
    box pose relative to the pelvis; record that change without treating it as
    lost retention. Continuous contact/clearance and physical faults still count.
    """
    def __init__(self,distance_m):
        if isinstance(distance_m,bool) or not isinstance(distance_m,(int,float)) or not .20<=distance_m<=.50:
            raise ValueError('retreat target must be20–50 cm')
        self.distance=distance_m
        self.history=deque(maxlen=51)
        self.anchor=None
        self.failure=None
        self.completed_at=None
        self.current=None
        self.peak_contact_N=0.
        self.maximum_box_pelvis_change_m=0.

    def update(self,obs):
        from .arena_retreat import contact_fault, relative_box
        self.history.append(obs);self.current=obs
        self.peak_contact_N=max(self.peak_contact_N,obs['robot_source_peak_N'])
        if self.peak_contact_N>5:self.failure='forbidden_source_contact'
        if obs['tilt']>math.radians(15):self.failure=self.failure or 'body_tilt_limit'
        stable=assess_hold(list(self.history))['ready'] and all(s['stance_clear'] for s in self.history)
        if self.anchor is None:
            if stable:
                from copy import deepcopy
                self.anchor=deepcopy(obs)
            return
        fault=contact_fault(obs)
        if not obs['bilateral'] or obs['clearance']<.03:fault=fault or 'grasp_or_height_lost'
        if abs(obs['root_pos'][1]-self.anchor['root_pos'][1])>.05:fault=fault or 'lateral_drift'
        self.maximum_box_pelvis_change_m=max(self.maximum_box_pelvis_change_m,
            math.dist(relative_box(self.anchor),relative_box(obs)))
        if angular_distance(self.anchor['root_quat'],obs['root_quat'])>.20:fault=fault or 'body_orientation_drift'
        self.failure=self.failure or fault
        if stable and self._at_goal() and self.failure is None:self.completed_at=obs['time']

    def _at_goal(self):
        if self.anchor is None or self.current is None:return False
        root=self.anchor['root_pos'][0]-self.current['root_pos'][0]
        box=self.anchor['box_pos'][0]-self.current['box_pos'][0]
        return self.distance-.02<=root<=self.distance+.03 and box>=self.distance-.05

    def metrics(self):
        stable=assess_hold(list(self.history))['ready'] and all(s['stance_clear'] for s in self.history)
        return dict(scoring_version='retreat_outcome_v2',
                    maximum_box_pelvis_change_m=self.maximum_box_pelvis_change_m,
                    success=self.completed_at is not None and self.failure is None and self._at_goal() and stable,
                    stable_hold_at_s=self.anchor['time'] if self.anchor else None,
                    retreat_hold_at_s=self.completed_at,failure=self.failure,
                    root_retreat_m=self.anchor['root_pos'][0]-self.current['root_pos'][0] if self.anchor else None,
                    box_retreat_m=self.anchor['box_pos'][0]-self.current['box_pos'][0] if self.anchor else None,
                    peak_robot_source_contact_N=self.peak_contact_N,observation_source='simulator_ground_truth')


class ArenaTransferTask:
    """Version1: actual lift, travel and released support on the selected table.

    Tool status/phase is never an input. Recoverable motion stops remain in the
    session log; floor contact, body tilt, table collisions and missing evidence
    latch failure. The 0.5m minimum travel is a development task requirement.
    """
    def __init__(self,surface_id,minimum_travel_m=.5,*,box_size_m=None,box_mass_kg=None):
        if surface_id not in ('source','destination'):raise ValueError('unknown support')
        if not isinstance(minimum_travel_m,(int,float)) or isinstance(minimum_travel_m,bool) or not math.isfinite(minimum_travel_m) or minimum_travel_m<=0:
            raise ValueError('positive finite minimum travel required')
        from .arena_surfaces import box_bounds, support_observation
        # Explicit owner parameters require matching evidence on every frame.
        # The no-parameter constructor retains historical20cm/0.1kg scoring.
        if (box_size_m is None)!=(box_mass_kg is None):raise ValueError('supply both box dimensions and mass')
        self.explicit_box=box_size_m is not None
        self.box_size=list(box_size_m) if self.explicit_box else [.2,.2,.2]
        self.box_mass=box_mass_kg if self.explicit_box else .1
        bound=box_bounds(dict(box_pos=[0,0,0],box_quat=[1,0,0,0],box_size_m=self.box_size))
        support_observation(bound,bound,0,0,0,mass_kg=self.box_mass)
        self.surface_id=surface_id;self.minimum_travel=minimum_travel_m
        self.history=deque(maxlen=51);self.anchor=None;self.current=None;self.bounds=None
        self.failure=None;self.released_at=None;self.release_verified=False
        self.peak_contact_N=0.;self.peak_floor_contact_N=0.

    def update(self,obs):
        from copy import deepcopy
        from .arena_surfaces import box_bounds,contained as fits
        try:
            if self.explicit_box and ('box_size_m' not in obs or 'box_mass_kg' not in obs):
                raise ValueError('missing object properties')
            if obs.get('box_size_m',[.2,.2,.2])!=self.box_size or obs.get('box_mass_kg',.1)!=self.box_mass:
                raise ValueError('object properties differ from owner configuration')
            surface=obs['surfaces'][self.surface_id]
            numbers=[obs['time'],obs['tilt'],obs['box_floor_contact_peak_N'],surface['box_upward_N'],
                     *obs['box_pos'],*obs['box_quat'],*obs['hand_forces_N'].values(),
                     *surface['bounds']['min'],*surface['bounds']['max']]
            numbers.extend(s['robot_contact_peak_N'] for s in obs['surfaces'].values())
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in numbers):
                raise ValueError('nonfinite measurement')
            if not {'source','destination'}<=obs['surfaces'].keys():raise ValueError('missing support')
            if self.bounds is None:self.bounds=deepcopy(surface['bounds'])
            elif self.bounds!=surface['bounds']:raise ValueError('static support changed')
            if self.current is not None and abs(obs['time']-self.current['time']-.02)>1e-6:
                self.failure=self.failure or 'missing_or_reordered_observation'
            self.current=deepcopy(obs)
            self.peak_contact_N=max(self.peak_contact_N,*(s['robot_contact_peak_N'] for s in obs['surfaces'].values()))
            self.peak_floor_contact_N=max(self.peak_floor_contact_N,obs['box_floor_contact_peak_N'])
            if self.peak_contact_N>5:self.failure=self.failure or 'forbidden_table_contact'
            if self.peak_floor_contact_N>.5:self.failure=self.failure or 'box_floor_contact'
            if obs['tilt']>math.radians(15):self.failure=self.failure or 'body_tilt_limit'
            self.history.append(self.current);samples=list(self.history)
            if self.anchor is None and assess_hold(samples)['ready'] and all(r['stance_clear'] for r in samples):
                self.anchor=deepcopy(obs)
            self.release_verified=False
            if self.anchor is None or len(samples)<51 or samples[0]['time']<=self.anchor['time']:return
            if math.dist(obs['box_pos'][:2],self.anchor['box_pos'][:2])<self.minimum_travel:return
            supported=all(r['surfaces'][self.surface_id]['box_upward_N']>=.5*self.box_mass*9.81
                          and abs(box_bounds(r)['min'][2]-self.bounds['max'][2])<=.01
                          and fits(box_bounds(r),self.bounds) and r['stance_clear']
                          and max(r['hand_forces_N'].values())<=.5 for r in samples)
            stable=all(math.dist(a['box_pos'],b['box_pos'])/.02<=.05
                       and angular_distance(a['box_quat'],b['box_quat'])/.02<=.2
                       for a,b in zip(samples,samples[1:]))
            self.release_verified=supported and stable
            if self.release_verified and self.released_at is None:self.released_at=obs['time']
        except (KeyError,TypeError,ValueError,IndexError):
            self.failure=self.failure or 'invalid_transfer_observation';self.release_verified=False

    def metrics(self):
        return dict(scoring_version='selected_table_transfer_v2' if self.explicit_box else 'selected_table_transfer_v1',surface_id=self.surface_id,
                    box_size_m=self.box_size,box_mass_kg=self.box_mass,
                    success=self.release_verified and self.failure is None,
                    stable_hold_at_s=self.anchor['time'] if self.anchor else None,
                    supported_release_at_s=self.released_at,failure=self.failure,
                    box_displacement_m=math.dist(self.current['box_pos'][:2],self.anchor['box_pos'][:2]) if self.anchor and self.current else None,
                    peak_robot_table_contact_N=self.peak_contact_N,peak_box_floor_contact_N=self.peak_floor_contact_N,
                    observation_source='simulator_ground_truth')
