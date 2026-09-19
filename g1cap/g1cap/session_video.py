"""Explain a recorded session using its fixed goal, measured poses and exact code.

Run on the simulation host after shutdown (the recording references its MJCF).
No dynamics are stepped. Setup is a separate real-time recording. Task playback
starts at unsupported admission and retains every revision gap; the overview
accelerates only code-writing intervals, explicitly.
"""
import argparse
import bisect
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


def checked_source(source, digest):
    if hashlib.sha256(source.encode()).hexdigest() != digest:
        raise ValueError('video source does not match executed source hash')
    return source


def recording_sections(frames, initial, end):
    """Split at recorded admission, never at a convenient successful action.

    Reject support inside the admitted task rather than cropping it away. No
    admission means startup failed: there is no task performance video to make.
    """
    start = frames[0]['sim_time']
    if initial is None:
        return [('setup', start, end, 1)]
    admitted = initial['sim_time']
    if not start <= admitted <= end:
        raise ValueError('task admission lies outside the recording')
    if (initial.get('no_support') is not True or initial.get('support_enabled') is not False
            or any(f.get('no_support') is not True for f in frames
                   if admitted <= f['sim_time'] <= end)):
        raise ValueError('task playback requires continuously unsupported evidence')
    return [('setup',start,admitted,1), ('full',admitted,end,1), ('overview',admitted,end,8)]


def video_schedule(start, end, events, *, fps=15, idle_speed=1):
    """Map output frames to simulation seconds; never accelerate executed rounds."""
    boundaries = {start, end}
    active = []
    opened = None
    for event in events:
        if event['type'] == 'round_start': opened = event['sim_time']
        if event['type'] in ('round_end', 'terminal') and opened is not None:
            active.append((opened, event['sim_time']))
            boundaries.update((opened, event['sim_time']))
            opened = None
    if opened is not None:
        active.append((opened, end)); boundaries.add(opened)
    schedule = []
    points = sorted(t for t in boundaries if start <= t <= end)
    for left, right in zip(points, points[1:]):
        speed = 1 if not events or any(a <= left < b for a, b in active) else idle_speed
        count = math.ceil((right-left)*fps/speed)
        schedule.extend((min(left+i*speed/fps, right), speed) for i in range(count))
    schedule.extend([(end, 0)]*(fps*4))  # Clearly labeled final freeze, not extra dwell.
    return schedule


def render(case, output, selected_sections=None):
    """Export all or selected sections; an empty selection rebuilds metadata only."""
    from .playback import read_recording
    # Software rendering leaves the controller GPUs available to their other users.
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/share/glvnd/egl_vendor.d/50_mesa.json'
    os.environ['LP_NUM_THREADS'] = '2'
    import mujoco as mj
    from mujoco.egl import egl_ext as EGL
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(len(EGL.eglQueryDevicesEXT())-1)
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from OpenGL.GL import glGetString, GL_RENDERER

    case, output = Path(case).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata, frames = read_recording(case/'runtime/poses.jsonl')
    scene = json.loads((case/'runtime/scene.json').read_text())
    trace = case/'session/trace.jsonl'
    events = [json.loads(line) for line in trace.open()] if trace.exists() else []
    summary_path = case/'session/summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    recipe_path = case.parent/(case.name+'.recipe.json')
    recipe = json.loads(recipe_path.read_text()) if recipe_path.exists() else {}
    task = summary.get('task', recipe)
    mobility = task.get('name',task.get('task')) == 'mobility'
    target_id = task.get('target_id')
    target = next((m for m in scene['markers'] if m['name'] == target_id), None)
    if mobility:
        target = dict(position_world=[*task['waypoints'][-1],.02],
                      label='Visit numbered regions in order; stop at each region')
    if target is None:
        raise ValueError('recording needs its fixed workstation target_id to render honestly')
    target_xyz = target['position_world']
    states = [e['observation'] for e in events if e['type'] == 'state']
    initial_state = next((e['observation'] for e in events if e['type'] == 'start'), None)
    initial = initial_state['sim_time'] if initial_state else None
    terminal = next((e for e in events if e['type'] == 'terminal'), None)
    end = terminal['sim_time'] if terminal else frames[-1]['sim_time']
    outcome = terminal['reason'] if terminal else 'startup failed; program never ran'
    if terminal and states:
        last = min(states, key=lambda s: abs(s['sim_time']-end))
        frames = [f for f in frames if f['sim_time'] < end]
        frames.append(dict(sim_time=end, qpos=last.get('qpos',last['pelvis_position']+
                           last['pelvis_quaternion_wxyz']+last['joint_positions']), no_support=last['no_support']))
    sections = recording_sections(frames, initial_state, end)
    if selected_sections and not set(selected_sections)<={s[0] for s in sections}:
        raise ValueError('requested video section is unavailable in this recording')
    if initial_state:
        # Exact measured boundary, not the preceding 30 Hz recording frame.
        frames = [f for f in frames if f['sim_time'] != initial]
        frames.append(dict(sim_time=initial, qpos=initial_state.get('qpos',initial_state['pelvis_position']+
                           initial_state['pelvis_quaternion_wxyz']+initial_state['joint_positions']),
                           no_support=initial_state['no_support']))
        frames.sort(key=lambda f: f['sim_time'])
    rounds = []
    for e in events:
        if e['type'] == 'round_start':
            index = len(rounds)
            source = checked_source((case/f'session/round-{index:02d}/policy.py').read_text(), e['code_sha256'])
            rounds.append(dict(start=e['sim_time'], source=source, sha256=e['code_sha256']))
            (output/f'policy-{index:02d}.py').write_text(source)
    times = [f['sim_time'] for f in frames]
    state_times = [s['sim_time'] for s in states]
    model = mj.MjModel.from_xml_path(metadata['model_path'])
    data = mj.MjData(model)
    model.vis.global_.offwidth = 1050
    model.vis.global_.offheight = 720
    model.vis.quality.offsamples = 0
    renderer = mj.Renderer(model, height=720, width=1050)
    renderer_name = glGetString(GL_RENDERER).decode()
    if 'llvmpipe' not in renderer_name: raise RuntimeError('expected CPU software renderer')
    camera = mj.MjvCamera(); mj.mjv_defaultCamera(camera)
    camera.lookat[:] = [target_xyz[0]-.3, target_xyz[1]+.2, .65]
    camera.distance, camera.azimuth, camera.elevation = 2.7, 45, -24
    options = mj.MjvOption(); options.sitegroup[:] = 0
    font_root = '/usr/share/fonts/truetype/dejavu/'
    font = ImageFont.truetype(font_root+'DejaVuSans.ttf', 24)
    small = ImageFont.truetype(font_root+'DejaVuSans.ttf', 20)
    code_font = ImageFont.truetype(font_root+'DejaVuSansMono.ttf', 14)
    title_font = ImageFont.truetype(font_root+'DejaVuSans-Bold.ttf', 32)
    fps = 15
    provenance = dict(kind='recorded physics poses; mj_forward only, no new dynamics',
                      case=str(case), renderer=renderer_name, target=target, outcome=outcome,
                      task_admission_sim_time=initial,
                      rendered_sections=([s[0] for s in sections] if selected_sections is None else selected_sections),
                      sections=[dict(name=n,start_sim_time=a,end_sim_time=b,idle_speed=s)
                                for n,a,b,s in sections],
                      rounds=[dict(index=i, sha256=r['sha256']) for i,r in enumerate(rounds)])
    try:
        for name, section_start, section_end, speed in sections:
            if selected_sections is not None and name not in selected_sections: continue
            setup = name == 'setup'
            schedule = video_schedule(section_start, section_end, events, fps=fps, idle_speed=speed)
            timeline = []
            log = (output/f'{name}-render.log').open('w')
            proc = subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24',
                    '-s','1920x1080','-r',str(fps),'-i','-','-an','-c:v','libx264',
                    '-threads','2','-preset','fast','-crf','21','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(output/f'{name}.mp4')],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            try:
                for k, (sim, rate) in enumerate(schedule):
                    frame = frames[max(0, bisect.bisect_right(times, sim)-1)]
                    data.qpos[:] = frame['qpos']; mj.mj_forward(model, data)
                    if mobility:
                        camera.lookat[:] = [data.qpos[0]+.3,data.qpos[1],.6]
                        camera.distance, camera.azimuth, camera.elevation = 3.4, 110, -28
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
                    renderer.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
                    markers = [(m['position_world'], .018, [1,.7,.1,1] if m['name']==target_id else [.5,.5,.5,1]) for m in scene['markers']]
                    wrist = data.body('right_wrist_yaw_link').xpos
                    if mobility:
                        markers += [([*p,.015],.12,[1,.7,.1,.6]) for p in task['waypoints']]
                    else:
                        markers += [(target_xyz, .05, [1,.7,.1,.2]), (wrist, .014, [0,1,1,1])]
                    for pos, radius, rgba in markers:
                        sc = renderer.scene
                        mj.mjv_initGeom(sc.geoms[sc.ngeom],
                                        mj.mjtGeom.mjGEOM_CYLINDER if mobility else mj.mjtGeom.mjGEOM_SPHERE,
                                        [radius,.005,0] if mobility else [radius,0,0],
                                        pos, np.eye(3).ravel(), rgba)
                        sc.ngeom += 1
                    canvas = Image.new('RGB', (1920,1080), '#101722')
                    canvas.paste(Image.fromarray(renderer.render()), (0,175))
                    draw = ImageDraw.Draw(canvas)
                    # Wrist origins lie inside the joint mesh. Project a ring on
                    # top of the image so the measured endpoint stays visible.
                    eye = np.mean([c.pos for c in renderer.scene.camera], axis=0)
                    forward = np.mean([c.forward for c in renderer.scene.camera], axis=0)
                    up = np.mean([c.up for c in renderer.scene.camera], axis=0)
                    right = np.cross(forward, up)
                    scale = 720/(2*math.tan(math.radians(model.vis.global_.fovy)/2))
                    def pixel(position):
                        delta = np.asarray(position)-eye
                        depth = np.dot(delta, forward)
                        return (525+scale*np.dot(delta,right)/depth,
                                175+360-scale*np.dot(delta,up)/depth)
                    wx, wy = pixel(wrist)
                    gx, gy = pixel(target_xyz)
                    if not setup and not mobility:
                        draw.line((wx,wy,gx,gy), fill='#53f3ff', width=2)
                        draw.ellipse((wx-9,wy-9,wx+9,wy+9), outline='#53f3ff', width=3)
                    def text(x,y,value,f=font,color='white'):
                        draw.text((x,y),value,font=f,fill=color)
                    if setup:
                        text(24,18,'SETUP ONLY | NOT BENCHMARK EXECUTION',title_font,'#f9cf60')
                        text(24,67,'Controller initialization with an artificial pelvis stabilizer, then unsupported settling.')
                        text(24,105,'The support applies force AND torque. It is not a modeled real harness.',small)
                        if not frame.get('no_support'):
                            px, py = pixel(data.body('pelvis').xpos)
                            draw.ellipse((px-24,py-24,px+24,py+24),outline='#f9cf60',width=4)
                            text(24,205,'YELLOW RING: EXTERNALLY STABILIZED PELVIS',small,'#f9cf60')
                    elif mobility:
                        text(24,18,'WALK THE ROUTE. STOP IN THE NUMBERED REGIONS.',title_font)
                        route=[task.get('start_world_xy',[0,0]),*task['waypoints']]
                        length=sum(math.dist(a,b) for a,b in zip(route,route[1:]))
                        text(24,67,f'Route length {length:.1f} m | corridor half-width {task.get("corridor_half_width",.6):.1f} m | final yaw {task.get("final_yaw")} rad')
                        text(24,105,'Stop within 12 cm at each region for 0.5 s. No external support during the task.',small)
                    else:
                        text(24,18,'WALK TO THE STATION. REACH THE GOLD MARKER.',title_font)
                        text(24,67,target['label'].capitalize()+'. Keep a stable stance; avoid hitting either station.')
                        text(24,105,'Start: verified unsupported standing. Goal: wrist within 5 cm, stable for 0.5 s.',small)
                    round_index = bisect.bisect_right([r['start'] for r in rounds],sim)-1
                    active = None
                    phase = 'Writing the first program; robot remains alive'
                    last_rejection = ''
                    for event in events:
                        if event['sim_time'] > sim: break
                        kind = event['type']
                        if kind == 'round_start': phase = 'Running program '+str(round_index+1)
                        elif kind == 'round_end':
                            phase = 'Editing the next program; same robot, same world'; active = None
                        elif kind == 'navigation_start':
                            active = event['operation']; phase = ('Walking toward the next region' if mobility else 'Walking toward the station') if active=='walk_to' else 'Executing '+active
                        elif kind == 'stationary_tool' and event['event']['type']=='command':
                            active = event['event']['operation']; phase = 'Reaching for the gold marker' if active=='reach_right' else 'Executing '+active
                        elif kind == 'tool_result' and event['method'] not in ('observe','observe_scene'):
                            if event['result'].get('status')=='rejected':
                                last_rejection = event['method']+': '+event['result'].get('reason','rejected')
                                phase = 'Reach plan rejected; robot has not completed the task'
                            active = None
                    if setup:
                        phase = 'Checking unsupported standing' if frame.get('no_support') else 'ARTIFICIAL SUPPORT ACTIVE'
                        if sim >= section_end:
                            phase = 'SETUP COMPLETE | unsupported task start' if initial is not None else 'SETUP FAILED | task never started'
                    elif sim >= end:
                        phase = ('SUCCESS' if outcome=='success' else 'NOT COMPLETED')+' | '+outcome
                        active = None
                    playback = 'FINAL POSE FREEZE' if rate==0 else f'{rate}x playback'
                    elapsed = f'Setup clock {sim-section_start:.1f} s' if setup else f'Task clock {sim-initial:.1f} s'
                    text(24,140,elapsed+' | '+playback+' | '+phase,small,'#8ce5ff')
                    if setup:
                        text(24,914,'No task is being scored. No agent program is executing.')
                    elif mobility:
                        text(24,914,f'Base world XY: ({data.qpos[0]:.2f}, {data.qpos[1]:.2f}) m')
                    else:
                        text(24,914,f'Wrist-to-goal distance: {math.dist(wrist,target_xyz)*100:.1f} cm   |   goal <= 5 cm')
                    if not setup and states and sim >= state_times[0]:
                        state = states[bisect.bisect_right(state_times,sim)-1]
                        text(24,952,f'Base speed {math.hypot(*state["planar_velocity"])*100:.1f} cm/s   |   goal <= 5 cm/s',small)
                    text(24,990,'Artificial setup is separate from task performance.' if setup else
                         ('Gold = arrival region. Map shows fixed route and current robot.' if mobility else
                          'Gold = goal region. Cyan ring = measured wrist (overlay).'),small,'#f9cf60')
                    text(24,1027,'Recorded simulation. No grasping. Simulator-derived observations.',small)
                    text(1080,185,'INITIALIZATION PROTOCOL' if setup else 'EXACT EXECUTED PYTHON',font,'#8ce5ff')
                    if setup:
                        text(1080,270,'1. Initialize the controller with external support.',small)
                        text(1080,320,'2. Establish load-bearing contact at both feet.',small)
                        text(1080,370,'3. Remove the artificial force and torque.',small)
                        text(1080,420,'4. Verify unsupported two-foot settling.',small)
                        text(1080,455,'   Startup thresholds are recorded by the runtime.',small)
                        text(1080,515,'5. Admit the task. No pose or time reset.',small)
                        text(1080,610,'If initialization fails, no task video is produced.',small,'#f9cf60')
                        text(1080,660,'Support is never restored to rescue the task.',small)
                    elif round_index >= 0:
                        current = rounds[round_index]
                        text(1080,223,f'Program {round_index+1} | SHA256 {current["sha256"][:16]}',small)
                        lines = current['source'].splitlines()
                        active_line = next((i for i,line in enumerate(lines) if active and 'robot.'+active+'(' in line), None)
                        first = max(0,(active_line or 0)-20) if len(lines)>41 else 0
                        for i,line in enumerate(lines[first:first+41], first):
                            visible = line if len(line)<=94 else line[:91]+'...'
                            text(1080,269+(i-first)*17,f'{i+1:02d} '+visible,code_font,
                                 '#ffd65b' if active_line==i else '#eeeeee')
                        text(1080,987,'Highlight follows tool events, not Python tracing.',small)
                    else:
                        text(1080,270,'No program has executed yet.',small)
                        text(1080,307,'The agent reads the task and tool contracts,',small)
                        text(1080,342,'then writes policy.py in its coding workspace.',small)
                    if not setup and last_rejection:
                        text(1080,1024,'Earlier result: '+last_rejection,small,'#f7bc91')
                    if mobility and not setup:
                        # Fixed world map complements the following camera. It never
                        # moves the goals or hides lateral drift at long distances.
                        points=[task.get('start_world_xy',[0,0]),*task['waypoints']]
                        xmin,xmax=min(p[0] for p in points)-.7,max(p[0] for p in points)+.7
                        ymin,ymax=min(p[1] for p in points)-.7,max(p[1] for p in points)+.7
                        scale_map=min(280/(xmax-xmin),100/(ymax-ymin))
                        def map_xy(p): return (40+(p[0]-xmin)*scale_map,845-(p[1]-ymin)*scale_map)
                        draw.rounded_rectangle((20,655,355,865),radius=12,fill='#172335')
                        text(35,665,'WORLD MAP | stops',small)
                        poly=[map_xy(p) for p in points]
                        draw.line(poly,fill='#314962',width=max(1,int(2*task.get('corridor_half_width',.6)*scale_map)))
                        draw.line(poly,fill='#afc5dd',width=2)
                        for j,p in enumerate(points[1:]):
                            x,y=map_xy(p); r=max(4,.12*scale_map)
                            draw.ellipse((x-r,y-r,x+r,y+r),fill='#ffd65b')
                            text(x+6,y-24,str(j+1),small)
                        x,y=map_xy(data.qpos[:2])
                        draw.ellipse((x-5,y-5,x+5,y+5),fill='#53f3ff')
                        if states and sim>=state_times[0]:
                            state=states[bisect.bisect_right(state_times,sim)-1]
                            progress=next((e.get('metrics',{}) for e in reversed(events)
                                           if e['sim_time']<=sim and e['type']=='state'),{})
                            # Metrics in trace are scored continuously, never inferred
                            # from the program's tool-completion messages.
                            count=progress.get('waypoint_index',0)
                            text(24,870,f'Stops completed: {count}/{len(task["waypoints"])}',small,'#8ce5ff')
                    proc.stdin.write(canvas.tobytes())
                    if k % fps == 0: timeline.append(dict(video_time=k/fps,sim_time=sim,rate=rate,round=round_index,phase=phase))
                    if k == len(schedule)-1: canvas.save(output/f'{name}-preview.jpg')
            finally:
                proc.stdin.close(); returncode = proc.wait(); log.close()
            if returncode: raise RuntimeError(f'{name} ffmpeg failed; see render log')
            (output/f'{name}-timeline.json').write_text(json.dumps(timeline,indent=2))
            print(output/f'{name}.mp4',flush=True)
        (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    finally:
        renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--section',action='append',choices=('setup','full','overview'),
                        help='Export only this section; repeat to recover selected video files.')
    args = parser.parse_args()
    render(args.case, args.out or args.case/'video', args.section)


if __name__ == '__main__': main()
