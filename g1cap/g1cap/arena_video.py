"""Render complete Arena recordings with actual program/tool/outcome provenance.

No simulation, reconstructed motion or substituted camera footage. Run after
shutdown; this CPU exporter works on the collected artifact directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_evidence(root):
    root=Path(root)
    states=read_rows(root/'physics/states.jsonl');actions=read_rows(root/'physics/actions.jsonl')
    if not states or len(states)!=len(actions)+1:raise ValueError('state/action count mismatch')
    if any(r['step']!=i or abs(r['time']-.02*i)>1e-6 for i,r in enumerate(states)):
        raise ValueError('recorded state sequence mismatch')
    if any(r['step']!=i+1 for i,r in enumerate(actions)):raise ValueError('recorded action sequence mismatch')
    sources=[];programs=[]
    for path in sorted((root/'session').glob('round-*/policy.py')):
        result=json.loads(path.with_name('result.json').read_text())
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if result['source_sha256']!=sha:raise ValueError('executed source hash mismatch')
        sources.append(dict(file=str(path.relative_to(root)),sha256=sha))
        if 'start_observation' in result and 'end_observation' in result:
            programs.append((result['start_observation']['time'],result['end_observation']['time']))
    return dict(states=states,actions=actions,sources=sources,programs=programs,
                trace=read_rows(root/'session/tool-trace.jsonl'),
                summary=json.loads((root/'session/summary.json').read_text()))


def render_session(root, *, timeout=1200):
    root=Path(root).resolve();evidence=load_evidence(root)
    rows=evidence['states'];summary=evidence['summary'];kind=summary['task'].get('task_kind','source_return')
    out=root/'video';out.mkdir(exist_ok=True);labels=out/'labels';labels.mkdir(exist_ok=True)
    fonts=[Path('/System/Library/Fonts/Supplemental/Arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    font=next((p for p in fonts if p.exists()),None)
    if font is None:raise RuntimeError('Arial or DejaVuSans font required')
    filters=['[0:v]scale=900:675,pad=1280:960:0:120:color=0x101b2b[scene]',
             '[1:v]scale=380:285[head]','[scene][head]overlay=900:130']
    def label(name,text,x,y,size=22,interval=None):
        for index,line in enumerate(text.splitlines()):
            path=labels/f'{name}-{index}.txt';path.write_text(line)
            # Quote paths for FFmpeg's filter parser. Project paths are ordinary
            # local paths, not shell commands; subprocess always receives argv.
            escape=lambda p:str(p).replace('\\','\\\\').replace("'", "'\\''").replace(':','\\:')
            filters[-1]+=f",drawtext=fontfile='{escape(font)}':textfile='{escape(path)}':x={x}:y={y+index*(size+5)}:fontsize={size}:fontcolor=white:box=1:boxcolor=black@0.65"
            if interval:filters[-1]+=f":enable='gte(t,{interval[0]})*lt(t,{interval[1]})'"
    title='TASK: PICK UP THE BOX, PUT IT BACK ON THE SOURCE TABLE, RELEASE'
    goal='GOAL\nLift with both hands\nReturn to source\nRelease and stand'
    if kind=='retreat_hold':
        distance=summary['task']['retreat_distance_m']
        title=f'TASK: PICK UP THE BOX, CARRY BACKWARD {distance*100:g} CM, STOP AND HOLD'
        goal=f'GOAL\nLift with both hands\nCarry back {distance*100:g} cm\nKeep holding and stand'
    elif kind=='table_transfer':
        title='TASK: MOVE THE BOX FROM THE GREY TABLE TO THE GREEN TABLE'
        goal='GOAL\nLift with both hands\nBox on green table\nRelease and stand'
    label('title',title,18,16,25)
    label('source','Recorded Arena physics | Exact executed Python and complete tool trace retained',18,58,21)
    label('head','HEAD CAMERA',915,430,22);label('goal',goal,915,480,22)
    size=summary['task'].get('box_size_m',[.2,.2,.2]);mass=summary['task'].get('box_mass_kg',.1)
    dimensions=' x '.join(f'{100*v:g}' for v in size)
    label('limits',f'SIMULATION\n{dimensions} cm / {mass:g} kg\nMeasured pose/contact\nNot hardware evidence',915,705,18)
    metrics=summary['metrics'];success=metrics['success']
    reason=metrics.get('failure') or summary.get('terminal_reason') or 'incomplete'
    label('outcome','FINAL PHYSICAL OUTCOME: '+('PASS' if success else 'FAILED / '+reason.replace('_',' ')),18,810,24)
    label('provenance','Full recording, 1x simulation time | All waits, actions and failures retained',18,925,20)
    names={'idle':'WAIT / RETAIN REFERENCES','wait':'WAIT WITH ZERO NAVIGATION','acquire':'PICK UP THE BOX',
           'verify_pickup':'VERIFY STABLE GRASP','supported_lift':'LIFT FROM SUPPORTED TWO-HAND GRASP',
           'verify_supported_lift':'VERIFY RAISED HOLD','lift':'RAISE HELD BOX','verify_lift':'VERIFY HEIGHT',
           'hold':'HOLD BOX','prepare_carry':'PREPARE GRASP ONCE','loaded_retreat':'CARRY BACKWARD',
           'settle_loaded':'STOP AND VERIFY HOLD','loaded_forward':'CARRY FORWARD','settle_forward':'STOP AND VERIFY HOLD',
           'loaded_turn':'TURN WHILE HOLDING','settle_turn':'SETTLE AT REQUESTED HEADING',
           'turn_hold':'VERIFY TURN WITH ZERO NAVIGATION','place_align':'LEVEL THE HELD BOX',
           'place_align_hold':'VERIFY LEVEL HOLD','place_retract':'CREATE TABLE CLEARANCE',
           'place_retract_hold':'VERIFY CLEARANCE AND HOLD','place_lower':'LOWER ONTO SELECTED TABLE',
           'place_supported_hold':'VERIFY ACTUAL TABLE SUPPORT','place_withdraw':'WITHDRAW BOTH HANDS',
           'place_settle':'VERIFY STABLE RELEASE'}
    boundaries=[]
    for row in rows:
        phase=row['phase'];programs=evidence['programs']
        if phase=='idle' and programs:
            if row['time']<programs[0][0]:phase='INITIAL STANDING / PROGRAM NOT STARTED'
            elif any(end<row['time']<start for (_,end),(start,_) in zip(programs,programs[1:])):
                phase='BETWEEN PROGRAMS / PHYSICS CONTINUES'
        if not boundaries or boundaries[-1][1]!=phase:boundaries.append((row['time'],phase))
    boundaries.append((rows[-1]['time']+.02,None))
    for i,((start,phase),(end,_)) in enumerate(zip(boundaries,boundaries[1:])):
        label(f'phase-{i}',names.get(phase,phase.replace('_',' ').upper()),18,95,22,(start,end))
    requests=[r for r in evidence['trace'] if r['type']=='tool_request']
    for i,r in enumerate(requests):
        end=requests[i+1]['sim_time'] if i+1<len(requests) else rows[-1]['time']+.02
        args=', '.join(f'{k}={v!r}' for k,v in r['arguments'].items())
        label(f'call-{i}',f"Program {int(r['round_id'].split(':')[-1])+1}: robot.{r['method']}({args})",18,850,22,(r['sim_time'],end))
    results=[r for r in evidence['trace'] if r['type']=='tool_result']
    for i,r in enumerate(results):
        end=results[i+1]['sim_time'] if i+1<len(results) else rows[-1]['time']+.02
        outcome=r['result'];detail=f"{outcome['status']}: {outcome['reason']}"
        text='LAST TOOL RESULT\n'+r['method']+'\n'+'\n'.join(textwrap.wrap(detail,34))
        label(f'result-{i}',text,915,605,16,(r['sim_time'],end))
    for i,row in enumerate(rows[::10]):
        text=f"Time {row['time']:.1f} s | Both hands holding: {row['bilateral']} | Box clearance: {row['clearance']*100:.1f} cm"
        if kind=='table_transfer':
            d=row['surfaces'][summary['task']['surface_id']]
            text=f"Time {row['time']:.1f} s | Target footprint: {d['contained']} | Table support: {d['supported']} | Hand contact: {max(row['hand_forces_N'].values()):.1f} N"
        label(f'state-{i}',text,18,889,20,(row['time'],row['time']+.2))
    filters[-1]+='[v]';script=out/'filters.txt';script.write_text(';'.join(filters));video=out/'full.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(root/'physics/overview.mp4'),'-i',str(root/'physics/head.mp4'),
                    '-filter_complex_script',str(script),'-map','[v]','-c:v','libx264','-threads','2','-crf','20',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True,timeout=timeout)
    verified={}
    for path in (root/'physics/overview.mp4',root/'physics/head.mp4',video):
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True,timeout=timeout)
        stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames,duration,width,height','-of','json',str(path)],timeout=timeout))['streams'][0]
        if int(stream['nb_read_frames'])!=len(rows):raise ValueError('camera/state frame count mismatch')
        verified[str(path.relative_to(root))]=stream
    provenance=dict(kind='recorded_physics_with_labels',sources=evidence['sources'],frames=len(rows),
                    renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    task=summary['task'],physical_metrics=metrics,terminal_reason=summary['terminal_reason'],media=verified)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    return video


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('artifacts',type=Path)
    args=parser.parse_args();print(render_session(args.artifacts))

if __name__=='__main__':main()
