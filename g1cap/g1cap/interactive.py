"""Mac-side bounded code revisions against one live Linux G1 session."""
import argparse
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import time
import uuid

from .runner import write_json
from .session_tools import session_api
from .workspace_agent import WorkspaceCodexAgent
from .visual_observation import validate_snapshot
from .source_overlay import stage_overlay


class RemoteSession:
    def __init__(self,host,root,gpu,output,*,recipe=None,max_rounds=4,port=15576):
        if not host or host.startswith('-') or not root.startswith('/'):
            raise ValueError('explicit SSH host and absolute remote root required')
        self.host,self.root,self.gpu=host,PurePosixPath(root),gpu
        self.output=Path(output).resolve()
        self.output.mkdir(parents=True,exist_ok=False)
        self.remote=self.root/'runs'/('session-'+uuid.uuid4().hex)
        self.recipe=recipe or {}
        self.arena=self.recipe.get('backend')=='arena'
        self.inbox=self.remote/('session/inbox' if self.arena else 'inbox')
        self.max_rounds,self.port=max_rounds,port
        self.process=self.log=None
        self.session_id=None
        self.index=0

    def _ssh(self,command,timeout=20):
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',self.host,command],
                              capture_output=True,text=True,timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'SSH failed: {result.stderr[-1500:]}')
        return result.stdout

    def _send_source_overlay(self):
        """Stage the exact local package in this session's isolated remote dir."""
        if not self.arena:
            return None
        # Keep the overlay beside (rather than inside) the launcher output;
        # arena_launch owns creation of the output directory itself.
        overlay=self.remote.with_name(self.remote.name+'.source-overlay')
        local=self.output/'source-overlay'
        stage_overlay(local)
        self._ssh('mkdir -p '+shlex.quote(str(overlay)),timeout=30)
        subprocess.run(['scp','-q','-r',str(local/'g1cap'),str(local/'source-manifest.json'),
                        f'{self.host}:{overlay}/'],check=True,timeout=90)
        return overlay

    def _json(self,path,timeout=20):
        content=self._ssh('if test -f '+shlex.quote(str(path))+'; then cat '+shlex.quote(str(path))+'; fi',timeout=timeout)
        return json.loads(content) if content.strip() else None

    def _send(self,local,remote,timeout=30):
        subprocess.run(['scp','-q',str(local),f'{self.host}:{remote}'],check=True,timeout=timeout)

    def start(self):
        recipe_file=self.output/'recipe.json'
        write_json(recipe_file,self.recipe)
        remote_recipe=str(self.remote)+'.recipe.json'
        self._send(recipe_file,remote_recipe)
        command=[str(self.root/'.venv-sonic/bin/python'),'-u','-m','g1cap.session_runtime',
                 '--root',str(self.root),'--out',str(self.remote),'--gpu',str(self.gpu),
                 '--port',str(self.port),'--recipe',remote_recipe,'--max-rounds',str(self.max_rounds)]
        if self.arena:
            overlay=self._send_source_overlay()
            command=['python3','-u','-m','g1cap.arena_launch','--root',str(self.root),'--out',str(self.remote),
                     '--gpu',str(self.gpu),'--model-gpu',str(self.recipe.get('model_gpu',0)),
                     '--port',str(self.port),'--recipe',remote_recipe,'--max-rounds',str(self.max_rounds),
                     '--source-overlay',str(overlay)]
        launch_cwd='/tmp' if self.arena else str(self.root)
        shell='cd '+shlex.quote(launch_cwd)+' && '
        if self.arena:
            # The launcher module itself must resolve from the same isolated
            # overlay before it can pass the overlay to the physics child.
            shell+='PYTHONPATH='+shlex.quote(str(overlay))+':'+shlex.quote(str(self.root))+':'+shlex.quote(str(self.root/'temp/Arena'))+' '
        shell+=shlex.join(command)
        self.log=(self.output/'remote.log').open('w')
        self.process=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',self.host,shell],
                                      stdout=self.log,stderr=subprocess.STDOUT)
        write_json(self.output/'remote.json',dict(host=self.host,root=str(self.root),artifact_path=str(self.remote),
                    gpu=self.gpu,max_rounds=self.max_rounds,command=command))
        startup_timeout=420 if self.arena else 200
        end=time.monotonic()+startup_timeout
        while time.monotonic()<end:
            ready=self._json(self.remote/'ready.json')
            if ready:
                self.session_id=ready['session_id']
                return self.status()
            if self.process.poll() is not None:
                raise RuntimeError(f'remote startup failed: {self._json(self.remote/"error.json")}')
            time.sleep(.5)
        raise TimeoutError(f'session startup exceeded {startup_timeout} seconds')

    def status(self):
        result=self._json(self.remote/'session/summary.json') or self._json(self.remote/'session/status.json')
        if result is None:
            raise RuntimeError('missing session status')
        return result

    def execute(self,source):
        folder=self.output/f'submission-{self.index:02d}'
        folder.mkdir()
        request=folder/'request.json'
        write_json(request,dict(session_id=self.session_id,source=source))
        destination=self.inbox/f'round-{self.index:02d}.json'
        self._send(request,str(destination)+'.part')
        self._ssh('mv '+shlex.quote(str(destination)+'.part')+' '+shlex.quote(str(destination)))
        if self.arena:
            from .arena_session import session_limits
            timeout=session_limits(self.recipe)['worker_timeout_s']+50
        else:timeout=120
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            result=self._json(self.remote/'session'/f'round-{self.index:02d}'/'result.json')
            if result:
                write_json(folder/'result.json',result)
                self.index+=1
                return result
            if self.process.poll() is not None:
                raise RuntimeError(f'session ended before round result: {self.status()}')
            time.sleep(.3)
        raise TimeoutError(f'round result exceeded {timeout} seconds')

    def visual_snapshot(self,directory):
        """Fetch one owner-published pair. Local paths never reach the remote worker."""
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        started=time.monotonic();deadline=started+30
        def remaining():
            seconds=deadline-time.monotonic()
            if seconds<=0:raise TimeoutError('camera snapshot exceeded 30 seconds')
            return seconds
        identity=uuid.uuid4().hex
        request=directory/(identity+'.request.json')
        write_json(request,dict(session_id=self.session_id,snapshot_id=identity))
        inbox=self.remote/'session/visual-inbox'
        destination=self.remote/'session/visual-snapshots'/identity
        self._ssh('mkdir -p '+shlex.quote(str(inbox)),timeout=remaining())
        self._send(request,str(inbox/(identity+'.json.part')),timeout=remaining())
        self._ssh('mv '+shlex.quote(str(inbox/(identity+'.json.part')))+' '+shlex.quote(str(inbox/(identity+'.json'))),timeout=remaining())
        while time.monotonic()<deadline:
            error=self._json(destination/'error.json',timeout=remaining())
            if error:raise RuntimeError('camera snapshot failed: '+error['error'])
            packet=self._json(destination/'manifest.json',timeout=remaining())
            if packet:
                subprocess.run(['scp','-q','-r',f'{self.host}:{destination}',str(directory)],check=True,
                               timeout=remaining())
                for frame in packet['frames']:frame['file']=identity+'/'+frame['file']
                validate_snapshot(packet,directory,self.session_id)
                write_json(directory/(identity+'.received.json'),dict(packet=packet,transfer_wall_s=time.monotonic()-started))
                return packet
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError('session ended before camera snapshot')
            time.sleep(.1)
        raise TimeoutError('camera snapshot exceeded 30 seconds')

    def _collect_artifacts(self):
        """Stream one compressed archive; collection never reruns physics.

        Many small RGB/depth files exceeded the old 90-second SCP deadline.
        Keep transfer bounded separately and retain even a partial archive on
        failure. The data filter prevents paths/links escaping the output.
        Per-step camera frames (physics/sensors, ~5 GB per trial) stay on the
        remote host: offline analysis uses the logs and videos, and the frames
        plus a kept archive filled ~700 GB locally. The archive is deleted once
        extracted, since the extracted files are the same data.
        """
        archive=self.output/'evidence.tar.gz'
        command='tar -czf - -C '+shlex.quote(str(self.remote))+' --exclude=./physics/sensors .'
        with archive.open('wb') as stream:
            subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
                            self.host,command],stdout=stream,check=True,timeout=600)
        with tarfile.open(archive,'r:gz') as bundle:
            bundle.extractall(self.output/'artifacts',filter='data')
        archive.unlink()

    def _request_shutdown(self):
        """Local SSH exit is not evidence that remote owned children stopped."""
        if self.process is None and self.session_id is None:return
        self._ssh('mkdir -p '+shlex.quote(str(self.inbox))+' && touch '+shlex.quote(str(self.inbox/'stop')))
        if self.process is not None and self.process.poll() is None:
            try:self.process.wait(timeout=100)
            except subprocess.TimeoutExpired:
                self.process.terminate();self.process.wait(timeout=10)
        if self.arena:
            deadline=time.monotonic()+100
            while time.monotonic()<deadline:
                cleanup=self._json(self.remote/'cleanup.json',timeout=min(20,max(.1,deadline-time.monotonic())))
                if cleanup and cleanup.get('owned_groups_stopped') is True:return
                time.sleep(.5)
            raise TimeoutError('remote owned-process cleanup was not confirmed')

    def close(self):
        try:self._request_shutdown()
        except Exception as error:
            write_json(self.output/'shutdown_error.json',dict(error=repr(error),remote=str(self.remote),
                collection_skipped=True,physics_state_unverified=True))
            raise
        finally:
            if self.log is not None:self.log.close()
        if self.recipe.get('task') in ('workstation_reach','ordered_reach','mobility'):
            command = [str(self.root/'.venv-sonic/bin/python'), '-u', '-m',
                       'g1cap.session_video', str(self.remote)]
            try:
                print('Rendering setup, full task and task overview on CPU...', flush=True)
                # Longer benchmarks produce setup plus two complete task
                # views. Keep export bounded, separately from physics time.
                render_timeout=max(600,6*float(self.recipe.get('deadline',180)))
                overlay=getattr(self,'source_overlay',None)
                # Python -m puts the current directory ahead of PYTHONPATH.
                # Start outside the installed source tree when using an overlay.
                shell='cd '+shlex.quote('/tmp' if overlay is not None else str(self.root))+' && '
                if overlay is not None:
                    shell+='PYTHONPATH='+shlex.quote(str(overlay)+':'+str(self.root))+' '
                self._ssh(shell+shlex.join(command), timeout=render_timeout)
            except Exception as error:
                # Rendering must never discard the source/physics evidence.
                write_json(self.output/'video_error.json', dict(error=str(error)))
                print('Video export failed; recorded evidence will still be copied:', error, flush=True)
        self._collect_artifacts()
        if self.arena:
            try:
                from .arena_video import render_session
                print('Rendering complete Arena recording on CPU...',flush=True)
                render_session(self.output/'artifacts',timeout=max(600,6*float(self.recipe.get('deadline',180))))
            except Exception as error:
                write_json(self.output/'video_error.json',dict(error=str(error)))
                print('Arena video export failed; raw recordings retained:',error,flush=True)


def compact_observation(status):
    raw=status['observation']
    if status['task'].get('backend')=='arena':
        return dict(raw)
    keys=('frame','session_id','elapsed','remaining','episode_status','approached','task_progress','hand_posture','arm_posture',
          'pelvis_position','pelvis_yaw','planar_velocity','yaw_rate','tilt',
          'left_wrist_position','left_wrist_quaternion_wxyz','left_wrist_velocity_world',
          'right_wrist_position','right_wrist_quaternion_wxyz','right_wrist_velocity_world','foot_normal_forces','state_age_s','settled_for')
    return {k:raw[k] for k in keys if k in raw}


def generation_request(status,previous_source,feedback,*,visual_observations=None):
    task=status['task']
    arena=task.get('backend')=='arena'
    mode=task.get('observation_mode')
    document='arena_v2.md' if mode=='sensor_state_v2' else 'arena_sensor.md' if mode=='sensor_estimates_v1' else 'arena_box.md'
    api=(Path(__file__).parent/'tool_docs'/document).read_text() if arena else session_api(task)
    request=dict(protocol='persistent_session',
                instruction=task.get('instruction','Approach the fixed location and reach the wrist target at the requested height.'),
                backend='arena' if arena else 'sonic',task=task,scene=status.get('scene'),observation=compact_observation(status),
                previous_source=previous_source,feedback=deepcopy(feedback),api=api)
    if visual_observations:request['visual_observations']=deepcopy(visual_observations)
    return request


def agent_feedback(result,status):
    """Compact v2 tool outcomes; leave older track feedback contracts intact."""
    execution={**result['execution'],'stderr':result['execution'].get('stderr','')[-2500:]}
    if status.get('task',{}).get('observation_mode')=='sensor_state_v2':
        tools=[]
        for call in result.get('tools',[]):
            outcome=call['result']
            item={'method':call['method'],'status':outcome.get('status'),
                  'reason':outcome.get('reason'),'advice':deepcopy(outcome.get('advice',{}))}
            if 'args' in call:item['args']=deepcopy(call['args'])
            tools.append(item)
        feedback=dict(round=result['index'],execution=execution,tools=tools)
    else:
        feedback=dict(round=result['index'],execution=execution,
                      tools=deepcopy(result.get('tools',[])),current_observation=compact_observation(status))
    for key in ('start_observation','end_observation'):
        if key in result:feedback[key]=deepcopy(result[key])
    return feedback


def run_interactive(remote,output,*,agent=None,policies=(),between_rounds=0.):
    output=Path(output)
    report=dict(protocol='persistent_session',model_sessions=0,rounds=[],generations=[],error=None)
    feedback=[]
    previous_source=None
    previous_visual=None
    try:
        status=remote.start()
        print('Session started:',status['session_id'],flush=True)
        for index in range(remote.max_rounds):
            status=remote.status()
            if status['terminal_reason']: break
            if policies:
                if index>=len(policies): break
                source=Path(policies[index]).read_text()
            else:
                folder=output/f'generation-{index:02d}'
                visual=[]
                if getattr(agent,'vision_mode','off')!='off':
                    current=remote.visual_snapshot(output/'visual-input')
                    status={**status,'observation':current['observation']}
                    visual=([previous_visual] if previous_visual else [])+[current]
                    previous_visual=current
                request=generation_request(status,previous_source,feedback,visual_observations=visual)
                if visual:request['visual_root']=str((output/'visual-input').resolve())
                report['model_sessions']+=1
                try:
                    source=agent.generate(request,folder)
                finally:
                    metadata=folder/'evidence/generation.json'
                    if metadata.exists(): report['generations'].append(json.loads(metadata.read_text()))
                submission_status=remote.status()
                folder.mkdir(parents=True,exist_ok=True)
                write_json(folder/'pre-submission-status.json',submission_status)
                if submission_status['terminal_reason']: break
            previous_source=source
            result=remote.execute(source)
            status=remote.status()
            report['rounds'].append(result)
            feedback.append(agent_feedback(result,status))
            print('Round',index,result['execution']['status'],status['metrics'],status['terminal_reason'],flush=True)
            if status['terminal_reason']: break
            # The evaluator ends a finished episode; the agent is told nothing beyond
            # the episode ending, and no further program is generated.
            if (status.get('metrics') or {}).get('success'):
                print('Task complete by independent scoring; ending episode.',flush=True)
                break
            if between_rounds: time.sleep(between_rounds)
        report['final_status']=remote.status()
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        print(report['error'],flush=True)
    finally:
        try: remote.close()
        except Exception as error: report['cleanup_error']=f'{type(error).__name__}: {error}'
        summary=remote.output/'artifacts/session/summary.json'
        if summary.exists(): report['final_status']=json.loads(summary.read_text())
        write_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-host',required=True)
    parser.add_argument('--remote-root',required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--recipe',type=Path)
    parser.add_argument('--max-rounds',type=int,default=4)
    parser.add_argument('--generation-timeout',type=float,default=120)
    parser.add_argument('--agent-model',default='gpt-5.6-luna')
    parser.add_argument('--agent-reasoning',default='low')
    parser.add_argument('--vision-mode',choices=('off','direct','structured'),default='off')
    parser.add_argument('--policy',type=Path,action='append',default=[])
    parser.add_argument('--between-rounds',type=float,default=0.)
    parser.add_argument('--port',type=int,default=15576,help='remote model-server port (distinct per concurrent session)')
    args=parser.parse_args()
    if not 1<=args.max_rounds<=4: parser.error('max-rounds must be in [1,4]')
    recipe=json.loads(args.recipe.read_text()) if args.recipe else {}
    if args.vision_mode!='off' and (recipe.get('backend')!='arena' or args.policy):
        parser.error('vision requires an Arena coding-agent session')
    try:
        agent=WorkspaceCodexAgent(timeout=args.generation_timeout,model=args.agent_model,
                                  reasoning=args.agent_reasoning,vision_mode=args.vision_mode) if not args.policy else None
    except ValueError as error:parser.error(str(error))
    args.out.mkdir(parents=True,exist_ok=False)
    remote=RemoteSession(args.ssh_host,args.remote_root,args.gpu,args.out/'remote',recipe=recipe,max_rounds=args.max_rounds,port=args.port)
    report=run_interactive(remote,args.out,agent=agent,
                           policies=args.policy,between_rounds=args.between_rounds)
    return 1 if report.get('error') or report.get('cleanup_error') else 0


if __name__=='__main__':
    raise SystemExit(main())
