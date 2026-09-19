"""Launch and clean up only this Arena session's model and simulator groups."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import hashlib

from .arena_session import atomic_json, session_limits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=1,help='rendering GPU')
    parser.add_argument('--model-gpu',type=int,default=0)
    parser.add_argument('--port',type=int,default=15555)
    parser.add_argument('--max-rounds',type=int,default=2)
    parser.add_argument('--source-overlay',type=Path,
                        help='isolated remote package overlay for this session')
    args=parser.parse_args()
    root,out=args.root.resolve(),args.out.resolve()
    args.recipe=args.recipe.resolve()
    out.mkdir(parents=True,exist_ok=False)
    recipe=json.loads(args.recipe.read_text())
    limits=session_limits(recipe)
    arena=root/'temp/Arena'
    def snapshot(name):
        value=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        (out/name).write_text(value)
        return {int(row.split(',')[0]):int(row.split(',')[2]) for row in value.splitlines()}
    def listening():
        with socket.socket() as sock:
            sock.settimeout(1.)
            return sock.connect_ex(('127.0.0.1',args.port))==0
    if listening():raise RuntimeError('requested model port occupied; unknown server left untouched')
    free=snapshot('gpu-before.csv')
    needed={args.model_gpu:9000}
    needed[args.gpu]=needed.get(args.gpu,0)+4000
    if any(free[gpu]<amount for gpu,amount in needed.items()):raise RuntimeError('insufficient free GPU memory for bounded allocation')
    overlay=args.source_overlay.resolve() if args.source_overlay else None
    if overlay is not None:
        package=overlay/'g1cap'
        if not package.is_dir():raise ValueError('source overlay must contain g1cap package')
        manifest={}
        for path in sorted(package.rglob('*.py')):
            manifest[str(path.relative_to(overlay))]=hashlib.sha256(path.read_bytes()).hexdigest()
        (out/'source-overlay-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    pythonpath=':'.join(str(p) for p in ([overlay,root] if overlay is not None else [root,]))+':'+str(arena)
    environment=dict(os.environ,PYTHONPATH=pythonpath,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                     HF_HOME=str(root/'temp/hf-cache'),XDG_CACHE_HOME=str(root/'temp/arena-runtime-cache'),PYTHONUNBUFFERED='1')
    server_command=[str(root/'temp/gr00t-venv/bin/python'),'-m','g1cap.arena_policy_server',
        '--modality-config-path',str(arena/'isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py'),
        '--model-path',str(root/'temp/arena-model'),'--embodiment-tag','NEW_EMBODIMENT','--device','cuda','--host','127.0.0.1','--port',str(args.port)]
    sim_command=[str(root/'temp/arena-venv/bin/python'),'-m','g1cap.arena_session_runtime',
        '--session-out',str(out),'--recipe',str(args.recipe),'--max-rounds',str(args.max_rounds),'--policy-port',str(args.port),
        '--viz','none','--device','cpu','--num_envs','1','--enable_cameras',
        'galileo_g1_locomanip_pick_and_place','--object','brown_box','--embodiment','g1_wbc_joint']
    owned=[];logs=[]
    def interrupted(signum,frame):raise InterruptedError(f'launcher signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        log=(out/'server.log').open('w');logs.append(log)
        server=subprocess.Popen(server_command,cwd=arena/'submodules/Isaac-GR00T',
            env=dict(environment,CUDA_VISIBLE_DEVICES=str(args.model_gpu),GROOT_SEED=str(recipe.get('gr00t_seed',0))),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        owned.append(server)
        atomic_json(out/'owned.json',dict(server_pid=server.pid,server_command=server_command))
        deadline=time.monotonic()+240
        while not listening():
            if server.poll() is not None or time.monotonic()>deadline:raise RuntimeError('owned model startup failed')
            time.sleep(1.)
        log=(out/'runtime.log').open('w');logs.append(log)
        runtime=subprocess.Popen(sim_command,cwd=arena,env=dict(environment,CUDA_VISIBLE_DEVICES=str(args.gpu),HEADLESS='1',OMNI_KIT_ACCEPT_EULA='YES'),
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        owned.append(runtime)
        atomic_json(out/'owned.json',dict(server_pid=server.pid,server_command=server_command,runtime_pid=runtime.pid,runtime_command=sim_command))
        deadline=time.monotonic()+limits['wall_timeout_s']+100
        while runtime.poll() is None:
            snapshot('gpu-during.csv')
            (out/'allocations.csv').write_text(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory,gpu_uuid','--format=csv'],text=True))
            if server.poll() is not None or time.monotonic()>deadline:raise RuntimeError('owned session process failed or timed out')
            time.sleep(5.)
        if runtime.returncode:raise RuntimeError(f'owned runtime exited {runtime.returncode}; retain recorded evidence')
    except Exception as error:
        atomic_json(out/'launch-error.json',dict(error=repr(error)))
        raise
    finally:
        for process in reversed(owned):
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
        for log in logs:log.close()
        snapshot('gpu-after.csv')
        atomic_json(out/'cleanup.json',dict(owned_groups_stopped=True,processes=[dict(pid=p.pid,returncode=p.returncode) for p in owned]))


if __name__=='__main__':main()
