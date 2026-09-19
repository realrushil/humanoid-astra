"""Run policy code across an OS-enforced boundary, with bounded JSON RPC."""
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time

TOOLS = frozenset({'observe', 'observe_scene', 'check_reach', 'check_hands', 'reach_hands', 'move_base', 'stop', 'hold', 'walk_to', 'turn_to', 'reach_right', 'set_posture', 'sonic_motion'})


def sandbox_command():
    python = str(Path(sys.executable).resolve())
    worker = str(Path(__file__).with_name('worker.py').resolve())
    if sys.platform == 'darwin':
        if not Path('/usr/bin/sandbox-exec').exists():
            raise RuntimeError('macOS sandbox-exec is unavailable; refusing unsandboxed execution')
        roots = {str(Path(sys.base_prefix).resolve()), str(Path(python).parent.parent),
                 '/System', '/usr/lib', '/Library/Apple/System/Library'}
        reads = ' '.join('(subpath ' + json.dumps(path) + ')' for path in sorted(roots))
        profile = ('(version 1)(deny default)(allow sysctl-read)(allow mach-lookup)'
                   '(allow process-info*)(allow signal (target self))'
                   '(allow process-exec (literal ' + json.dumps(python) + ') (subpath ' + json.dumps(str(Path(sys.base_prefix).resolve())) + '))'
                   '(allow file-read-metadata)(allow file-read-data (literal "/"))'
                   '(allow file-read* ' + reads + ' (literal ' + json.dumps(worker) + ')'
                   ' (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random"))')
        return ['/usr/bin/sandbox-exec', '-p', profile, python, '-I', '-S', '-u', worker], 'macos-seatbelt'
    if sys.platform == 'linux':
        bwrap = shutil.which('bwrap')
        if not bwrap:
            raise RuntimeError('Linux bubblewrap is unavailable; refusing unsandboxed execution')
        cmd = [bwrap, '--die-with-parent', '--unshare-all', '--new-session', '--cap-drop', 'ALL']
        roots = {'/usr', '/bin', '/lib', '/lib64', str(Path(sys.base_prefix).resolve())}
        # Only the Python environment and operating-system libraries are readable.
        selected = []
        for root in sorted(roots, key=len):
            if Path(root).exists() and not any(root == p or root.startswith(p + '/') for p in selected):
                selected.append(root)
                cmd += ['--ro-bind', root, root]
        cmd += ['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
                '--ro-bind', worker, '/worker.py', '--chdir', '/tmp', python, '-I', '-S', '-u', '/worker.py']
        return cmd, 'linux-bubblewrap'
    raise RuntimeError('No verified policy sandbox for this platform')


def execute_policy(source, task, episode_id, dispatch, *, wall_timeout=10.0,
                   max_requests=1000, output_limit=1_000_000, memory_limit_mb=256,
                   tools=TOOLS, wall_deadline=None):
    """Dispatch must return promptly; real controller I/O belongs in a separate backend.

    Child completion only means Python returned, never that a robot task succeeded.
    Unsupported or failed sandbox startup is reported without an unsafe fallback.
    wall_deadline optionally supplies a session's absolute monotonic deadline in
    seconds, including trusted sandbox setup; it never extends wall_timeout.
    """
    if not isinstance(source, str) or len(source.encode()) > 65536:
        raise ValueError('policy source must be at most 65536 UTF-8 bytes')
    if not math.isfinite(wall_timeout) or wall_timeout <= 0 or max_requests < 1 or output_limit < 1:
        raise ValueError('execution budgets must be positive and finite')
    if wall_deadline is not None and not math.isfinite(wall_deadline):
        raise ValueError('wall deadline must be finite monotonic seconds')
    if not isinstance(memory_limit_mb, int) or memory_limit_mb < 32:
        raise ValueError('memory limit must be an integer of at least 32 MiB')
    # The trusted session advertises only its own implemented operations. The
    # default remains the existing SONIC API; other controllers opt in explicitly.
    if isinstance(tools,(str,bytes)):
        raise ValueError('tools must be a collection of public method names')
    allowed=frozenset(tools)
    if not allowed or any(not isinstance(name,str) or not name.isidentifier() or name.startswith('_') for name in allowed):
        raise ValueError('tools must contain public method names')
    result = {'status': 'sandbox_error', 'error': '', 'stderr': '', 'requests': 0,
              'sandbox': None, 'returncode': None, 'memory_limit_mb': memory_limit_mb,
              'memory_enforcement': 'rss_poll_50ms' if sys.platform == 'darwin' else 'rlimit_as'}
    if wall_deadline is not None and time.monotonic() >= wall_deadline:
        result.update(status='wall_timeout', wall_time=0.)
        return result
    try:
        cmd, result['sandbox'] = sandbox_command()
    except RuntimeError as error:
        result['error'] = str(error)
        return result
    start = time.monotonic()
    # Preserve standalone relative-budget callers; a session's deadline also
    # bounds setup before this timestamp and cannot be renewed here.
    deadline = start + wall_timeout
    if wall_deadline is not None:
        deadline = min(deadline, wall_deadline)
    stderr = bytearray()
    pending = bytearray()
    total = 0
    last_memory_check = 0.0
    terminal_message = None
    proc = None
    try:
        if time.monotonic() >= deadline:
            result['status'] = 'wall_timeout'
            return result
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1'},
                                cwd='/tmp', start_new_session=True, bufsize=0)
        payload = json.dumps({'source': source, 'task': task, 'tools': sorted(allowed), 'episode_id': episode_id, 'memory_limit_mb': memory_limit_mb}, allow_nan=False).encode() + b'\n'
        # The trusted child reads setup before executing any user source.
        view = memoryview(payload)
        while view:
            written = proc.stdin.write(view)
            if not written:
                raise BrokenPipeError('worker did not accept setup')
            view = view[written:]
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ, 'protocol')
            selector.register(proc.stderr, selectors.EVENT_READ, 'stderr')
            while selector.get_map() or proc.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    result['status'] = 'wall_timeout'
                    break
                if sys.platform == 'darwin' and time.monotonic() - last_memory_check >= .05:
                    # Aggregate the child's process group; this is a polling guard,
                    # not an instantaneous OS address-space limit on macOS.
                    snapshot = subprocess.run(['/bin/ps', '-axo', 'pgid=,rss='],
                                              capture_output=True, text=True, timeout=min(remaining, 1.0))
                    if snapshot.returncode:
                        result.update(status='sandbox_error', error='memory watchdog unavailable')
                        break
                    resident_kib = 0
                    for row in snapshot.stdout.splitlines():
                        fields = row.split()
                        if len(fields) == 2 and fields[0] == str(proc.pid):
                            resident_kib += int(fields[1])
                    last_memory_check = time.monotonic()
                    if resident_kib > memory_limit_mb * 1024:
                        result['status'] = 'memory_limit'
                        break
                for key, _ in selector.select(min(remaining, 0.05)):
                    if time.monotonic() >= deadline:
                        result['status'] = 'wall_timeout'
                        break
                    data = os.read(key.fileobj.fileno(), 8192)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    total += len(data)
                    if key.data == 'stderr':
                        stderr.extend(data[:max(0, output_limit - len(stderr))])
                    if total > output_limit:
                        result['status'] = 'output_limit'
                        break
                    if key.data != 'protocol':
                        continue
                    pending.extend(data)
                    if len(pending) > 65536:
                        result['status'] = 'protocol_error'
                        result['error'] = 'protocol line exceeded 65536 bytes'
                        break
                    while b'\n' in pending:
                        line, _, rest = pending.partition(b'\n')
                        pending[:] = rest
                        try:
                            msg = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                            if not isinstance(msg, dict) or terminal_message is not None:
                                raise ValueError('unexpected worker message')
                            if msg.get('type') in {'done', 'error'}:
                                terminal_message = msg
                                continue
                            if msg.get('type') != 'call' or msg.get('method') not in allowed:
                                raise ValueError('invalid RPC operation')
                            if msg.get('episode_id') != episode_id or msg.get('id') != result['requests'] + 1:
                                raise ValueError('invalid RPC episode or sequence')
                            if not isinstance(msg.get('args'), list) or not isinstance(msg.get('kwargs'), dict):
                                raise ValueError('invalid RPC arguments')
                            if result['requests'] >= max_requests:
                                result['status'] = 'request_limit'
                                break
                            result['requests'] += 1
                            value = dispatch(msg['method'], msg['args'], msg['kwargs'], episode_id)
                            reply = json.dumps({'id': msg['id'], 'result': value}, allow_nan=False).encode() + b'\n'
                            if len(reply) > 65536:
                                raise ValueError('backend response too large')
                            # A conforming worker is waiting on this reply. Writes are
                            # nonblocking so forged traffic cannot defeat the watchdog.
                            os.set_blocking(proc.stdin.fileno(), False)
                            count = os.write(proc.stdin.fileno(), reply)
                            if count != len(reply):
                                raise ValueError('partial RPC response')
                        except (ValueError, TypeError, KeyError, UnicodeError) as error:
                            result['status'] = 'protocol_error'
                            result['error'] = str(error)
                            break
                    if result['status'] != 'sandbox_error':
                        break
                if result['status'] != 'sandbox_error':
                    break
        if result['status'] == 'sandbox_error':
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result['status'] = 'wall_timeout'
                return result
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                result['status'] = 'wall_timeout'
            else:
                if pending:
                    result.update(status='protocol_error', error='unterminated worker message')
                elif terminal_message and terminal_message['type'] == 'error':
                    result.update(status='memory_limit' if terminal_message.get('reason') == 'memory_limit' else 'policy_error', error=str(terminal_message.get('error', ''))[:8000])
                elif terminal_message and proc.returncode == 0:
                    result['status'] = 'completed'
                else:
                    result['error'] = 'worker exited without a complete protocol result'
    except (OSError, subprocess.SubprocessError) as error:
        result['error'] = str(error)
    except Exception as error:
        result.update(status='backend_error', error=f'{type(error).__name__}: {error}')
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            result['returncode'] = proc.returncode
            for pipe in [proc.stdin, proc.stdout, proc.stderr]:
                pipe.close()
        result['stderr'] = stderr.decode('utf-8', errors='replace')
        result['wall_time'] = round(time.monotonic() - start, 6)
    return result
