"""Restricted child entry point. Only the parent owns robot state and scoring."""
import json
import math
import resource
import sys
import traceback


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    if sys.platform == 'linux':
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    incoming, outgoing = sys.stdin, sys.stdout
    sys.stdout = sys.stderr
    setup = json.loads(incoming.readline())
    if sys.platform == 'linux':
        limit = setup['memory_limit_mb'] * 1024**2
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    sequence = 0

    def emit(message):
        outgoing.write(json.dumps(message, allow_nan=False) + '\n')
        outgoing.flush()

    class RobotClient:
        def __getattr__(self, method):
            if method not in setup['tools']:
                raise AttributeError(method)
            def call(*args, **kwargs):
                nonlocal sequence
                sequence += 1
                emit({'type': 'call', 'id': sequence, 'episode_id': setup['episode_id'],
                      'method': method, 'args': args, 'kwargs': kwargs})
                reply = json.loads(incoming.readline())
                if reply.get('id') != sequence:
                    raise RuntimeError('RPC response sequence mismatch')
                if 'error' in reply:
                    raise RuntimeError(reply['error'])
                return reply['result']
            return call

    try:
        namespace = {'__name__': 'candidate_policy', 'math': math}
        exec(compile(setup['source'], 'policy.py', 'exec'), namespace)
        namespace['run'](RobotClient(), setup['task'])
        emit({'type': 'done'})
    except MemoryError:
        emit({'type': 'error', 'reason': 'memory_limit', 'error': 'Python allocation exceeded worker memory budget'})
        return 1
    except BaseException:
        emit({'type': 'error', 'error': traceback.format_exc(limit=8)[-8000:]})
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
