"""Resolve only checked-out robot/reference LFS pointers, checking each SHA256."""
import concurrent.futures
import hashlib
from pathlib import Path
import subprocess
import sys
import urllib.request

repo = Path(sys.argv[1]).resolve()
revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
roots = ['gear_sonic/data/robot_model/model_data/g1', 'gear_sonic_deploy/reference/example',
         'gear_sonic_deploy/thirdparty/unitree_sdk2/lib/x86_64',
         'gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64']
pointers = []
for root in roots:
    for path in (repo/root).rglob('*'):
        if path.is_file() and not path.is_symlink() and path.stat().st_size < 1024:
            data = path.read_bytes()
            if data.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                fields = dict(line.split(' ', 1) for line in data.decode().splitlines())
                pointers.append((path, fields['oid'].removeprefix('sha256:'), int(fields['size'])))

def download(item):
    path, expected, size = item
    relative = path.relative_to(repo).as_posix()
    url = f'https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/{revision}/{relative}'
    with urllib.request.urlopen(url, timeout=120) as response:
        content = response.read(size+1)
    if len(content) != size or hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError(f'LFS asset verification failed: {relative}')
    path.write_bytes(content)
    print(relative, size, expected, flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(download, pointers))
print(f'Verified and resolved {len(pointers)} LFS pointers.', flush=True)
