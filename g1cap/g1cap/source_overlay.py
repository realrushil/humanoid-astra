"""Immutable per-launch package snapshot, including static assets and contracts."""
import hashlib
import json
from pathlib import Path
import shutil


def package_manifest(overlay):
    overlay=Path(overlay)
    package=overlay/'g1cap'
    if not package.is_dir():raise ValueError('source manifest package missing')
    result={}
    for path in sorted(package.rglob('*')):
        if '__pycache__' in path.parts or path.suffix in ('.pyc','.pyo'):continue
        if path.is_symlink():raise ValueError('source manifest cannot contain symlinks')
        if path.is_file():
            result[path.relative_to(overlay).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def stage_overlay(destination, *, package=None):
    """Snapshot this imported package, never a similarly named directory in CWD."""
    package=Path(package) if package is not None else Path(__file__).resolve().parent
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copytree(package,destination/'g1cap',symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo'))
    manifest=package_manifest(destination)
    (destination/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def verify_overlay(overlay):
    """Refuse missing, changed or extra package files before launching physics."""
    overlay=Path(overlay)
    try:expected=json.loads((overlay/'source-manifest.json').read_text())
    except (OSError,ValueError) as error:raise ValueError('source manifest missing or invalid') from error
    actual=package_manifest(overlay)
    if not actual or actual!=expected:raise ValueError('source manifest mismatch')
    return actual


def overlay_command(python, module, overlay, record, *arguments):
    """Pin a child import before Python can consider its native working directory."""
    bootstrap=('import sys; sys.path.insert(0,sys.argv[1]); '
        'from g1cap.source_overlay import run_overlay; '
        'run_overlay(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4:])')
    return [str(python),'-c',bootstrap,str(Path(overlay).resolve()),module,str(record),*arguments]


def run_overlay(overlay, module, record, arguments):
    """Verify child source/assets and record its actual entry path before startup."""
    import importlib.util
    import runpy
    import sys
    overlay=Path(overlay).resolve()
    if Path(__file__).resolve().parent!=overlay/'g1cap':
        raise ValueError('child must import from requested overlay')
    manifest=verify_overlay(overlay)
    spec=importlib.util.find_spec(module)
    expected=overlay.joinpath(*module.split('.')).with_suffix('.py')
    if spec is None or spec.origin is None or Path(spec.origin).resolve()!=expected:
        raise ValueError('child entry point differs from requested overlay')
    Path(record).write_text(json.dumps(dict(module=module,module_path=str(expected),
        package_path=str(overlay/'g1cap'),manifest=manifest),indent=2)+'\n')
    sys.argv=[module,*arguments]
    runpy.run_module(module,run_name='__main__',alter_sys=True)
