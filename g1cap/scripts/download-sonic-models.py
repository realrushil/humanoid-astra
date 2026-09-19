"""Download only a pinned public deployment model set; no training data."""
import hashlib
import json
from pathlib import Path
import sys
from huggingface_hub import HfApi, hf_hub_download

root = Path(sys.argv[1]).resolve()
repo = root / 'deps/GR00T-WholeBodyControl'
model_id = 'nvidia/GEAR-SONIC'
revision = HfApi().model_info(model_id, token=False).sha
files = {
    'config.json': 'release-manifest.json',
    'sonic_v1_1/model_encoder.onnx': 'policy/sonic_v1_1/model_encoder.onnx',
    'sonic_v1_1/model_decoder.onnx': 'policy/sonic_v1_1/model_decoder.onnx',
    'sonic_v1_1/observation_config.yaml': 'policy/sonic_v1_1/observation_config.yaml',
    'planner_sonic.onnx': 'planner/target_vel/V2/planner_sonic.onnx',
}
record={'repo_id':model_id,'revision':revision,'variant':'sonic_v1_1','files':{}}
for filename, destination in files.items():
    downloaded=Path(hf_hub_download(model_id, filename, revision=revision, token=False,
                                   local_dir=root/'deps/models-hf'))
    output=repo/'gear_sonic_deploy'/destination
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        if output.is_symlink(): output.unlink()
        elif output.read_bytes() != downloaded.read_bytes():
            raise RuntimeError(f'refusing to replace different existing model {output}')
    if not output.exists(): output.symlink_to(downloaded)
    sha=hashlib.sha256()
    with downloaded.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''): sha.update(chunk)
    record['files'][filename]={'sha256':sha.hexdigest(),'bytes':downloaded.stat().st_size,'path':str(output)}
    print(filename,record['files'][filename]['bytes'],flush=True)
(root/'setup-logs/models.json').write_text(json.dumps(record,indent=2)+'\n')
