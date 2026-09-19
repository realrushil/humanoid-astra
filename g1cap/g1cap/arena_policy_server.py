"""Seed the separate native policy process without editing upstream code."""
import json
import os
import random
import runpy

import numpy as np
import torch

seed=int(os.environ['GROOT_SEED'])
assert 0<=seed<2**32
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark=False
print('GROOT_RANDOMNESS '+json.dumps({'seed':seed,'python_numpy_torch_seeded':True,
    'cudnn_benchmark':False,'bitwise_determinism_verified':False}),flush=True)
runpy.run_module('gr00t.eval.run_gr00t_server',run_name='__main__')
