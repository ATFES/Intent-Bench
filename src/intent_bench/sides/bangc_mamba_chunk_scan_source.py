from __future__ import annotations
from intent_bench.sides import Side
import importlib
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import reference_launch

def prepare(context):
    x = torch.randn((1, 2048, 32, 64), device='cuda', dtype=torch.bfloat16)
    matrix = torch.randn((1, 2048, 8, 128), device='cuda', dtype=torch.bfloat16)
    cb = torch.randn((1, 8, 8, 256, 256), device='cuda', dtype=torch.bfloat16) * 0.01
    dt = torch.rand((1, 32, 8, 256), device='cuda', dtype=torch.float32) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    previous = torch.randn((1, 8, 32, 64, 128), device='cuda', dtype=torch.float32) * 0.01
    residual = torch.randn((32,), device='cuda', dtype=torch.float32) * 0.01
    source = _source(context, 'ssd_chunk_scan', '_chunk_scan_fwd')
    return Side(reference_launch(lambda : source(cb, x, dt, decay, matrix, previous, D=residual)[0]), Tolerance(atol=0.1, rtol=0.05), device_type='cuda', owners=locals())

def _source(context, module, function):
    load_module(context.project_root / ('corpus/gpu/baselines/triton/state-spaces-mamba/mamba_ssm/ops/triton/' + module + '_runtime.py'), 'intent_bangc_' + module)
    return getattr(importlib.import_module('mamba_ssm.ops.triton.' + module), function)
