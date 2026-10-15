from __future__ import annotations
from intent_bench.sides import Side
import importlib
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, sequence, heads, dimension) = (1, 2048, 32, 64)
    (groups, state_dimension, chunk) = (8, 128, 256)
    chunks = sequence // chunk
    x = torch.randn((batch, sequence, heads, dimension), device='cuda', dtype=torch.bfloat16)
    state_matrix = torch.randn((batch, sequence, groups, state_dimension), device='cuda', dtype=torch.bfloat16)
    cb = torch.randn((batch, chunks, groups, chunk, chunk), device='cuda', dtype=torch.bfloat16) * 0.01
    dt = torch.rand((batch, heads, chunks, chunk), device='cuda', dtype=torch.float32) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    previous = torch.randn((batch, chunks, heads, dimension, state_dimension), device='cuda', dtype=torch.float32) * 0.01
    residual = torch.randn((heads,), device='cuda', dtype=torch.float32) * 0.01
    _activate(context, 'corpus/gpu/baselines/triton/state-spaces-mamba/mamba_ssm/ops/triton/ssd_chunk_scan_runtime.py', 'intent_v2_triton_chunk_scan_runtime')
    source_function = importlib.import_module('mamba_ssm.ops.triton.ssd_chunk_scan')._chunk_scan_fwd
    source_base = functional_launch(lambda : source_function(cb, x, dt, decay, state_matrix, previous, D=residual))
    source = PreparedLaunch(launch=source_base.launch, outputs=lambda : source_base.outputs()[0])
    return Side(source, Tolerance(atol=0.1, rtol=0.05), cuda_graph=True, owners=locals())

def _activate(context, runtime_path, name):
    return load_module(context.project_root / runtime_path, name)
