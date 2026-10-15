from __future__ import annotations
from intent_bench.sides import Side
import importlib
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, sequence, heads, dimension) = (1, 2048, 32, 64)
    (groups, state_dimension, chunk) = (8, 128, 256)
    chunks = sequence // chunk
    x = torch.randn((batch, sequence, heads, dimension), device='cuda', dtype=torch.bfloat16)
    state_basis = torch.randn((batch, sequence, groups, state_dimension), device='cuda', dtype=torch.bfloat16)
    dt = torch.rand((batch, heads, chunks, chunk), device='cuda', dtype=torch.float32) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    _activate(context, 'corpus/gpu/baselines/triton/state-spaces-mamba/mamba_ssm/ops/triton/ssd_chunk_state_runtime.py', 'intent_v2_triton_chunk_state_runtime')
    source_function = importlib.import_module('mamba_ssm.ops.triton.ssd_chunk_state')._chunk_state_fwd
    source = functional_launch(lambda : source_function(state_basis, x, dt, decay, states_in_fp32=True))
    return Side(source, Tolerance(atol=0.02, rtol=0.02), cuda_graph=True, owners=locals())

def _activate(context, runtime_path, name):
    return load_module(context.project_root / runtime_path, name)
