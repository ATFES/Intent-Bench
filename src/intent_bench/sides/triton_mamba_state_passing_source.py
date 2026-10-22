from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import importlib
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, chunks, heads, state) = (1, 8, 32, 8192)
    chunk_states = torch.randn((batch, chunks, heads, state), device='cuda', dtype=torch.float32) * 0.01
    decay = -torch.rand((batch, heads, chunks), device='cuda', dtype=torch.float32) * 0.1
    initial = torch.zeros((batch, heads, state), device='cuda', dtype=torch.float32)
    _activate(context, 'corpus/gpu/baselines/triton/state-spaces-mamba/mamba_ssm/ops/triton/ssd_state_passing_runtime.py', 'intent_v2_triton_state_passing_runtime')
    source_function = importlib.import_module('mamba_ssm.ops.triton.ssd_state_passing')._state_passing_fwd
    source = functional_launch(lambda : source_function(chunk_states, decay, initial))
    return checked_call(source, Tolerance(atol=1e-05, rtol=1e-05), cuda_graph=True, owners=locals())

def _activate(context, runtime_path, name):
    return load_module(context.project_root / runtime_path, name)
