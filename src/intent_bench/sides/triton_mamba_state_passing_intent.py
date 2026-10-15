from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.streaming.mamba import mamba_state_passing_fwd
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, chunks, heads, state) = (1, 8, 32, 8192)
    chunk_states = torch.randn((batch, chunks, heads, state), device='cuda', dtype=torch.float32) * 0.01
    decay = -torch.rand((batch, heads, chunks), device='cuda', dtype=torch.float32) * 0.1
    initial = torch.zeros((batch, heads, state), device='cuda', dtype=torch.float32)
    (_, generated) = compile_single(context, mamba_state_passing_fwd, (chunk_states, decay, initial))
    return Side(generated, Tolerance(atol=1e-05, rtol=1e-05), cuda_graph=True, owners=locals())
