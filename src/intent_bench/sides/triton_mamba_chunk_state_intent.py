from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.streaming.mamba import mamba_chunk_state_bf16_fwd
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, sequence, heads, dimension) = (1, 2048, 32, 64)
    (groups, state_dimension, chunk) = (8, 128, 256)
    chunks = sequence // chunk
    x = torch.randn((batch, sequence, heads, dimension), device='cuda', dtype=torch.bfloat16)
    state_basis = torch.randn((batch, sequence, groups, state_dimension), device='cuda', dtype=torch.bfloat16)
    dt = torch.rand((batch, heads, chunks, chunk), device='cuda', dtype=torch.float32) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    (_, generated) = compile_single(context, mamba_chunk_state_bf16_fwd, (state_basis, x, dt, decay), constexprs={'HEAD_GROUP': heads // groups})
    return checked_call(generated, Tolerance(atol=0.02, rtol=0.02), cuda_graph=True, owners=locals())
