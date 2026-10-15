from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.streaming.selective_scan import mamba_chunk_scan_bf16_fwd
from intent_bench.corpus._common.measurement import compile_single
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
    arguments = (cb, x, dt, decay, state_matrix, previous, residual)
    (_, generated) = compile_single(context, mamba_chunk_scan_bf16_fwd, arguments, constexprs={'HEADS_PER_GROUP': heads // groups})
    return Side(generated, Tolerance(atol=0.1, rtol=0.05), cuda_graph=True, owners=locals())
