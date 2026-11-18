from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import torch
from intent.examples.programs.structured import VarlenCausalConvolution
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance

def prepare(context):
    lengths = (2048, 1536, 1024, 512)
    offsets = torch.tensor((0, 2048, 3584, 4608, 5120), device='cuda', dtype=torch.int64)
    (hidden, width, chunk_size) = (4096, 4, 64)
    x = torch.randn((sum(lengths), hidden), device='cuda', dtype=torch.bfloat16)
    weight = torch.randn((hidden, width), device='cuda', dtype=torch.float32)
    bias = torch.randn((hidden,), device='cuda', dtype=torch.float32)
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/fla/conv/causal1d/causal_conv_varlen_runtime.py', 'intent_v2_triton_varlen_causal_conv_runtime')
    chunk_indices = runtime.RUNTIME._prepare_chunk_indices(offsets, chunk_size)
    program = VarlenCausalConvolution.from_jit(context.compile)
    generated = intent_call(program, (x, offsets, chunk_indices, weight, bias), device='cuda')
    return checked_call(generated, (Tolerance(atol=0.05, rtol=0.05), Tolerance(atol=0.0)), cuda_graph=False, owners=locals())
