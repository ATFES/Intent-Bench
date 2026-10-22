from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.ragged.grouped_gemm import ragged_grouped_gemm_bf16
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    rows = (256, 512, 1024, 2048)
    boundaries = (0, 256, 768, 1792, 3840)
    hidden = output = 4096
    group_x = tuple((torch.randn((row, hidden), device='cuda', dtype=torch.bfloat16) for row in rows))
    group_weight = tuple((torch.randn((hidden, output), device='cuda', dtype=torch.bfloat16) for _ in rows))
    x = torch.cat(group_x, dim=0)
    weight = torch.stack(group_weight, dim=0)
    offsets = torch.tensor(boundaries, device='cuda', dtype=torch.int32)
    (_, generated) = compile_single(context, ragged_grouped_gemm_bf16, (x, offsets, weight))
    generated_tensor = generated.result()
    generated = PreparedCall(generated.launch, lambda : tuple((generated_tensor[begin:end] for (begin, end) in zip(boundaries[:-1], boundaries[1:]))))
    return checked_call(generated, Tolerance(atol=0.05, rtol=0.02), cuda_graph=False, note='同算法；source 使用 host group shapes', owners=locals())
