from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.activation.pointwise import geglu_tanh
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((4096, 28672), device='cuda', dtype=torch.float16)
    output = torch.empty((4096, 14336), device='cuda', dtype=torch.float16)
    (_, generated_base) = compile_single(context, geglu_tanh, (x, output))
    generated = PreparedCall(generated_base.launch, lambda : output)
    return checked_call(generated, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, note='同算法；source 保留 f16 中间舍入', owners=locals())
