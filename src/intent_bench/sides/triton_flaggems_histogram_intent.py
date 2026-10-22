from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.statistics.histogram import histogram_256
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    samples = torch.randint(0, 256, (8 * 1024 * 1024,), device='cuda', dtype=torch.int32).to(torch.float32)
    (_, generated_base) = compile_single(context, histogram_256, (samples,))
    generated = PreparedCall(launch=generated_base.launch, result=generated_base.result, reset=generated_base.result().zero_)
    return checked_call(generated, Tolerance(atol=0.0), cuda_graph=False, owners=locals())
