from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.references.torch_ops import histogram
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    samples = torch.randint(0, 256, (8 * 1024 * 1024,), device='cuda', dtype=torch.int32).to(torch.float32)
    source_histogram = torch.zeros((256,), device=samples.device, dtype=torch.int32)

    def source_launch():
        histogram(samples, source_histogram)
    source_launch()
    source = PreparedCall(launch=source_launch, result=lambda : source_histogram, reset=source_histogram.zero_)
    return checked_call(source, Tolerance(atol=0.0), cuda_graph=False, owners=locals())
