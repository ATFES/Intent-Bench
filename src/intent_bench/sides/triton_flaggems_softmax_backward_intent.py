from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.backward.softmax import softmax_backward as softmax_backward_kernel
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    probabilities = torch.randn((4096, 4097), device='cuda', dtype=torch.float32)
    gradient = torch.randn_like(probabilities)
    (_, generated) = compile_single(context, softmax_backward_kernel, (probabilities, gradient))
    return checked_call(generated, Tolerance(atol=0.0001, rtol=1e-05), cuda_graph=True, owners=locals())
