from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.indexing.relations import scaled_index_add_unique
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (destination_rows, routed_rows, hidden) = (65536, 32768, 4096)
    initial = torch.randn((destination_rows, 1, hidden), device='cuda', dtype=torch.float16)
    routed = torch.randn((routed_rows, 1, hidden), device='cuda', dtype=torch.float16)
    indices = torch.arange(0, routed_rows * 2, 2, device='cuda', dtype=torch.int64)
    scaling = torch.randn((hidden,), device='cuda', dtype=torch.float16)
    generated_destination = initial.clone()
    arguments = (generated_destination, indices, routed, scaling, 1.0)
    (_, generated_base) = compile_single(context, scaled_index_add_unique, arguments)
    generated = PreparedCall(launch=generated_base.launch, result=lambda : generated_destination, reset=lambda : generated_destination.copy_(initial))
    return checked_call(generated, Tolerance(atol=0.02, rtol=0.01), cuda_graph=False, owners=locals())
