from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.indexing.relations import scaled_index_add_unique
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (destination_rows, routed_rows, hidden) = (65536, 32768, 4096)
    initial = torch.randn((destination_rows, 1, hidden), device='cuda', dtype=torch.float16)
    routed = torch.randn((routed_rows, 1, hidden), device='cuda', dtype=torch.float16)
    indices = torch.arange(0, routed_rows * 2, 2, device='cuda', dtype=torch.int64)
    scaling = torch.randn((hidden,), device='cuda', dtype=torch.float16)
    generated_destination = initial.clone()
    arguments = (generated_destination, indices, routed, scaling, 1.0)
    (_, generated_base) = compile_single(context, scaled_index_add_unique, arguments)
    generated = PreparedLaunch(launch=generated_base.launch, outputs=lambda : generated_destination, prepare=lambda : generated_destination.copy_(initial))
    return Side(generated, Tolerance(atol=0.02, rtol=0.01), cuda_graph=False, owners=locals())
