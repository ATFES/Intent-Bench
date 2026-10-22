from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.indexing.relations import index_select_rows
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (source_rows, selected_rows, hidden) = (65536, 32768, 4096)
    source_tensor = torch.randn((source_rows, hidden), device='cuda', dtype=torch.float16)
    indices = torch.arange(0, selected_rows * 2, 2, device='cuda', dtype=torch.int64)
    (_, generated) = compile_single(context, index_select_rows, (source_tensor, indices))
    return checked_call(generated, Tolerance(atol=0.0), cuda_graph=True, owners=locals())
