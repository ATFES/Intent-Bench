from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (experts, rows, hidden, intermediate) = (8, 1024, 4096, 14336)
    x = torch.randn((experts * rows, hidden), device='cuda', dtype=torch.float16)
    weight = torch.randn((experts, hidden, intermediate), device='cuda', dtype=torch.float16)
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/triton/gemm/grouped/08-grouped-gemm_runtime.py', 'intent_v2_triton_grouped_gemm')
    source_function = runtime.load_grouped_gemm()
    source_autotuner = source_function.__globals__['grouped_matmul_kernel']
    group_x = list(x.view(experts, rows, hidden).unbind(0))
    group_weight = list(weight.unbind(0))
    source_outputs = [torch.empty((rows, intermediate), device='cuda', dtype=torch.float16) for _ in range(experts)]
    source_a_ptrs = torch.tensor([value.data_ptr() for value in group_x], device='cuda')
    source_b_ptrs = torch.tensor([value.data_ptr() for value in group_weight], device='cuda')
    source_c_ptrs = torch.tensor([value.data_ptr() for value in source_outputs], device='cuda')
    source_sizes = torch.tensor([rows, intermediate, hidden] * experts, dtype=torch.int32, device='cuda')
    source_lds = torch.tensor([hidden, intermediate, intermediate] * experts, dtype=torch.int32, device='cuda')

    def launch():
        source_autotuner[lambda meta: (meta['NUM_SM'],)](source_a_ptrs, source_b_ptrs, source_c_ptrs, source_sizes, source_lds, experts)
    launch()
    source = PreparedLaunch(launch=launch, outputs=lambda : torch.cat(source_outputs, dim=0))
    return Side(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=False, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
