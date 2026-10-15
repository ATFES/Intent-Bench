from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (source_rows, selected_rows, hidden) = (65536, 32768, 4096)
    source_tensor = torch.randn((source_rows, hidden), device='cuda', dtype=torch.float16)
    indices = torch.arange(0, selected_rows * 2, 2, device='cuda', dtype=torch.int64)
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/xformers/indexing/index_select_cat/k_index_select_cat_runtime.py', 'intent_v2_triton_index_select')
    source_module = runtime.load_source()
    source_output = torch.empty((selected_rows, hidden), device='cuda', dtype=torch.float16)

    def launch():
        source_module.index_select_cat_fwd(source_output, source_tensor, indices)
    launch()
    source = PreparedLaunch(launch=launch, outputs=lambda : source_output)
    return Side(source, Tolerance(atol=0.0), cuda_graph=True, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
