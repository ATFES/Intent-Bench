from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, query_heads, kv_heads, dimension) = (16, 32, 8, 128)
    (sequence, page_size, splits) = (8192, 16, 8)
    pages_per_sequence = sequence // page_size
    pages = batch * pages_per_sequence
    q = torch.randn((batch, query_heads, dimension), device='cuda', dtype=torch.float16)
    key_cache = torch.randn((pages, page_size, kv_heads, dimension), device='cuda', dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    page_indices = torch.arange(pages, device='cuda', dtype=torch.int32)
    page_offsets = torch.arange(0, pages + 1, pages_per_sequence, device='cuda', dtype=torch.int32)
    lengths = torch.full((batch,), sequence, device='cuda', dtype=torch.int32)
    scale = dimension ** (-0.5)
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/xformers/attention/splitk/splitk_kernels_runtime.py', 'intent_v2_triton_xformers_splitk_runtime')
    arguments = (q, key_cache, value_cache, page_offsets, page_indices, lengths, scale)
    source = functional_launch(lambda : runtime.upstream(arguments))
    return Side(source, Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, owners=locals())
