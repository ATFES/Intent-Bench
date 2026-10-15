from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch
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
    page_table = page_indices.view(batch, pages_per_sequence)
    lengths = torch.full((batch,), sequence, device='cuda', dtype=torch.int32)
    scale = dimension ** (-0.5)
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/vllm/attention/paged_decode/paged_gqa_decode_runtime.py', 'intent_v2_triton_paged_gqa_runtime')
    source_module = runtime.RUNTIME.load_source(context.project_root / 'corpus/gpu/baselines/triton/vllm/attention/paged_decode/triton_decode_attention.py', 'intent_v2_triton_paged_gqa_source')
    source_output = torch.empty_like(q)
    source_lse = torch.empty((batch, query_heads), device='cuda', dtype=torch.float32)
    workspace = torch.empty((batch, query_heads, splits, dimension + 1), device='cuda', dtype=torch.float32)

    def launch():
        source_module.decode_attention_fwd(q, key_cache, value_cache, source_output, source_lse, page_table, lengths, workspace, splits, scale, page_size=page_size)
    launch()
    source = PreparedLaunch(launch=launch, outputs=lambda : source_output)
    return Side(source, Tolerance(atol=0.05, rtol=0.05), cuda_graph=False, owners=locals())
