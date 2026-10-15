from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    (batch, query_heads, kv_heads, dimension) = (16, 32, 8, 128)
    (sequence, page_size, splits) = (8192, 16, 8)
    pages_per_sequence = sequence // page_size
    pages_per_split = pages_per_sequence // splits
    pages = batch * pages_per_sequence
    q = torch.randn((batch, query_heads, dimension), dtype=torch.float16)
    key_cache = torch.randn((pages, page_size, kv_heads, dimension), dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    page_indices = torch.arange(pages, dtype=torch.int32)
    page_offsets = torch.arange(0, pages + 1, pages_per_sequence, dtype=torch.int32)
    split_offsets = torch.arange(0, pages + 1, pages_per_split, dtype=torch.int32)
    lengths = torch.full((batch,), sequence, dtype=torch.int32)
    scale = dimension ** (-0.5)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_paged_gqa_decode')
    source_state = {}

    def source_launch():
        source_state['output'] = runtime.paged_gqa_decode(q, key_cache, value_cache, page_offsets, page_indices, lengths, split_offsets, scale, page_size=page_size)
    return Side(PreparedLaunch(source_launch, lambda : source_state['output']), Tolerance(atol=0.05, rtol=0.05), cuda_graph=False, device_type='cpu', note='完整 paged GQA partial+split-k pipeline；生成端输出及 partial 预分配，CPU reference 计入 page table 解码、临时分配和 attention。', owners=locals())
