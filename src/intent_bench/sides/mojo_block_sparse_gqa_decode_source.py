from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    (batch, query_heads, key_heads, sequence, dimension) = (8, 32, 8, 8192, 128)
    (block_size, selected_blocks, splits) = (32, 128, 4)
    q = torch.randn((batch, query_heads, dimension), dtype=torch.float16)
    k = torch.randn((batch, sequence, key_heads, dimension), dtype=torch.float16)
    v = torch.randn_like(k)
    selected = torch.arange(sequence // block_size - 1, sequence // block_size - selected_blocks - 1, -1, dtype=torch.int32).view(1, 1, selected_blocks).expand(batch, key_heads, selected_blocks).contiguous()
    lengths = torch.full((batch,), sequence, dtype=torch.int32)
    split_offsets = torch.arange(0, selected_blocks + 1, selected_blocks // splits, dtype=torch.int32)
    scale = dimension ** (-0.5)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_block_sparse_gqa_decode')
    source_state = {}

    def source_launch():
        source_state['output'] = runtime.block_sparse_gqa_decode(q, k, v, selected, lengths, split_offsets, scale, block_size=block_size)
    return Side(PreparedLaunch(source_launch, lambda : source_state['output']), Tolerance(atol=0.05, rtol=0.05), cuda_graph=False, device_type='cpu', note='完整 block-sparse GQA partial+combine pipeline；保持原倒序 selected blocks；生成端输出及 partial 预分配，CPU reference 计入 block gather、临时分配和 attention。', owners=locals())
