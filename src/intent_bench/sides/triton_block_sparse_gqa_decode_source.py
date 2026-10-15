from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, query_heads, kv_heads, dimension) = (8, 32, 8, 128)
    (sequence, selected_blocks, block_size, splits) = (8192, 32, 128, 4)
    blocks_per_sequence = sequence // block_size
    q = torch.randn((batch, query_heads, dimension), device='cuda', dtype=torch.float16)
    key = torch.randn((batch, sequence, kv_heads, dimension), device='cuda', dtype=torch.float16)
    value = torch.randn_like(key)
    selected = torch.arange(selected_blocks, device='cuda', dtype=torch.int32).view(1, 1, selected_blocks).expand(batch, kv_heads, selected_blocks).contiguous()
    lengths = torch.full((batch,), sequence, device='cuda', dtype=torch.int32)
    scale = dimension ** (-0.5)
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/vllm/attention/minimax_m3/sparse_decode_runtime.py', 'intent_v2_triton_sparse_decode_runtime')
    source_module = runtime.RUNTIME.load_source(context.project_root / 'corpus/gpu/baselines/triton/vllm/attention/minimax_m3/sparse_attn.py', 'intent_v2_triton_sparse_decode_source')
    key_blocks = key.view(batch, blocks_per_sequence, block_size, kv_heads, dimension).permute(0, 1, 3, 2, 4)
    value_blocks = value.view(batch, blocks_per_sequence, block_size, kv_heads, dimension).permute(0, 1, 3, 2, 4)
    kv_cache = torch.cat((key_blocks, value_blocks), dim=-1).reshape(batch * blocks_per_sequence, kv_heads, block_size, 2 * dimension)
    block_table = torch.arange(batch * blocks_per_sequence, device='cuda', dtype=torch.int32).view(batch, blocks_per_sequence)
    topk = selected.permute(1, 0, 2).contiguous()
    source_output = torch.empty_like(q)

    def source_launch():
        source_module.minimax_m3_sparse_attn_decode(q, kv_cache, topk, block_table, lengths, kv_heads, scale, source_output, decode_query_len=1)
    source_launch()
    source = PreparedLaunch(launch=source_launch, outputs=lambda : source_output)
    return Side(source, Tolerance(atol=0.05, rtol=0.05), cuda_graph=False, note='同算法；source 使用 packed KV/topk ABI；输入布局适配在计时前完成，完整 source 调用含 partial buffer 分配', owners=locals())
