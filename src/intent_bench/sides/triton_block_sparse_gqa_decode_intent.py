from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.streaming.block_sparse_attention import block_sparse_gqa_decode_combine
from intent.examples.kernels.streaming.block_sparse_attention import block_sparse_gqa_decode_partials
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, query_heads, kv_heads, dimension) = (8, 32, 8, 128)
    (sequence, selected_blocks, block_size, splits) = (8192, 32, 128, 4)
    q = torch.randn((batch, query_heads, dimension), device='cuda', dtype=torch.float16)
    key = torch.randn((batch, sequence, kv_heads, dimension), device='cuda', dtype=torch.float16)
    value = torch.randn_like(key)
    selected = torch.arange(selected_blocks, device='cuda', dtype=torch.int32).view(1, 1, selected_blocks).expand(batch, kv_heads, selected_blocks).contiguous()
    lengths = torch.full((batch,), sequence, device='cuda', dtype=torch.int32)
    split_offsets = torch.arange(0, selected_blocks + 1, selected_blocks // splits, device='cuda', dtype=torch.int32)
    scale = dimension ** (-0.5)
    (_, partial) = compile_single(context, block_sparse_gqa_decode_partials, (q, key, value, selected, lengths, split_offsets, scale), constexprs={'HEAD_GROUP': query_heads // kv_heads, 'BLOCK_SIZE': block_size, 'SPLITS': splits})
    (partial_lse, partial_output) = partial.result()
    (_, combined) = compile_single(context, block_sparse_gqa_decode_combine, (partial_lse, partial_output), constexprs={'SPLITS': splits})

    def generated_launch():
        partial.launch()
        combined.launch()
    generated = PreparedCall(launch=generated_launch, result=combined.result)
    return checked_call(generated, Tolerance(atol=0.05, rtol=0.05), cuda_graph=False, note='同算法；source 使用 packed KV/topk ABI；输入布局适配在计时前完成，完整 source 调用含 partial buffer 分配', owners=locals())
