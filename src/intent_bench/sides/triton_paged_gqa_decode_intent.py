from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.streaming.paged_attention import paged_gqa_decode_partials
from intent.examples.kernels.streaming.splitk_reduce import splitk_attention_f32_to_f16_reduce
from intent.examples.programs.composition import PagedDecode
from intent_bench.corpus._common.measurement import compile_kernel, prepare_program
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, query_heads, kv_heads, dimension) = (16, 32, 8, 128)
    (sequence, page_size, splits) = (8192, 16, 8)
    pages_per_sequence = sequence // page_size
    pages_per_split = pages_per_sequence // splits
    pages = batch * pages_per_sequence
    q = torch.randn((batch, query_heads, dimension), device='cuda', dtype=torch.float16)
    key_cache = torch.randn((pages, page_size, kv_heads, dimension), device='cuda', dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    page_indices = torch.arange(pages, device='cuda', dtype=torch.int32)
    page_offsets = torch.arange(0, pages + 1, pages_per_sequence, device='cuda', dtype=torch.int32)
    split_offsets = torch.arange(0, pages + 1, pages_per_split, device='cuda', dtype=torch.int32)
    lengths = torch.full((batch,), sequence, device='cuda', dtype=torch.int32)
    scale = dimension ** (-0.5)
    partials = compile_kernel(context, paged_gqa_decode_partials, constexprs={'PAGE_SIZE': page_size, 'HEAD_GROUP': query_heads // kv_heads, 'SPLITS': splits})
    reduction = compile_kernel(context, splitk_attention_f32_to_f16_reduce)
    generated = prepare_program(PagedDecode(partials, reduction), (q, key_cache, value_cache, page_offsets, page_indices, lengths, split_offsets, scale))
    return checked_call(generated, Tolerance(atol=0.05, rtol=0.05), cuda_graph=False, owners=locals())
