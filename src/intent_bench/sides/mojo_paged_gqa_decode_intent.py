from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from programs.composition import PagedDecode
from kernels.streaming.paged_attention import paged_gqa_decode_partials
from kernels.streaming.splitk_reduce import splitk_attention_f32_to_f16_reduce
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
    partials = intent.compile(paged_gqa_decode_partials, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options, constexprs={'PAGE_SIZE': page_size, 'HEAD_GROUP': query_heads // kv_heads, 'SPLITS': splits})
    reduction = intent.compile(splitk_attention_f32_to_f16_reduce, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    partial_lse = torch.empty((batch, query_heads, splits), dtype=torch.float32)
    partial_output = torch.empty((batch, query_heads, splits, dimension), dtype=torch.float32)
    output = torch.empty((batch, query_heads, dimension), dtype=torch.float16)
    program = PagedDecode(partials, reduction)

    def generated_launch():
        program.into(q, key_cache, value_cache, page_offsets, page_indices, lengths, split_offsets, partial_lse, partial_output, output, scale)
    generated = PreparedLaunch(generated_launch, lambda : output)
    return Side(generated, Tolerance(atol=0.05, rtol=0.05), cuda_graph=False, device_type='cpu', note='完整 paged GQA partial+split-k pipeline；生成端输出及 partial 预分配，CPU reference 计入 page table 解码、临时分配和 attention。', owners=locals())
