from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.streaming.mla import paged_mla_decode_partials
from kernels.streaming.splitk_reduce import splitk_attention_f32_to_f16_reduce
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, query_heads, kv_heads) = (8, 128, 1)
    (sequence, latent_dimension, rope_dimension) = (8192, 512, 64)
    key_dimension = latent_dimension + rope_dimension
    (page_size, splits) = (16, 8)
    pages_per_sequence = sequence // page_size
    pages = batch * pages_per_sequence
    q_latent = torch.randn((batch, query_heads, latent_dimension), device='cuda', dtype=torch.float16)
    q_rope = torch.randn((batch, query_heads, rope_dimension), device='cuda', dtype=torch.float16)
    latent_cache = torch.randn((pages, page_size, kv_heads, latent_dimension), device='cuda', dtype=torch.float16)
    rope_cache = torch.randn((pages, page_size, kv_heads, rope_dimension), device='cuda', dtype=torch.float16)
    page_indices = torch.arange(pages, device='cuda', dtype=torch.int32)
    page_offsets = torch.arange(0, pages + 1, pages_per_sequence, device='cuda', dtype=torch.int32)
    lengths = torch.full((batch,), sequence, device='cuda', dtype=torch.int32)
    scale = key_dimension ** (-0.5)
    (_, partials) = compile_single(context, paged_mla_decode_partials, (q_latent, q_rope, latent_cache.view(pages * page_size, kv_heads, latent_dimension), rope_cache.view(pages * page_size, kv_heads, rope_dimension), page_offsets, page_indices, lengths, scale), constexprs={'PAGE_SIZE': page_size, 'HEAD_GROUP': query_heads // kv_heads, 'SPLITS': splits})
    (partial_lse, partial_output) = partials.outputs()
    (_, reduction) = compile_single(context, splitk_attention_f32_to_f16_reduce, (partial_output, partial_lse))

    def generated_launch():
        partials.launch()
        reduction.launch()
    generated = PreparedLaunch(generated_launch, reduction.outputs)
    return Side(generated, Tolerance(atol=0.05, rtol=0.05), cuda_graph=False, note='同算法；source 使用 packed KV/page-table ABI 并写辅助 LSE；输入布局适配在计时前完成，完整 source 调用含 scale 分配', owners=locals())
