from __future__ import annotations
from intent_bench.sides import Side
import torch
import triton
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, query_heads, kv_heads) = (32, 32, 8)
    (cache_length, dimension) = (8192, 128)
    padded_length = cache_length + 1
    query = torch.randn((batch, query_heads, dimension), device='cuda', dtype=torch.float16)
    key = torch.randn((batch, kv_heads, dimension), device='cuda', dtype=torch.float16)
    value = torch.randn_like(key)
    sequence_lengths = torch.full((batch,), padded_length, device='cuda', dtype=torch.int32)
    position = cache_length
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/xformers/position/rope_padded/rope_padded_kernels_runtime.py', 'intent_v2_triton_padded_rope_runtime')
    source_module = runtime.load_source()
    source_query = query.unsqueeze(0)
    source_key = key.unsqueeze(0)
    source_value = value.unsqueeze(0)
    source_query_output = torch.empty_like(source_query)
    source_key_cache = torch.empty((1, batch * padded_length, kv_heads, dimension), device='cuda', dtype=torch.float16)
    source_value_cache = torch.empty_like(source_key_cache)
    query_offsets = torch.arange(batch + 1, device='cuda', dtype=torch.int32)
    cache_offsets = query_offsets * padded_length
    block_size = max(128, min(4096, triton.next_power_of_2(dimension)))
    total_heads = query_heads + 2 * kv_heads

    def source_launch():
        q_stride = source_query.stride()
        k_stride = source_key.stride()
        v_stride = source_value.stride()
        ck_stride = source_key_cache.stride()
        cv_stride = source_value_cache.stride()
        out_stride = source_query_output.stride()
        source_module._rope_padded_kernel[1, batch, total_heads](source_query, source_key, source_value, source_query_output, source_key_cache, source_value_cache, query_offsets, cache_offsets, sequence_lengths, 10000.0, 1.0, False, 0, 0, 0, 0, None, None, query_heads, query_heads + kv_heads, 1, dimension, q_stride[1], 0, q_stride[-2], k_stride[1], 0, k_stride[-2], v_stride[1], 0, v_stride[-2], ck_stride[1], 0, ck_stride[-2], cv_stride[1], 0, cv_stride[-2], query_offsets.stride(0), cache_offsets.stride(0), sequence_lengths.stride(0), out_stride[1], 0, out_stride[-2], 0, 'f32', const_batch_strides=False, cache_padding_length=0, seqlenk_shift=0, BLOCK_SIZE=block_size, adjacents=False, num_warps=1)
    source_launch()
    source_key_view = source_key_cache.view(batch, padded_length, kv_heads, dimension)
    source_value_view = source_value_cache.view(batch, padded_length, kv_heads, dimension)
    source = PreparedLaunch(launch=source_launch, outputs=lambda : (source_query_output.squeeze(0), source_key_view[:, position], source_value_view[:, position]))
    return Side(source, (Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.0)), cuda_graph=False, owners=locals())
