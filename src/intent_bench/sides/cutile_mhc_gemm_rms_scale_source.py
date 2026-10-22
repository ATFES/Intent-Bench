from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import cuda.tile as ct
import torch
from intent_bench.corpus._common.measurement import initial_launch
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (tokens, hidden, streams) = (2048, 4096, 4)
    x = torch.randn((tokens, streams * hidden), device='cuda', dtype=torch.bfloat16)
    width = streams * (streams + 2)
    weight = torch.randn((streams * hidden, width), device='cuda', dtype=torch.bfloat16)
    bias = torch.randn((width,), device='cuda', dtype=torch.bfloat16)
    source_module = _source(context)
    (source_partial_linear, source_partial_square_sum, config) = initial_launch(lambda : source_module._cutile_autotune_mhc_split_gemm_rms(torch.cuda.current_stream(), x, weight, tokens, width, x.shape[1]), side='source')
    tile_m = config.TILE_SIZE_M
    tile_n = config.TILE_SIZE_N
    tile_k = config.TILE_SIZE_K
    group_m = config.GROUP_SIZE_M
    source_split_count = config.SPLIT_K
    row_tiles = (tokens + tile_m - 1) // tile_m
    column_tiles = (width + tile_n - 1) // tile_n
    source_mixed = torch.empty((tokens, width), device='cuda', dtype=torch.bfloat16)
    source_rms = torch.empty((tokens, 1), device='cuda', dtype=torch.float32)

    def source_launch():
        ct.launch(torch.cuda.current_stream(), (row_tiles * column_tiles, source_split_count, 1), source_module._mhc_split_gemm_rms_kernel, (x, weight, source_partial_linear, source_partial_square_sum, tokens, width, x.shape[1], tile_m, tile_n, tile_k, source_split_count, group_m))
        ct.launch(torch.cuda.current_stream(), (row_tiles, column_tiles, 1), source_module._mhc_finalize_scale_bias_sigmoid_kernel, (source_partial_linear, source_partial_square_sum, source_mixed, source_rms, streams, 1.0, 1.0, 1.0, bias, tokens, width, x.shape[1], tile_m, tile_n, source_split_count))
    initial_launch(source_launch, side='source')
    source = PreparedCall(source_launch, lambda : (source_mixed, source_rms))
    return checked_call(source, (Tolerance(atol=0.05, rtol=0.02), Tolerance(atol=0.005, rtol=0.001)), cuda_graph=False, note='同算法；source 用乘倒数实现归一化', owners=locals())

def _source(context):
    return tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/mhc/fused/mhc.py', 'mhc')
