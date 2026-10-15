from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (batch, sequence, heads, key_dimension, value_dimension) = (2, 2048, 8, 128, 128)
    query = torch.randn((batch, sequence, heads, key_dimension), device='cuda', dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn((batch, sequence, heads, value_dimension), device='cuda', dtype=torch.bfloat16) * 0.1
    gate = -torch.rand((batch, sequence, heads), device='cuda', dtype=torch.bfloat16) * 0.5
    beta = torch.sigmoid(torch.randn_like(gate))
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/scan/gated_delta_chunk/chunk_gated_delta_rule.py', 'chunk_gated_delta', needs_utils=True)
    source = functional_launch(lambda : source_module.chunk_gated_delta_rule(query, key, value, gate, beta, chunk_size=64, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=False))
    tolerance = Tolerance(atol=0.1, rtol=0.05)
    return Side(source, (tolerance, tolerance), cuda_graph=False, note='同算法；generated 使用 BF16 矩阵输入、FP32 累加与状态，source 使用 TF32 矩阵输入', owners=locals())
