from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import math
import torch
from intent.examples.kernels.streaming.gated_delta import chunk_gated_delta_prepare
from intent.examples.kernels.streaming.gated_delta import chunk_gated_delta_recurrence
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, sequence, heads, key_dimension, value_dimension) = (2, 2048, 8, 128, 128)
    query = torch.randn((batch, sequence, heads, key_dimension), device='cuda', dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn((batch, sequence, heads, value_dimension), device='cuda', dtype=torch.bfloat16) * 0.1
    gate = -torch.rand((batch, sequence, heads), device='cuda', dtype=torch.bfloat16) * 0.5
    beta = torch.sigmoid(torch.randn_like(gate))
    scale = 1.0 / math.sqrt(key_dimension)
    (_, prepare) = compile_single(context, chunk_gated_delta_prepare, (query, key, value, gate, beta, scale))
    intermediate_outputs = prepare.result()
    (_, recurrence) = compile_single(context, chunk_gated_delta_recurrence, intermediate_outputs)

    def generated_launch():
        prepare.launch()
        recurrence.launch()
    generated = PreparedCall(generated_launch, recurrence.result)
    tolerance = Tolerance(atol=0.1, rtol=0.05)
    return checked_call(generated, (tolerance, tolerance), cuda_graph=False, note='同算法；generated 使用 BF16 矩阵输入、FP32 累加与状态，source 使用 TF32 矩阵输入', owners=locals())
