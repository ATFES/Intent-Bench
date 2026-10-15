from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.streaming.gated_delta import chunk_gated_delta_prepare, chunk_gated_delta_recurrence
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    query = torch.randn((2, 2048, 8, 128), dtype=torch.bfloat16) * 0.1
    (key, value) = (torch.randn_like(query) * 0.1, torch.randn_like(query) * 0.1)
    gate = -torch.rand((2, 2048, 8), dtype=torch.bfloat16) * 0.5
    beta = torch.sigmoid(torch.randn_like(gate))
    arguments = (query, key, value, gate, beta, 128 ** (-0.5))
    (preparation, recurrence) = (intent.compile(definition, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options) for definition in (chunk_gated_delta_prepare, chunk_gated_delta_recurrence))
    (generated, source) = ({}, {})

    def launch():
        generated['output'] = recurrence.run(*preparation.run(*arguments))
    return Side(PreparedLaunch(launch, lambda : generated['output']), (Tolerance(atol=0.1, rtol=0.05), Tolerance(atol=0.1, rtol=0.05)), cuda_graph=False, device_type='cpu', note='原B2/S2048/H8/K128/V128 BF16 chunk64 gated-delta完整prepare+recurrence，含中间buffer分配，比较output和f32 final state；CPU参考保留source guarded Neumann/serial triangular solve和chunk recurrence，BF16中间ABI，matmul使用f32而非原GPU TF32或Intent显式BF16输入转换，原容差。', owners=locals())
