from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.streaming.mamba import mamba3_siso_step
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, qk_heads, heads) = (32, 4, 16)
    (qk_dimension, value_dimension, angle_dimension) = (32, 64, 16)
    dtype = torch.bfloat16
    query = torch.randn((batch, qk_heads, qk_dimension), device='cuda', dtype=dtype) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn((batch, heads, value_dimension), device='cuda', dtype=dtype) * 0.1
    adt = -torch.rand((batch, heads), device='cuda', dtype=torch.float32) * 0.1
    dt = torch.rand((batch, heads), device='cuda', dtype=torch.float32) * 0.1
    trap = torch.rand((batch, heads), device='cuda', dtype=torch.float32) * 0.1
    query_bias = torch.randn((heads, qk_dimension), device='cuda', dtype=dtype) * 0.01
    key_bias = torch.randn_like(query_bias) * 0.01
    angles = torch.randn((batch, heads, angle_dimension), device='cuda', dtype=torch.float32) * 0.01
    residual = torch.randn((heads,), device='cuda', dtype=torch.float32) * 0.01
    gate = torch.randn((batch, heads, value_dimension), device='cuda', dtype=dtype) * 0.1
    states = (torch.zeros((batch, heads, angle_dimension), device='cuda', dtype=torch.float32), torch.zeros((batch, heads, value_dimension, qk_dimension), device='cuda', dtype=torch.float32), torch.zeros((batch, heads, qk_dimension), device='cuda', dtype=torch.float32), torch.zeros((batch, heads, value_dimension), device='cuda', dtype=torch.float32))
    arguments = (query, key, value, adt, dt, trap, query_bias, key_bias, angles, residual, gate, *states)
    (_, generated) = compile_single(context, mamba3_siso_step, arguments, constexprs={'HEAD_GROUP': heads // qk_heads})
    return checked_call(generated, (Tolerance(atol=0.005, rtol=0.05), Tolerance(atol=1e-05, rtol=1e-05), Tolerance(atol=0.0001, rtol=0.0001), Tolerance(atol=0.005, rtol=0.05)), cuda_graph=False, note='同算法；source 使用近似 tanh/sin/cos；完整调用计时包含其输出分配，generated 使用预分配输出', owners=locals())
