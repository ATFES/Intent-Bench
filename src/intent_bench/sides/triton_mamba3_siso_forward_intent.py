from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.streaming.mamba import mamba3_siso_forward as mamba3_siso_forward_kernel
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, sequence, qk_heads, heads) = (1, 2048, 4, 16)
    (qk_dimension, value_dimension, angle_dimension) = (32, 64, 16)
    dtype = torch.bfloat16
    query = torch.randn((batch, sequence, qk_heads, qk_dimension), device='cuda', dtype=dtype) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn((batch, sequence, heads, value_dimension), device='cuda', dtype=dtype) * 0.1
    adt = -torch.rand((batch, heads, sequence), device='cuda', dtype=torch.float32) * 0.1
    dt = torch.rand_like(adt) * 0.1
    trap = torch.rand_like(adt) * 0.1
    query_bias = torch.randn((heads, qk_dimension), device='cuda', dtype=dtype) * 0.01
    key_bias = torch.randn_like(query_bias) * 0.01
    angles = torch.randn((batch, sequence, heads, angle_dimension), device='cuda', dtype=torch.float32) * 0.01
    residual = torch.randn((heads,), device='cuda', dtype=torch.float32) * 0.01
    gate = torch.randn((batch, sequence, heads, value_dimension), device='cuda', dtype=dtype) * 0.1
    arguments = (query, key, value, adt, dt, trap, query_bias, key_bias, angles, residual, gate, torch.empty((batch, sequence, heads, qk_dimension), device='cuda', dtype=dtype), torch.empty((batch, sequence, heads, qk_dimension), device='cuda', dtype=dtype), torch.empty((batch, heads, sequence), device='cuda', dtype=torch.float32), torch.empty((batch, heads, sequence), device='cuda', dtype=torch.float32), torch.empty((batch, heads, sequence), device='cuda', dtype=torch.float32))
    (_, generated_base) = compile_single(context, mamba3_siso_forward_kernel, arguments, constexprs={'HEAD_GROUP': heads // qk_heads})
    generated = PreparedCall(generated_base.launch, generated_base.result)
    return checked_call(generated, Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, note='同算法；source 使用近似 sin/cos；完整调用计时包含其输出及 workspace 分配，generated 使用预分配缓冲', owners=locals())
