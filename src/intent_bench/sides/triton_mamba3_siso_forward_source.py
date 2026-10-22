from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import importlib
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
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
    _activate(context, 'corpus/gpu/baselines/triton/state-spaces-mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd_runtime.py', 'intent_v2_triton_mamba3_siso_forward_runtime')
    source_function = importlib.import_module('mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd').mamba3_siso_fwd
    source_base = functional_launch(lambda : source_function(query, key, value, adt, dt, trap, query_bias, key_bias, angles, D=residual, Z=gate, chunk_size=64))
    source = PreparedCall(source_base.launch, lambda : source_base.result()[0])
    return checked_call(source, Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, note='同算法；source 使用近似 sin/cos；完整调用计时包含其输出及 workspace 分配，generated 使用预分配缓冲', owners=locals())

def _activate(context, runtime_path, name):
    return load_module(context.project_root / runtime_path, name)
