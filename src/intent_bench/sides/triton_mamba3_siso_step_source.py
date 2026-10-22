from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import importlib
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.model import PreparedCall
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
    _activate(context, 'corpus/gpu/baselines/triton/state-spaces-mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_step_runtime.py', 'intent_v2_triton_mamba3_runtime')
    source_function = importlib.import_module('mamba_ssm.ops.triton.mamba3.mamba3_siso_step').mamba3_siso_step
    source_base = functional_launch(lambda : source_function(query, key, value, adt, dt, trap, query_bias, key_bias, angles, D=residual, Z=gate, Input_States=states))
    source = PreparedCall(launch=source_base.launch, result=lambda : (source_base.result()[0], source_base.result()[1][0], source_base.result()[1][1], source_base.result()[1][2]))
    return checked_call(source, (Tolerance(atol=0.005, rtol=0.05), Tolerance(atol=1e-05, rtol=1e-05), Tolerance(atol=0.0001, rtol=0.0001), Tolerance(atol=0.005, rtol=0.05)), cuda_graph=False, note='同算法；source 使用近似 tanh/sin/cos；完整调用计时包含其输出分配，generated 使用预分配输出', owners=locals())

def _activate(context, runtime_path, name):
    return load_module(context.project_root / runtime_path, name)
