from __future__ import annotations
import math
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall
ATTENTION_BATCH = 2
ATTENTION_HEAD_DIMENSION = 64
ATTENTION_KV_HEADS = 2
ATTENTION_QUERY_HEADS = 8
ATTENTION_SCALE = 0.125
ATTENTION_SEQUENCE = 1024

def _inputs(context):
    shape_q = (ATTENTION_BATCH, ATTENTION_QUERY_HEADS, ATTENTION_SEQUENCE, ATTENTION_HEAD_DIMENSION)
    shape_kv = (ATTENTION_BATCH, ATTENTION_KV_HEADS, ATTENTION_SEQUENCE, ATTENTION_HEAD_DIMENSION)
    q = torch.randn(shape_q, dtype=torch.float16) * 0.5
    k = torch.randn(shape_kv, dtype=torch.float16) * 0.5
    v = torch.randn(shape_kv, dtype=torch.float16) * 0.5
    grad_output = torch.randn_like(q) * 0.05
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_attention_backward')
    (output, lse, source_backward) = runtime.prepare_attention_backward(q, k, v, grad_output, ATTENTION_SCALE)
    return (q, k, v, output, grad_output, lse * math.log(2.0), ATTENTION_SCALE), source_backward

def prepare_native(context):
    from intent_bench.implementations.aten_cpu import prepare as native
    arguments, _ = _inputs(context)
    call = native('attention_backward', arguments, context.environment)
    return checked_call(call, (Tolerance(atol=0.125), Tolerance(atol=0.25), Tolerance(atol=0.125)),
                        device_type='cpu', owners={'inputs': arguments})

def prepare(context):
    _, source_backward = _inputs(context)
    source_state = {}

    def source_launch():
        source_state['output'] = source_backward()
    return checked_call(PreparedCall(source_launch, lambda : source_state['output']), (Tolerance(atol=0.125), Tolerance(atol=0.25), Tolerance(atol=0.125)), cuda_graph=False, device_type='cpu', note='既有B2-HQ8-HK2-S1024-D64 f16 causal attention backward；delta、dkdv、dq三kernel完整调用；CPU source为原f32 autograd参考，保留前向图及中间量，生成端仅接收output/lse并重算概率，双方前向准备不计时。', owners=locals())
