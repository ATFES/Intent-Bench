from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.backward.attention import attention_backward_delta, attention_backward_dkdv, attention_backward_dq
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
ATTENTION_BATCH = 2
ATTENTION_HEAD_DIMENSION = 64
ATTENTION_HEAD_GROUP = 4
ATTENTION_KV_HEADS = 2
ATTENTION_QUERY_HEADS = 8
ATTENTION_SCALE = 0.125
ATTENTION_SEQUENCE = 1024

def prepare(context):
    shape_q = (ATTENTION_BATCH, ATTENTION_QUERY_HEADS, ATTENTION_SEQUENCE, ATTENTION_HEAD_DIMENSION)
    shape_kv = (ATTENTION_BATCH, ATTENTION_KV_HEADS, ATTENTION_SEQUENCE, ATTENTION_HEAD_DIMENSION)
    q = torch.randn(shape_q, dtype=torch.float16) * 0.5
    k = torch.randn(shape_kv, dtype=torch.float16) * 0.5
    v = torch.randn(shape_kv, dtype=torch.float16) * 0.5
    grad_output = torch.randn_like(q) * 0.05
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_attention_backward')
    (output, lse, source_backward) = runtime.prepare_attention_backward(q, k, v, grad_output, ATTENTION_SCALE)
    delta_artifact = intent.compile(attention_backward_delta, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    constexprs = {'HEAD_GROUP': ATTENTION_HEAD_GROUP, 'CAUSAL': True}
    dkdv_artifact = intent.compile(attention_backward_dkdv, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options, constexprs=constexprs)
    dq_artifact = intent.compile(attention_backward_dq, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options, constexprs=constexprs)
    delta = torch.empty(shape_q[:-1], dtype=torch.float32)
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    generated_state = {}

    def generated_launch():
        delta_artifact(output, grad_output, delta)
        dkdv_artifact(q, k, v, grad_output, lse, delta, grad_k, grad_v, ATTENTION_SCALE)
        dq_artifact(q, k, v, grad_output, lse, delta, grad_q, ATTENTION_SCALE)
        generated_state['output'] = (grad_q, grad_k, grad_v)
    return Side(PreparedLaunch(generated_launch, lambda : generated_state['output']), (Tolerance(atol=0.125), Tolerance(atol=0.25), Tolerance(atol=0.125)), cuda_graph=False, device_type='cpu', note='既有B2-HQ8-HK2-S1024-D64 f16 causal attention backward；delta、dkdv、dq三kernel完整调用；CPU source为原f32 autograd参考，保留前向图及中间量，生成端仅接收output/lse并重算概率，双方前向准备不计时。', owners=locals())
