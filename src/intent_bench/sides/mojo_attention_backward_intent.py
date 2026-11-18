from __future__ import annotations
import math
from intent_bench.implementations.prepared import checked_call, intent_call
import torch
from intent.examples.programs.attention_backward import AttentionBackward
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
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
    # The retained reference helper exports log2 LSE; the product kernels take
    # natural-log LSE and apply LOG2E themselves when reconstructing probability.
    lse = lse * math.log(2.0)
    program = AttentionBackward.from_jit(context.compile, head_group=ATTENTION_HEAD_GROUP, causal=True)
    delta = torch.empty(shape_q[:-1], dtype=torch.float32)
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)

    generated = intent_call(program, (q, k, v, output, grad_output, lse, ATTENTION_SCALE),
                            bindings={'outputs': (grad_q, grad_k, grad_v), 'workspace': delta})
    return checked_call(generated, (Tolerance(atol=0.125), Tolerance(atol=0.25), Tolerance(atol=0.125)), cuda_graph=False, device_type='cpu', note='既有B2-HQ8-HK2-S1024-D64 f16 causal attention backward；delta、dkdv、dq三kernel完整native序列；CPU source为原f32 autograd参考，保留前向图及中间量，生成端仅接收output/lse并重算概率，双方前向准备不计时。', owners=locals())
