from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, IntentSequence
import math
import intent
import torch
from intent.examples.kernels.streaming.attention_specialized import gemma_gqa_decode_partials
from intent.examples.kernels.streaming.splitk_reduce import splitk_attention_reduce
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, query_heads, key_heads, sequence, dimension, split_size) = (32, 32, 8, 8192, 128, 256)
    splits = (sequence + split_size - 1) // split_size
    q = torch.randn((batch, query_heads, 1, dimension), dtype=torch.bfloat16)
    k = torch.randn((batch, key_heads, sequence, dimension), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scale = 1.0 / math.sqrt(dimension)
    partials = context.compile(gemma_gqa_decode_partials, constexprs={'HEAD_GROUP': query_heads // key_heads, 'WINDOW': 1024, 'SOFT_CAP': 50.0, 'P': splits})
    reduction = context.compile(splitk_attention_reduce)
    partial_lse = torch.empty((batch, query_heads, splits), dtype=torch.float32)
    partial_output = torch.empty((batch, query_heads, splits, dimension), dtype=torch.bfloat16)
    output = torch.empty((batch, query_heads, dimension), dtype=torch.bfloat16)

    def generated_launch():
        partials(q, k, v, partial_lse, partial_output, scale)
        reduction(partial_output, partial_lse, output)
    sequence = IntentSequence()
    sequence.add(partials, q, k, v, scale, outputs=(partial_lse, partial_output))
    sequence.add(reduction, partial_output, partial_lse, outputs=(output,))
    generated = sequence.prepared(lambda: output.unsqueeze(2), host_run=generated_launch)
    return checked_call(generated, Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, device_type='cpu', note='完整 Gemma GQA decode partial+split-k native序列；输出及 partial 预绑定，CPU reference 为 PyTorch eager，计入其临时分配；原输入、输出、容差不变。', owners=locals())
