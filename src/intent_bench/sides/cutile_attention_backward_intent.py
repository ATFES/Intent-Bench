from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import torch
from intent.examples.programs.attention_backward import AttentionBackward
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, query_heads, key_heads, sequence, dimension) = (2, 8, 2, 1024, 64)
    shape = (batch, query_heads, sequence, dimension)
    query = torch.randn(shape, device='cuda', dtype=torch.float16) * 0.5
    key = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.float16) * 0.5
    value = torch.randn_like(key) * 0.5
    scale = dimension ** (-0.5)
    repeated_key = key.repeat_interleave(query_heads // key_heads, dim=1)
    repeated_value = value.repeat_interleave(query_heads // key_heads, dim=1)
    scores = torch.matmul(query.float(), repeated_key.float().transpose(-1, -2)) * scale
    causal = torch.ones((sequence, sequence), device='cuda', dtype=torch.bool).tril()
    scores.masked_fill_(~causal, -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    output = torch.matmul(torch.softmax(scores, dim=-1), repeated_value.float()).half()
    grad_output = torch.randn_like(output) * 0.05
    program = AttentionBackward.from_jit(context.compile, head_group=query_heads // key_heads, causal=True)
    generated = intent_call(program, (query, key, value, output, grad_output, lse, scale), device='cuda')
    return checked_call(generated, (Tolerance(atol=0.125), Tolerance(atol=0.25), Tolerance(atol=0.125)), cuda_graph=True, note='同算法；中间精度和舍入不同', owners=locals())
