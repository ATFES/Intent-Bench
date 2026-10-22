from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.position.rope import rotary_qk_inplace
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, sequence, query_heads, kv_heads, dimension) = (4, 4096, 32, 8, 128)
    query_initial = torch.randn((batch, query_heads, sequence, dimension), device='cuda', dtype=torch.float16)
    key_initial = torch.randn((batch, kv_heads, sequence, dimension), device='cuda', dtype=torch.float16)
    positions = torch.arange(sequence, device='cuda', dtype=torch.float32)
    inverse = 1.0 / 10000 ** (torch.arange(0, dimension, 2, device='cuda', dtype=torch.float32) / dimension)
    frequency = torch.outer(positions, inverse)
    embedding = torch.cat((frequency, frequency), dim=-1)
    cosine = embedding.cos().unsqueeze(0).to(torch.float16)
    sine = embedding.sin().unsqueeze(0).to(torch.float16)
    generated_query = query_initial.clone()
    generated_key = key_initial.clone()
    (_, generated_base) = compile_single(context, rotary_qk_inplace, (generated_query, generated_key, cosine, sine))
    generated = PreparedCall(launch=generated_base.launch, result=lambda : (generated_query, generated_key), reset=lambda : (generated_query.copy_(query_initial), generated_key.copy_(key_initial)))
    return checked_call(generated, (Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.002, rtol=0.002)), cuda_graph=False, owners=locals())
