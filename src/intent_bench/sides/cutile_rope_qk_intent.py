from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.position.rope import rotary_qk_bf16_inplace
from intent_bench.corpus._common.measurement import compile_kernel
from intent_bench.implementations.prepared import intent_call
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, sequence, query_heads, key_heads, dimension) = (2, 4096, 32, 8, 128)
    query = torch.randn((batch, query_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    key = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    cosine = torch.randn((1, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    sine = torch.randn_like(cosine)
    initial_query = query.clone()
    initial_key = key.clone()
    artifact = compile_kernel(context, rotary_qk_bf16_inplace)

    def generated_prepare():
        query.copy_(initial_query)
        key.copy_(initial_key)
    arguments = (query, key, cosine, sine)
    def host_factory():
        operator = torch.compile(artifact.as_torch_op('intent_gpu_benchmark::rope_qk'), fullgraph=True)
        return lambda: operator(*arguments)
    generated = intent_call(artifact, arguments, device=str(query.device), reset=generated_prepare,
                            observe=lambda _: (query, key), host_factory=host_factory)
    return checked_call(generated, (Tolerance(atol=0.02, rtol=0.01), Tolerance(atol=0.02, rtol=0.01)), cuda_graph=True, note='Prepared in-place RoPE; original bf16 rounding preserved. torch.compile integration is opt-in via host_call.', owners=locals())
