from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.position.rope import rotary_qk_bf16_inplace
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.measurement import initial_launch
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, sequence, query_heads, key_heads, dimension) = (2, 4096, 32, 8, 128)
    query = torch.randn((batch, query_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    key = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    cosine = torch.randn((1, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    sine = torch.randn_like(cosine)
    initial_query = query.clone()
    initial_key = key.clone()
    (artifact, _) = compile_single(context, rotary_qk_bf16_inplace, (query, key, cosine, sine))

    def generated_prepare():
        query.copy_(initial_query)
        key.copy_(initial_key)
    operator = torch.compile(artifact.as_torch_op('intent_gpu_benchmark::rope_qk'), fullgraph=True)
    generated_launch = lambda : operator(query, key, cosine, sine)
    generated_prepare()
    initial_launch(generated_launch, side='generated')
    generated_prepare()
    generated = PreparedLaunch(generated_launch, lambda : (query, key), generated_prepare)
    return Side(generated, (Tolerance(atol=0.02, rtol=0.01), Tolerance(atol=0.02, rtol=0.01)), cuda_graph=False, note='同算法；source 保留 bf16 中间舍入；Intent 通过 mutable custom op 与 torch.compile(fullgraph=True) 调用，计完整执行，含框架产生的状态处理', owners=locals())
