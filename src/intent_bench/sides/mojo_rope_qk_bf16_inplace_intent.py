from __future__ import annotations
from intent_bench.sides import Side
import torch
import intent
from kernels.position.rope import rotary_qk_bf16_inplace
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    (batch, sequence, query_heads, kv_heads, dimension) = (2, 4096, 32, 8, 128)
    query_initial = torch.randn((batch, query_heads, sequence, dimension), dtype=torch.bfloat16)
    key_initial = torch.randn((batch, kv_heads, sequence, dimension), dtype=torch.bfloat16)
    cosine = torch.randn((1, sequence, dimension), dtype=torch.bfloat16)
    sine = torch.randn_like(cosine)
    artifact = intent.compile(rotary_qk_bf16_inplace, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    generated_query = query_initial.clone()
    generated_key = key_initial.clone()

    def generated_launch():
        artifact.run(generated_query, generated_key, cosine, sine)

    def restore_generated():
        generated_query.copy_(query_initial)
        generated_key.copy_(key_initial)
    generated = PreparedLaunch(generated_launch, lambda : (generated_query, generated_key), prepare=restore_generated)
    return Side(generated, (Tolerance(atol=0.02, rtol=0.01), Tolerance(atol=0.02, rtol=0.01)), cuda_graph=False, device_type='cpu', note='既有 Q/K rotary bf16 InOut；CPU reference 按 Intent 显式 f32 中间运算后转回 bf16，与原 CUDA source 的 bf16 中间舍入不同；每次恢复 query/key，恢复不计时；双方计完整host调用。', owners=locals())
