from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    (batch, sequence, query_heads, kv_heads, dimension) = (2, 4096, 32, 8, 128)
    query_initial = torch.randn((batch, query_heads, sequence, dimension), dtype=torch.bfloat16)
    key_initial = torch.randn((batch, kv_heads, sequence, dimension), dtype=torch.bfloat16)
    cosine = torch.randn((1, sequence, dimension), dtype=torch.bfloat16)
    sine = torch.randn_like(cosine)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_rope_qk_bf16')
    source_query = query_initial.clone()
    source_key = key_initial.clone()

    def source_launch():
        runtime.rope_qk_bf16(source_query, source_key, cosine, sine)

    def restore_source():
        source_query.copy_(query_initial)
        source_key.copy_(key_initial)
    source = PreparedLaunch(source_launch, lambda : (source_query, source_key), prepare=restore_source)
    return Side(source, (Tolerance(atol=0.02, rtol=0.01), Tolerance(atol=0.02, rtol=0.01)), cuda_graph=False, device_type='cpu', note='既有 Q/K rotary bf16 InOut；CPU reference 按 Intent 显式 f32 中间运算后转回 bf16，与原 CUDA source 的 bf16 中间舍入不同；每次恢复 query/key，恢复不计时；双方计完整host调用。', owners=locals())
