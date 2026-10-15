from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, query_heads, kv_heads) = (8, 128, 1)
    (sequence, latent_dimension, rope_dimension) = (8192, 512, 64)
    key_dimension = latent_dimension + rope_dimension
    (page_size, splits) = (16, 8)
    pages_per_sequence = sequence // page_size
    pages = batch * pages_per_sequence
    q_latent = torch.randn((batch, query_heads, latent_dimension), device='cuda', dtype=torch.float16)
    q_rope = torch.randn((batch, query_heads, rope_dimension), device='cuda', dtype=torch.float16)
    latent_cache = torch.randn((pages, page_size, kv_heads, latent_dimension), device='cuda', dtype=torch.float16)
    rope_cache = torch.randn((pages, page_size, kv_heads, rope_dimension), device='cuda', dtype=torch.float16)
    page_indices = torch.arange(pages, device='cuda', dtype=torch.int32)
    page_table = page_indices.view(batch, pages_per_sequence)
    lengths = torch.full((batch,), sequence, device='cuda', dtype=torch.int32)
    scale = key_dimension ** (-0.5)
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/vllm/attention/paged_decode/paged_mla_decode_runtime.py', 'intent_v2_triton_paged_mla_runtime')
    source_module = runtime.RUNTIME.load_source(context.project_root / 'corpus/gpu/baselines/triton/vllm/attention/paged_decode/triton_decode_attention.py', 'intent_v2_triton_paged_mla_source')
    q = torch.cat((q_latent, q_rope), dim=-1)
    key = torch.cat((latent_cache, rope_cache), dim=-1)
    source_output = torch.empty((batch, query_heads, latent_dimension), device='cuda', dtype=torch.float16)
    source_lse = torch.empty((batch, query_heads), device='cuda', dtype=torch.float32)
    workspace = torch.empty((batch, query_heads, splits, latent_dimension + 1), device='cuda', dtype=torch.float32)

    def launch():
        source_module.decode_attention_fwd(q, key, latent_cache, source_output, source_lse, page_table, lengths, workspace, splits, scale, page_size=page_size, is_mla=True)
    launch()
    source = PreparedLaunch(launch=launch, outputs=lambda : source_output)
    return Side(source, Tolerance(atol=0.05, rtol=0.05), cuda_graph=False, note='同算法；source 使用 packed KV/page-table ABI 并写辅助 LSE；输入布局适配在计时前完成，完整 source 调用含 scale 分配', owners=locals())
