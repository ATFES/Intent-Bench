from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (batch, sequence, heads, key_dimension, value_dimension) = (2, 2048, 8, 128, 128)
    query = torch.randn((batch, sequence, heads, key_dimension), device='cuda', dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn((batch, sequence, heads, value_dimension), device='cuda', dtype=torch.bfloat16) * 0.1
    gate = -torch.rand((batch, sequence, heads), device='cuda', dtype=torch.bfloat16) * 0.5
    beta = torch.sigmoid(torch.randn_like(gate))
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/scan/gated_delta_recurrent/recurrent_gated_delta_rule.py', 'recurrent_gated_delta', needs_utils=True)
    source = functional_launch(lambda : source_module.recurrent_gated_delta_rule(query, key, value, gate, beta, initial_state=None, output_final_state=True, compiler_timeout=context.compiler_timeout_seconds))
    return Side(source, (Tolerance(atol=0.1, rtol=0.05), Tolerance(atol=0.1, rtol=0.05)), cuda_graph=False, owners=locals())
