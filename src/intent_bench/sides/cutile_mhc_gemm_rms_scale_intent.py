from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.routing.mhc import mhc_gemm_rms_finalize
from intent.examples.kernels.routing.mhc import mhc_gemm_rms_partial
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (tokens, hidden, streams) = (2048, 4096, 4)
    x = torch.randn((tokens, streams * hidden), device='cuda', dtype=torch.bfloat16)
    width = streams * (streams + 2)
    weight = torch.randn((streams * hidden, width), device='cuda', dtype=torch.bfloat16)
    bias = torch.randn((width,), device='cuda', dtype=torch.bfloat16)
    split_count = 16
    (_, partial) = compile_single(context, mhc_gemm_rms_partial, (x, weight), constexprs={'P': split_count})
    (partial_linear, partial_square_sum) = partial.result()
    (_, finalize) = compile_single(context, mhc_gemm_rms_finalize, (partial_linear, partial_square_sum, bias, x.shape[1]), constexprs={'P': split_count, 'STREAMS': streams, 'ALPHA_PRE': 1.0, 'ALPHA_POST': 1.0, 'ALPHA_RESIDUAL': 1.0})

    def generated_launch():
        partial.launch()
        finalize.launch()
    generated = PreparedCall(generated_launch, finalize.result)
    return checked_call(generated, (Tolerance(atol=0.05, rtol=0.02), Tolerance(atol=0.005, rtol=0.001)), cuda_graph=False, note='同算法；source 用乘倒数实现归一化', owners=locals())
