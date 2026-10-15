from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.routing.mhc import mhc_gemm_rms_partial, mhc_gemm_rms_finalize
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    x = torch.randn((2048, 16384), dtype=torch.bfloat16)
    weight = torch.randn((16384, 24), dtype=torch.bfloat16)
    bias = torch.randn((24,), dtype=torch.bfloat16)
    partial = intent.compile(mhc_gemm_rms_partial, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options, constexprs={'P': 16})
    finalize = intent.compile(mhc_gemm_rms_finalize, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options, constexprs={'P': 16, 'STREAMS': 4, 'ALPHA_PRE': 1.0, 'ALPHA_POST': 1.0, 'ALPHA_RESIDUAL': 1.0})
    (generated, source) = ({}, {})

    def launch():
        (linear, square_sum) = partial.run(x, weight)
        generated['output'] = finalize.run(linear, square_sum, bias, x.shape[1])
    return Side(PreparedLaunch(launch, lambda : generated['output']), (Tolerance(atol=0.05, rtol=0.02), Tolerance(atol=0.005, rtol=0.001)), cuda_graph=False, device_type='cpu', note='既有 T2048/H4096/streams4/P16 bf16 MHC GEMM+RMS；partial/finalize 两 kernel 完整 host 调用，含中间及输出分配；PyTorch CPU reference，保留 split16、bf16 mixed、f32 rms 与原容差，单 NUMA 8 核。', owners=locals())
