from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.contraction.block_scaled import scaled_fp8_splitk_matmul
from intent_bench.corpus._common.model import Tolerance
from intent_bench.corpus._common.model import PreparedLaunch

def prepare(context):
    lhs = torch.randn((4096, 4096), dtype=torch.bfloat16).to(torch.float8_e4m3fn).view(4096, 4, 4, 256)
    rhs = torch.randn((4096, 14336), dtype=torch.bfloat16).to(torch.float8_e4m3fn).view(4, 4, 256, 14336)
    artifact = intent.compile(scaled_fp8_splitk_matmul, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)

    def side(function):
        output = torch.zeros((4096, 14336), dtype=torch.float16)
        return PreparedLaunch(lambda : function(lhs, rhs, output, 1.0, 1.0), lambda : output, prepare=lambda : output.zero_())
    return Side(side(artifact.run), Tolerance(atol=0.5, rtol=0.01), cuda_graph=False, device_type='cpu', note='原4096x4096x14336/IT4/split4/Kblock256 FP8 GEMM；每split独立f32累加、缩放并转f16，再对InOut输出f16相加；双方输出预分配，每次清零不计时；PyTorch CPU reference采用递增split顺序，保留原容差。', owners=locals())
