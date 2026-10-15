from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.ragged.grouped_gemm import ragged_grouped_gemm_backward_weight
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    left = torch.randn((3840, 4096), dtype=torch.float16)
    right = torch.randn_like(left)
    offsets = torch.tensor((0, 256, 768, 1792, 3840), dtype=torch.int32)
    generated = torch.empty((4, 4096, 4096), dtype=torch.float16)
    artifact = intent.compile(ragged_grouped_gemm_backward_weight, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    return Side(PreparedLaunch(lambda : artifact(left, right, offsets, generated), lambda : generated), Tolerance(atol=0.05, rtol=0.02), cuda_graph=False, device_type='cpu', note='既有 rows256/512/1024/2048、K4096/N4096、f16 grouped GEMM weight backward；f32 累加，完整分组调用；双方输出预分配，PyTorch CPU reference，单 NUMA 8 核，沿用原容差。', owners=locals())
