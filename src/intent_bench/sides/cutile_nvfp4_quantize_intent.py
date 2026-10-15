from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.quantization.nvfp4 import nvfp4_quantize as nvfp4_kernel
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (rows, columns) = (8192, 4096)
    x = torch.randn((rows, columns), device='cuda', dtype=torch.bfloat16)
    global_scale = torch.ones((1,), device='cuda', dtype=torch.float32)
    packed = torch.empty((rows, columns // 2), device='cuda', dtype=torch.uint8)
    (row_blocks, column_blocks) = (rows // 128, columns // 64)
    scales = torch.empty((row_blocks, column_blocks, 32, 4, 4), device='cuda', dtype=torch.uint8)
    (_, generated_base) = compile_single(context, nvfp4_kernel, (x, global_scale, packed, scales))
    generated = PreparedLaunch(generated_base.launch, lambda : (packed, scales.view(-1)))
    return Side(generated, Tolerance(atol=0.0), cuda_graph=True, note='同算法、group16；CUDA Graph 重放下的算子 GPU 执行延迟，不含编译、调优与输出分配', owners=locals())
