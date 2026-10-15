from __future__ import annotations
from intent_bench.sides import Side
import cuda.tile as ct
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (rows, columns) = (8192, 4096)
    x = torch.randn((rows, columns), device='cuda', dtype=torch.bfloat16)
    global_scale = torch.ones((1,), device='cuda', dtype=torch.float32)
    packed = torch.empty((rows, columns // 2), device='cuda', dtype=torch.uint8)
    (row_blocks, column_blocks) = (rows // 128, columns // 64)
    scales = torch.empty((row_blocks, column_blocks, 32, 4, 4), device='cuda', dtype=torch.uint8)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/quantization/nvfp4/nvfp4_quantize.py', 'nvfp4_quantize')
    source_packed = torch.empty_like(packed)
    source_scales = torch.empty_like(scales).view(-1)
    source_kernel = source_module._make_nvfp4_kernel(occupancy=6)

    def launch_source():
        ct.launch(torch.cuda.current_stream(), (column_blocks, row_blocks, 1), source_kernel, (x, source_scales, source_packed, global_scale, column_blocks, row_blocks, columns, rows, False, False))
        return (source_packed, source_scales)
    return Side(functional_launch(launch_source), Tolerance(atol=0.0), cuda_graph=True, note='同算法、group16；CUDA Graph 重放下的算子 GPU 执行延迟，不含编译、调优与输出分配', owners=locals())
