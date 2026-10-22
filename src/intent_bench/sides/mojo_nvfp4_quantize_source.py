from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    (rows, columns) = (8192, 4096)
    x = torch.randn((rows, columns), dtype=torch.bfloat16)
    global_scale = torch.ones((1,), dtype=torch.float32)
    packed = torch.empty((rows, columns // 2), dtype=torch.uint8)
    (row_blocks, column_blocks) = (rows // 128, columns // 64)
    scales_shape = (row_blocks, column_blocks, 32, 4, 4)
    generated_scales = torch.empty(scales_shape, dtype=torch.uint8)
    source_packed = packed.clone()
    source_scales = torch.empty_like(generated_scales)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_nvfp4_quantize')

    def source_launch():
        runtime.nvfp4_quantize(x, global_scale, source_packed, source_scales)

    def source_outputs():
        return (source_packed, source_scales.view(-1))
    return checked_call(PreparedCall(source_launch, source_outputs), Tolerance(atol=0.0), cuda_graph=False, device_type='cpu', note='既有 N8192-K4096 BF16 NVFP4 group16；保留 E2M1 nibble packing、E4M3 per-block scale 和原 packed/scales 输出 ABI；双方计完整 host 调用，输出 storage 分配不计时。', owners=locals())
