from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.quantization.nvfp4 import nvfp4_quantize as nvfp4_quantize_definition
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    (rows, columns) = (8192, 4096)
    x = torch.randn((rows, columns), dtype=torch.bfloat16)
    global_scale = torch.ones((1,), dtype=torch.float32)
    packed = torch.empty((rows, columns // 2), dtype=torch.uint8)
    (row_blocks, column_blocks) = (rows // 128, columns // 64)
    scales_shape = (row_blocks, column_blocks, 32, 4, 4)
    generated_packed = packed.clone()
    generated_scales = torch.empty(scales_shape, dtype=torch.uint8)
    artifact = intent.compile(nvfp4_quantize_definition, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)

    def generated_launch():
        artifact.run(x, global_scale, generated_packed, generated_scales)

    def generated_outputs():
        return (generated_packed, generated_scales.view(-1))
    return Side(PreparedLaunch(generated_launch, generated_outputs), Tolerance(atol=0.0), cuda_graph=False, device_type='cpu', note='既有 N8192-K4096 BF16 NVFP4 group16；保留 E2M1 nibble packing、E4M3 per-block scale 和原 packed/scales 输出 ABI；双方计完整 host 调用，输出 storage 分配不计时。', owners=locals())
