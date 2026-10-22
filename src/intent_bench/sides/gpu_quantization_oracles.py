"""Numerical definitions independent of native block-scaled GPU instructions."""
from ..corpus.cpu.baselines.pytorch import cpu_runtime
from ..implementations.context import Tolerance
from ..model import PreparedCall


def block_scaled(context):
    from .cutile_block_scaled_gemm_intent import inputs
    arguments = inputs(context)
    state = {}

    def launch():
        state['output'] = cpu_runtime.block_scaled_matmul(*arguments)

    return PreparedCall(launch, lambda: state['output'], device='cuda',
                        tolerance=Tolerance(atol=1.0, rtol=0.02),
                        metadata={'numerical_reference': 'decoded_block_scaled_matmul'})


def nvfp4(context):
    from .cutile_nvfp4_quantize_intent import inputs
    x, global_scale, packed, scales = inputs(context)

    def launch():
        cpu_runtime.nvfp4_quantize(x, global_scale, packed, scales)

    return PreparedCall(launch, lambda: (packed, scales.view(-1)), device='cuda',
                        tolerance=Tolerance(atol=0.0),
                        metadata={'numerical_reference': 'explicit_e2m1_rounding'})
