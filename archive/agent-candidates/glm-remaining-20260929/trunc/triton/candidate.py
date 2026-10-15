from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_exact_kernel(input_ptr, output_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, values)


@triton.jit
def _copy_masked_kernel(input_ptr, output_ptr, num_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < num_elements
    values = tl.load(input_ptr + offsets, mask=valid)
    tl.store(output_ptr + offsets, values, mask=valid)


@triton.jit
def _trunc_f32_exact_kernel(input_ptr, output_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    truncated = tl.inline_asm_elementwise(
        asm="cvt.rzi.f32.f32 $0, $1;",
        constraints="=f,f",
        args=[values],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(output_ptr + offsets, truncated)


@triton.jit
def _trunc_f32_masked_kernel(input_ptr, output_ptr, num_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < num_elements
    values = tl.load(input_ptr + offsets, mask=valid)
    truncated = tl.inline_asm_elementwise(
        asm="cvt.rzi.f32.f32 $0, $1;",
        constraints="=f,f",
        args=[values],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(output_ptr + offsets, truncated, mask=valid)


_BLOCK_SIZE = 1024


def build(context):
    def trunc(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        output = torch.empty_like(input) if out is None else out
        num_elements = input.numel()
        exact = num_elements % _BLOCK_SIZE == 0

        if input.dtype == torch.float32:
            if exact:
                _trunc_f32_exact_kernel[(num_elements // _BLOCK_SIZE,)](
                    input,
                    output,
                    BLOCK=_BLOCK_SIZE,
                )
            else:
                _trunc_f32_masked_kernel[(triton.cdiv(num_elements, _BLOCK_SIZE),)](
                    input,
                    output,
                    num_elements,
                    BLOCK=_BLOCK_SIZE,
                )
        elif input.dtype.is_floating_point:
            raise TypeError(f"trunc is implemented only for float32 CUDA tensors, got {input.dtype}")
        else:
            if exact:
                _copy_exact_kernel[(num_elements // _BLOCK_SIZE,)](
                    input,
                    output,
                    BLOCK=_BLOCK_SIZE,
                )
            else:
                _copy_masked_kernel[(triton.cdiv(num_elements, _BLOCK_SIZE),)](
                    input,
                    output,
                    num_elements,
                    BLOCK=_BLOCK_SIZE,
                )

        return output

    return trunc
