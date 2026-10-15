import triton
import triton.language as tl


@triton.jit
def softmax(x, y, columns: tl.constexpr, stride: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    values = tl.load(x + row * stride + col, col < columns, other=-float("inf")).to(tl.float32)
    numer = tl.exp(values - tl.max(values, 0))
    tl.store(y + row * columns + col, numer / tl.sum(numer, 0), col < columns)


@triton.jit
def relu(x, y, columns: tl.constexpr, stride: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    values = tl.load(x + row * stride + col, col < columns, other=0)
    tl.store(y + row * columns + col, tl.maximum(values, 0), col < columns)
