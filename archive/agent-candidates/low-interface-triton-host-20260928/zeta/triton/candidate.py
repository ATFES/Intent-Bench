import torch
import triton
import triton.language as tl


@triton.jit
def _zeta_kernel(x_ptr, q_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    q = tl.load(q_ptr + offs, mask=mask, other=0.0)

    # Sum a short prefix directly, then use the Euler-Maclaurin tail.
    s = x
    term_sum = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(32):
        y = q + float(k)
        term_sum += tl.exp(-s * tl.log(y))
    z = q + 32.0
    p = tl.exp(-s * tl.log(z))
    tail = tl.exp((1.0 - s) * tl.log(z)) / (s - 1.0)
    tail += 0.5 * p
    tail += (s / 12.0) * p / z

    # Bernoulli corrections B4, B6, B8, and B10.
    a = s * (s + 1.0) * (s + 2.0)
    tail -= (a / 720.0) * p / (z * z * z)
    a *= s + 3.0
    tail += (a * (s + 4.0) / 30240.0) * p / (z * z * z * z * z)
    a *= (s + 5.0) * (s + 6.0)
    tail -= (a / 1209600.0) * p / (z * z * z * z * z * z * z)
    a *= (s + 7.0) * (s + 8.0)
    tail += (a / 47900160.0) * p / (z * z * z * z * z * z * z * z * z)

    result = term_sum + tail
    q_integer = tl.floor(q) == q
    invalid_x = s < 1.0
    pole_x = s == 1.0
    bad_nonpositive_q = (q <= 0.0) & (~q_integer)
    result = tl.where(bad_nonpositive_q, float("nan"), result)
    result = tl.where((q <= 0.0) & q_integer & (s > 1.0), float("inf"), result)
    result = tl.where(invalid_x, float("nan"), result)
    result = tl.where(pole_x, float("inf"), result)
    tl.store(out_ptr + offs, result, mask=mask)


def build(context):
    def zeta(input: torch.Tensor, other: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        n = input.numel()
        _zeta_kernel[(triton.cdiv(n, 256),)](input, other, out, n, BLOCK=256)
        return out

    return zeta
