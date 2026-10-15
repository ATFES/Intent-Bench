import torch
import triton
import triton.language as tl


@triton.jit
def _zeta_kernel(x_ptr, q_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    q = tl.load(q_ptr + offsets, mask=mask, other=0.0)

    # Shift the Euler-Maclaurin expansion away from the singular endpoint.
    # Eight explicit terms keep the remainder small for the resulting q >= 8.
    s = x
    qp = q
    partial = tl.zeros_like(s)
    for _ in range(8):
        partial += tl.exp(-s * tl.log(qp))
        qp += 1.0

    inv = tl.exp(-s * tl.log(qp))
    result = partial + inv / (s - 1.0) + 0.5 * inv
    rising = s
    power = s + 1.0
    # Bernoulli corrections B2/2!, B4/4!, ..., B16/16!.
    coeffs = (1.0 / 12.0, -1.0 / 720.0, 1.0 / 30240.0,
              -1.0 / 1209600.0, 1.0 / 47900160.0, -691.0 / 1307674368000.0,
              1.0 / 74724249600.0, -3617.0 / 10670622842880000.0)
    for k in range(8):
        result += coeffs[k] * rising * tl.exp(-power * tl.log(qp))
        rising *= (s + 2.0 * k + 1.0) * (s + 2.0 * k)
        power += 2.0

    nan = float('nan')
    inf = float('inf')
    x_is_one = s == 1.0
    x_below_one = s < 1.0
    q_nonpositive = q <= 0.0
    q_is_integer = q == tl.floor(q)
    x_is_integer = s == tl.floor(s)
    result = tl.where(x_below_one, nan, result)
    result = tl.where(x_is_one, inf, result)
    result = tl.where((s > 1.0) & q_nonpositive & q_is_integer, inf, result)
    result = tl.where((s > 1.0) & q_nonpositive & (~q_is_integer) & (~x_is_integer), nan, result)
    tl.store(out_ptr + offsets, result, mask=mask)


def build(context):
    compiled = context.compile("hurwitz_zeta", _zeta_kernel)

    def zeta(input: torch.Tensor, other: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        compiled(input, other, output, n_elements, BLOCK=256)
        return output

    return zeta
