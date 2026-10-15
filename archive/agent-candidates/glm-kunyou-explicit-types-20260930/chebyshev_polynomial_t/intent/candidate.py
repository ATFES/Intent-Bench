import torch
import intent
import intent.language as I

@intent.kernel
def chebyshev_kernel(input: I.In[I.f32, (1048576,)], n: I.i32,
                     out: I.Out[I.f32, (1048576,)]):
    N = input.shape[0]
    idx = I.domain(0, N)

    if n == 0:
        for i in I.parallel(idx):
            out[i] = I.cast(1.0, I.f32)
    elif n == 1:
        for i in I.parallel(idx):
            out[i] = input[i]
    else:
        two = I.cast(2.0, I.f32)
        t_prev = I.buffer((N,), I.f32)
        t_curr = I.buffer((N,), I.f32)

        for i in I.parallel(idx):
            t_prev[i] = I.cast(1.0, I.f32)
            t_curr[i] = input[i]

        for k in I.domain(2, n + 1):
            for i in I.parallel(idx):
                t_next_i = two * input[i] * t_curr[i] - t_prev[i]
                t_prev[i] = t_curr[i]
                t_curr[i] = t_next_i

        for i in I.parallel(idx):
            out[i] = t_curr[i]


def build(context):
    artifact = context.compile("chebyshev_polynomial_t", chebyshev_kernel)

    def wrapper(input: torch.Tensor, n: int, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        artifact(input, n, out)
        return out

    return wrapper

