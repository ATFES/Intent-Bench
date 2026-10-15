import torch
import triton
import triton.language as tl


@triton.jit
def _fused_cholesky_solve(A, b, L, y, out):
    # The factorization and substitutions are globally dependent by row.  A
    # single persistent program keeps that dependency explicit while avoiding
    # 256 separate launches.
    if tl.program_id(0) == 0:
        for i in range(256):
            for j in range(i + 1):
                acc = tl.load(A + i * 256 + j).to(tl.float32)
                for q in range(j):
                    acc -= tl.load(L + i * 256 + q) * tl.load(L + j * 256 + q)
                if i == j:
                    value = tl.sqrt(acc)
                else:
                    value = acc / tl.load(L + j * 256 + j)
                tl.store(L + i * 256 + j, value)

        for i in range(256):
            acc = tl.load(b + i)
            for q in range(i):
                acc -= tl.load(L + i * 256 + q) * tl.load(y + q)
            value = acc / tl.load(L + i * 256 + i)
            tl.store(y + i, value)

        for i in range(255, -1, -1):
            acc = tl.load(y + i)
            for q in range(i + 1, 256):
                acc -= tl.load(L + q * 256 + i) * tl.load(out + q)
            value = acc / tl.load(L + i * 256 + i)
            tl.store(out + i, value)


def build(context):
    def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        L = torch.empty((256, 256), device=A.device, dtype=torch.float32)
        y = torch.empty((256,), device=A.device, dtype=torch.float32)
        out = torch.empty_like(b)
        _fused_cholesky_solve[(1,)](A, b, L, y, out)
        return out

    return fused_cholesky_solve
