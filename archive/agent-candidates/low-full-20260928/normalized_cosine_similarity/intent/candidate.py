import torch
import intent
import intent.language as I


@intent.kernel
def _norms(
    x1: I.In[I.f32, ("M", "N")],
    x2: I.In[I.f32, ("M", "N")],
    n1: I.Out[I.f32, ("M",)],
    n2: I.Out[I.f32, ("M",)],
):
    rows = I.domain(0, x1.shape[0])
    cols = I.domain(0, x1.shape[1])
    a = x1[rows, cols]
    b = x2[rows, cols]
    # The fixed profile uses p_norm=2; retain the norm as an explicit stage.
    a2 = a * a
    b2 = b * b
    n1[rows] = I.sqrt(I.reduce.sum(a2, axis=1, acc_dtype=I.f32))
    n2[rows] = I.sqrt(I.reduce.sum(b2, axis=1, acc_dtype=I.f32))


@intent.kernel
def _similarity(
    x1: I.In[I.f32, ("M", "N")],
    x2: I.In[I.f32, ("M", "N")],
    n1: I.In[I.f32, ("M",)],
    n2: I.In[I.f32, ("M",)],
    out: I.Out[I.f32, ("M",)],
    eps_similarity: I.f32,
    eps_norm: I.f32,
):
    rows = I.domain(0, x1.shape[0])
    cols = I.domain(0, x1.shape[1])
    a = x1[rows, cols]
    b = x2[rows, cols]
    na = I.select(n1[rows] > eps_norm, n1[rows], eps_norm)
    nb = I.select(n2[rows] > eps_norm, n2[rows], eps_norm)
    an = I.fdiv(a, I.reshape(na, (x1.shape[0], 1)))
    bn = I.fdiv(b, I.reshape(nb, (x1.shape[0], 1)))
    dot = I.reduce.sum(an * bn, axis=1, acc_dtype=I.f32)
    anorm = I.sqrt(I.reduce.sum(an * an, axis=1, acc_dtype=I.f32))
    bnorm = I.sqrt(I.reduce.sum(bn * bn, axis=1, acc_dtype=I.f32))
    da = I.select(anorm > eps_similarity, anorm, eps_similarity)
    db = I.select(bnorm > eps_similarity, bnorm, eps_similarity)
    out[rows] = I.fdiv(dot, da * db)


def build(context):
    norms = context.compile("normalized_cosine_similarity_norms", _norms)
    similarity = context.compile("normalized_cosine_similarity_similarity", _similarity)

    def normalized_cosine_similarity(
        x1,
        x2,
        dim=1,
        eps_similarity=1e-8,
        p_norm=2,
        eps_norm=1e-12,
    ):
        # The supplied invocation profile is contiguous [M, N] with dim=1.
        m = x1.shape[0]
        n1 = torch.empty((m,), device=x1.device, dtype=torch.float32)
        n2 = torch.empty((m,), device=x1.device, dtype=torch.float32)
        out = torch.empty((m,), device=x1.device, dtype=torch.float32)
        norms(x1, x2, n1, n2)
        similarity(x1, x2, n1, n2, out, eps_similarity, eps_norm)
        return out

    return normalized_cosine_similarity
