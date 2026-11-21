"""GPU comparison with the established weighted RMSNorm author shape and scalar ABI."""

import numpy as np
from ..model import Case, Implementation, Workload, PreparedCall
from ..implementations.prepared import intent_call
from ..implementations.intent import make_kernel


def inputs(case, seed):
    p = case.parameters
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((p["rows"], p["columns"]), dtype=np.float32),
        rng.standard_normal(p["columns"], dtype=np.float32),
        1.0 / p["columns"],
        1e-6,
    )


def tensors(data, device):
    import torch

    return tuple(
        torch.from_numpy(x).to(device) if isinstance(x, np.ndarray) else x for x in data
    )


def reference(case, data):
    import torch

    x, w, inv, eps = tensors(data, "cpu")
    return ((x * torch.rsqrt((x * x).sum(-1, keepdim=True) * inv + eps)) * w).numpy()


def build(name):
    def prepare(impl, case, data, env):
        args = tensors(data, env.device)
        x, w, inv, eps = args
        if impl == "torch":
            import torch
            import torch.nn.functional as F

            # The declared case binds inv=1/N, as required by native RMSNorm.
            if inv != 1 / x.size(-1):
                raise ValueError("native RMSNorm requires inverse width")
            state = {}

            def launch():
                state["out"] = F.rms_norm(x, (x.size(-1),), w, eps)

            return PreparedCall(launch, lambda: state["out"], device=env.device)
        if impl == "intent:triton":
            from intent.examples.kernels.normalization.rms_norm import weighted_rms_norm

            return intent_call(
                make_kernel(weighted_rms_norm, env, "triton"), args, device=env.device
            )
        from intentdsl.examples import rms_norm
        from ..implementations.intent_v2 import prepare as prepare_v2

        return prepare_v2(rms_norm, args, env)

    impls = {
        k: Implementation(k, lambda c, d, e, k=k: prepare(k, c, d, e))
        for k in ("torch", "intent:triton", "intent-v2:triton")
    }
    return Workload(
        name,
        [Case("8192x4096-f32", dict(rows=8192, columns=4096))],
        inputs,
        impls,
        oracle=reference,
        atol=5e-5,
        rtol=0,
        family="normalization",
    )
