"""ATen compositions on the requested device, using the catalog's complete inputs."""
from functools import partial

from ..model import EnvironmentUnavailable, Implementation, PreparedCall
from ..implementations.prepared import attach_device_timer


def histogram(indices, bins):
    import torch
    keys = indices.to(torch.int64)
    valid = (keys >= 0) & (keys < bins)
    output = torch.zeros(bins + 1, dtype=torch.int32, device=keys.device)
    output.scatter_add_(0, torch.where(valid, keys, bins), torch.ones_like(keys, dtype=torch.int32))
    return output[:bins]


def q4_projection(records, x):
    import torch
    rows = x.reshape(-1, 256)
    extreme = rows.gather(1, rows.abs().argmax(1, keepdim=True))
    inverse = torch.where(extreme != 0, -127. / extreme, 0.)
    scale = torch.where(inverse != 0, 1. / inverse, 0.)
    codes = (rows * inverse).round().clamp(-128, 127).int().view(-1, 8, 32)
    fields = records[..., :4].contiguous().view(torch.float16).float()
    packed = records[..., 4:16].int()
    lo, hi, extra = packed.split(4, dim=-1)
    gains = torch.cat((lo & 63, (extra & 15) | ((lo >> 6) << 4)), -1)
    mins = torch.cat((hi & 63, (extra >> 4) | ((hi >> 6) << 4)), -1)
    qs = records[..., 16:].int().unflatten(-1, (4, 32))
    quant = torch.stack((qs & 15, qs >> 4), dim=-2).flatten(-3, -2)
    dot = (quant * codes).sum(-1, dtype=torch.int32)
    sum_a = codes.sum(-1, dtype=torch.int32)
    dot = (gains * dot).sum(-1, dtype=torch.int32).float()
    correction = (mins * sum_a).sum(-1, dtype=torch.int32).float()
    return ((fields[..., 0] * dot - fields[..., 1] * correction) * scale.T).sum(-1)


def raw_moment_norm(x, gain, offset, inverse_width, epsilon):
    center = x.sum(dim=-1, keepdim=True) * inverse_width
    second = x.square().sum(dim=-1, keepdim=True) * inverse_width
    return (x - center) * (second - center.square() + epsilon).rsqrt() * gain + offset


def softmax(x):
    v = x.float()
    w = (v - v.amax(-1, keepdim=True)).exp()
    return (w / w.sum(-1, keepdim=True)).to(x.dtype)


def weighted_rms(x, gain, inverse, epsilon):
    return (x * (x.square().sum(-1, keepdim=True) * inverse + epsilon).rsqrt()) * gain


def bucketed_norm_backward(x, dy, gain, mean, inv, inverse_width):
    import torch
    dx = torch.empty_like(x)
    buckets = 128
    dw = x.new_zeros((buckets, x.shape[1]))
    db = torch.zeros_like(dw)
    for offset in range(0, x.shape[0], buckets):
        rows = slice(offset, offset + buckets)
        norm = (x[rows].float() - mean[rows, None]) * inv[rows, None]
        grad = dy[rows].float()
        weighted = gain.float() * grad
        center = weighted.sum(-1, keepdim=True) * inverse_width
        projection = (norm * weighted).sum(-1, keepdim=True) * inverse_width
        dx[rows] = ((weighted - center - norm * projection) * inv[rows, None]).to(x.dtype)
        dw = (dw.float() + (grad * norm).to(x.dtype).float()).to(x.dtype)
        db = (db.float() + grad).to(x.dtype)
    return dx, dw.float().sum(0), db.float().sum(0)


def welford_norm(x, gain, bias, epsilon):
    import torch
    var, mean = torch.var_mean(x, dim=-1, keepdim=True, unbiased=False)
    return ((x - mean) * (var + epsilon).rsqrt()) * gain + bias


def log_softmax_kl(x, target):
    import torch
    peak = x.amax(-1, keepdim=True)
    lp = x - (peak + (x - peak).exp().sum(-1, keepdim=True).log())
    kl = torch.where(target > 0, target * (target.log() - lp), 0).sum(-1)
    return lp, kl


def cross_entropy(logits, targets, grad):
    import torch
    x = logits.float()
    peak = x.amax(-1, keepdim=True)
    weights = (x - peak).exp()
    mass = weights.sum(-1, keepdim=True)
    index = targets.long().view(-1, 1)
    loss = peak + mass.log() - x.gather(1, index)
    probabilities = weights * mass.reciprocal()
    probabilities.scatter_add_(1, index, -torch.ones_like(loss))
    return loss.squeeze(-1), (probabilities * grad[:, None]).to(logits.dtype)


def grouped_split_decode(q, keys, values, scale):
    import torch
    b, h, _, d = q.shape
    groups = keys.shape[1]
    query = q.float().reshape(b, groups, h // groups, d)
    partials, logs = [], []
    for k, v in zip(keys.split(256, dim=2), values.split(256, dim=2)):
        score = (query @ k.float().transpose(-2, -1)) * scale
        peak = score.amax(dim=-1, keepdim=True)
        w = (score - peak).exp()
        mass = w.sum(dim=-1, keepdim=True)
        partials.append(((w.to(q.dtype).float() @ v.float()) * mass.reciprocal()).to(q.dtype))
        logs.append((peak + mass.log()).squeeze(-1))
    split_weights = torch.stack(logs, dim=-1).softmax(dim=-1)
    out = (torch.stack(partials, dim=-2).float() * split_weights.unsqueeze(-1)).sum(dim=-2)
    return out.reshape(b, h, 1, values.shape[-1]).to(q.dtype)


def ordered_linear_attention(q, k, v):
    state = v.new_zeros((v.shape[0], k.shape[-1], v.shape[-1]))
    out = v.new_empty((*q.shape[:-1], v.shape[-1]))
    for t in range(q.shape[1]):
        state = state + k[:, t, :, None] * v[:, t, None, :]
        out[:, t] = (q[:, t, :, None] * state).sum(1)
    return out, state


def paged_gqa(q, keys, values, pages, table, lengths, cuts, scale):
    import math
    import torch
    out = q.new_empty((*q.shape[:2], values.shape[-1]))
    for b in range(q.shape[0]):
        query = q[b].float().reshape(keys.shape[2], -1, q.shape[-1])
        origin, length = int(pages[b]), int(lengths[b])
        pieces, logs = [], []
        bounds = cuts[b*8:b*8+9].tolist()
        for start, stop in zip(bounds, bounds[1:]):
            count = min(length - (start - origin) * keys.shape[1], (stop - start) * keys.shape[1])
            if count <= 0:
                continue
            ids = table[start:stop].long()
            k, v = (c[ids].flatten(0, 1)[:count].transpose(0, 1).float() for c in (keys, values))
            score = (query @ k.transpose(1, 2)) * (scale * math.log2(math.e))
            peak = score.amax(-1, keepdim=True)
            w = (score - peak).exp2()
            mass = w.sum(-1, keepdim=True)
            pieces.append((w.half().float() @ v) * mass.reciprocal())
            logs.append(peak + mass.log() * math.log2(math.e))
        if pieces:
            lse = torch.stack(logs)
            weights = (lse - lse.amax(0)).exp2()
            result = (torch.stack(pieces) * weights).sum(0) * weights.sum(0).reciprocal()
            out[b] = result.reshape_as(out[b]).half()
        else:
            out[b].fill_(float("nan"))
    return out


def paged_split_decode(q, qr, latent_cache, rope_cache, offsets, table, lengths, scale):
    import torch
    out = torch.empty_like(q)
    for batch in range(q.shape[0]):
        length = int(lengths[batch])
        if not length:
            out[batch].fill_(float("nan"))
            continue
        pages = table[offsets[batch]:offsets[batch + 1]].long()
        latent = latent_cache[pages].flatten(0, 1)[:length].transpose(0, 1).float()
        rope = rope_cache[pages].flatten(0, 1)[:length].transpose(0, 1).float()
        groups = latent.shape[0]
        query = q[batch].reshape(groups, -1, q.shape[-1]).float()
        pos = qr[batch].reshape(groups, -1, qr.shape[-1]).float()
        partial, log_mass = [], []
        pitch = (length + 7) // 8
        for key, rkey in zip(latent.split(pitch, dim=1), rope.split(pitch, dim=1)):
            scores = (query @ key.transpose(1, 2) + pos @ rkey.transpose(1, 2)) * scale
            peak = scores.amax(-1, keepdim=True)
            w = (scores - peak).exp()
            mass = w.sum(-1, keepdim=True)
            partial.append((w.half().float() @ key) * mass.reciprocal())
            log_mass.append(peak + mass.log())
        logs = torch.stack(log_mass, dim=0)
        weights = (logs - logs.amax(0)).exp()
        combined = (torch.stack(partial) * weights).sum(0) / weights.sum(0)
        out[batch] = combined.reshape_as(q[batch]).to(q.dtype)
    return out


def row_attention(kind, q, k, v, *parameters):
    import math
    import torch
    rows, tokens = q.shape[2], k.shape[2]
    if not tokens:
        return q.new_zeros((*q.shape[:-1], v.shape[-1]))
    axis = torch.arange(tokens, device=q.device)[None, :] - torch.arange(rows, device=q.device)[:, None]
    mask = axis > 0
    scale = parameters[-1] if kind != "window" else parameters[0]
    if kind == "window":
        mask = mask | (axis <= -parameters[1])
    out = q.new_empty((*q.shape[:-1], v.shape[-1]))
    group = q.shape[1] // k.shape[1]
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            raw = q[batch, head].float() @ k[batch, head // group].float().T
            if kind == "unscaled":
                raw = raw.masked_fill(mask, -float("inf"))
                weights = ((raw - raw.amax(-1, keepdim=True)) * scale).exp().masked_fill(mask, 0.)
            else:
                scores = raw * scale
                if kind == "alibi":
                    scores = scores + axis * parameters[0][head]
                scores = (scores * math.log2(math.e)).masked_fill(mask, -float("inf"))
                weights = (scores - scores.amax(-1, keepdim=True)).exp2().masked_fill(mask, 0.)
            mass = weights.sum(-1, keepdim=True)
            accumulated = weights.to(v.dtype).float() @ v[batch, head // group].float()
            out[batch, head] = accumulated * mass.reciprocal()
    return out


def expert_projection(x, offsets, routes, weights):
    top_k = routes.numel() // x.shape[0]
    out = x.new_empty((routes.numel(), weights.shape[1]))
    for e in range(weights.shape[0]):
        dest = routes[offsets[e]:offsets[e+1]].long()
        out[dest] = (x[dest // top_k].float() @ weights[e].float().T).to(x.dtype)
    return out.view(x.shape[0], top_k, -1)


def flash_backward(q, k, v, out, dy, log2_lse, scale):
    import math
    import torch
    # The catalog carries base-two LSE. Round P and dS before their
    # contractions; independent head tiles bound temporary storage.
    grads = [torch.empty_like(t) for t in (q, k, v)]
    causal = torch.ones((q.shape[-2], k.shape[-2]), device=q.device, dtype=torch.bool).tril()
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            query, key, value, saved, grad = (t[batch,head].float() for t in (q,k,v,out,dy))
            scores = (query @ key.T) * (scale * math.log2(math.e))
            p = torch.exp2(scores - log2_lse[batch,head,:,None]).masked_fill(~causal, 0.)
            ds = (p * ((grad @ value.T) - (saved * grad).sum(dim=-1,keepdim=True))).half().float()
            for dst, data in zip(grads,(ds @ key * scale, ds.T @ query * scale, p.half().float().T @ grad)):
                dst[batch,head].copy_(data)
    return tuple(grads)


def registrations(name):
    functions = {
        "corpus/weft/q4_k_projection": q4_projection,
        "corpus/mojo/layer_norm_backward": bucketed_norm_backward,
        "softmax": softmax,
        "weighted_rms_norm": weighted_rms,
        "group7/layer_norm_welford": welford_norm,
        "group7/log_softmax_kl": log_softmax_kl,
        "group7/cross_entropy": cross_entropy,
        "group7/alibi_attention": partial(row_attention, "alibi"),
        "group7/attention_unscaled_max": partial(row_attention, "unscaled"),
        "group7/sliding_window_attention": partial(row_attention, "window"),
        "corpus/triton/flaggems_histogram": lambda x: histogram(x, 256),
        "corpus/triton/fused_add_rms_norm": "fused_add_rms_norm",
        "corpus/mojo/dropout_residual_rms_norm": "dropout_residual_rms_norm",
        "corpus/mojo/qkv_projection": "qkv_projection",
        "corpus/mojo/selective_state_scan": "selective_state_scan",
        "corpus/mojo/rotary_embedding_equivalent_index": "rotary_embedding_flat",
        "corpus/cutile/grouped_flash_decode": grouped_split_decode,
        "corpus/triton/moe_expert_projection": expert_projection,
        "corpus/triton/flash_attention_backward": flash_backward,
        "corpus/mojo/paged_mla_decode_partials": paged_split_decode,
    }
    if name not in functions and name != "cpu/histogram":
        return {}

    def prepare(cpu, case, data, env):
        import torch
        from ..corpus.cpu.baselines.pytorch import cpu_runtime

        device = env.device
        if device == "auto":
            device = "cpu" if cpu else "cuda:0"
        if (device == "cpu") != cpu:
            raise EnvironmentUnavailable(f"reference does not target {device}")
        if device.startswith("mlu:"):
            import torch_mlu  # noqa: F401
        if name == "cpu/histogram":
            args = (data["indices"], case.parameters["columns"])
            function = lambda keys, bins: histogram(keys, bins).float()
        else:
            args = getattr(data, "arguments", data)
            function = functions[name]
            if isinstance(function, str):
                function = getattr(cpu_runtime, function)

        def move(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            if not hasattr(x, "shape"):
                return x
            if x.dtype.name == "bfloat16":
                return torch.from_numpy(x.view("uint16")).view(torch.bfloat16).to(device)
            return torch.as_tensor(x, device=device)

        args = tuple(move(x) for x in args)
        state = {}
        def run():
            state["output"] = function(*args)
        call = PreparedCall(run, lambda: state["output"], device=device, owners=args,
                            metadata={"library": ("NumPy Philox and ATen RMSNorm" if cpu and name.endswith("dropout_residual_rms_norm") else "ATen composition"), "catalog_workload": name})
        call = attach_device_timer(call)
        if device.startswith("cuda"):
            from ..workloads.programs import _tree
            call.result = lambda: _tree(state["output"], lambda x: x.cpu())
        return call

    if name in {"softmax", "weighted_rms_norm", "corpus/mojo/layer_norm_backward"} or name.startswith("group7/"):
        return {"aten:source": Implementation("aten:source", lambda case, data, env: prepare(env.device == "cpu" or env.device == "auto", case, data, env))}
    devices = (True, False)
    return {key: Implementation(key, partial(prepare, cpu))
            for cpu in devices for key in ["aten:cpu" if cpu else "aten:accelerator"]}
