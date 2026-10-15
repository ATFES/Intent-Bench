import torch
import intent
import intent.language as I


@intent.kernel
def _conv2d_add_fixed(
    inp: I.In[I.f32, (16, 32, 32, 32)],
    weight: I.In[I.f32, (32, 32, 3, 3)],
    other: I.In[I.f32, (16, 32, 32, 32)],
    out: I.Out[I.f32, (16, 32, 32, 32)],
    alpha: I.f32,
):
    n_domain = I.domain(0, 16)
    oc_domain = I.domain(0, 32)
    oh_domain = I.domain(0, 32)
    ow_domain = I.domain(0, 32)
    ic_domain = I.domain(0, 32)
    kh_domain = I.domain(0, 3)
    kw_domain = I.domain(0, 3)

    # The six logical axes are N, OH, OW, IC, KH, KW.  Size-one
    # dimensions keep each coordinate relation explicit for broadcasting.
    n = I.reshape(I.indices(n_domain), (16, 1, 1, 1, 1, 1))
    oh = I.reshape(I.indices(oh_domain), (1, 32, 1, 1, 1, 1))
    ow = I.reshape(I.indices(ow_domain), (1, 1, 32, 1, 1, 1))
    ic = I.reshape(I.indices(ic_domain), (1, 1, 1, 32, 1, 1))
    kh = I.reshape(I.indices(kh_domain), (1, 1, 1, 1, 3, 1))
    kw = I.reshape(I.indices(kw_domain), (1, 1, 1, 1, 1, 3))

    ih = oh + kh - 1
    iw = ow + kw - 1
    valid_h = (ih >= 0)
    valid_h = I.select(valid_h, ih < 32, False)
    valid_w = (iw >= 0)
    valid_w = I.select(valid_w, iw < 32, False)
    valid = I.select(valid_h, valid_w, False)

    input_linear = I.reshape(inp, (16 * 32 * 32 * 32,))
    input_index = n * 32768 + ic * 1024 + ih * 32 + iw
    lhs = I.gather(
        input_linear,
        input_index,
        valid=valid,
        fill=I.cast(0.0, I.f32),
    )

    weight_linear = I.reshape(weight, (32 * 32 * 3 * 3,))
    oc_w = I.reshape(I.indices(oc_domain), (32, 1, 1, 1))
    ic_w = I.reshape(I.indices(ic_domain), (1, 32, 1, 1))
    kh_w = I.reshape(I.indices(kh_domain), (1, 1, 3, 1))
    kw_w = I.reshape(I.indices(kw_domain), (1, 1, 1, 3))
    weight_index = oc_w * 288 + ic_w * 9 + kh_w * 3 + kw_w
    rhs = I.gather(weight_linear, weight_index, fill=I.cast(0.0, I.f32))

    conv = I.contract(
        lhs,
        rhs,
        reduce=((3, 1), (4, 2), (5, 3)),
        batch=(),
        acc_dtype=I.f32,
    )

    other_nhwc = I.transpose(other, (0, 2, 3, 1))
    result_nhwc = conv + alpha * other_nhwc
    result = I.transpose(result_nhwc, (0, 3, 1, 2))
    out[n_domain, oc_domain, oh_domain, ow_domain] = result


def build(context):
    artifact = context.compile("conv2d_add_fixed", _conv2d_add_fixed, constexprs={})

    def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0,
                   dilation=1, groups=1, alpha=1, out=None):
        if bias is not None:
            raise NotImplementedError("the fixed profile has no bias")
        if other is None or not isinstance(other, torch.Tensor):
            raise NotImplementedError("the fixed profile uses a tensor other")
        if stride != 1 or padding != 1 or dilation != 1 or groups != 1:
            raise NotImplementedError("unsupported parameters outside the fixed profile")
        if out is None:
            out = torch.empty((16, 32, 32, 32), device=input.device, dtype=input.dtype)
        artifact(input, weight, other, out, float(alpha))
        return out

    return conv2d_add
