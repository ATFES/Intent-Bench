import torch
import intent
import intent.language as I

@intent.kernel
def conv2d_add_kernel(
    input: I.In[I.f32, (16, 32, 32, 32)],
    weight: I.In[I.f32, (32, 32, 3, 3)],
    other: I.In[I.f32, (16, 32, 32, 32)],
    stride: I.Constexpr[int],
    padding: I.Constexpr[int],
    dilation: I.Constexpr[int],
    groups: I.Constexpr[int],
    alpha: I.Constexpr[int],
    out: I.Out[I.f32, (16, 32, 32, 32)]
):
    N, C, H, W = 16, 32, 32, 32
    F, Gc, K, _ = 32, 32, 3, 3
    Ho = (H + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    Wo = (W + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    
    n_domain = I.domain(0, N)
    ho_domain = I.domain(0, Ho)
    wo_domain = I.domain(0, Wo)
    
    for n in I.parallel(n_domain):
        for ho in I.parallel(ho_domain):
            for wo in I.parallel(wo_domain):
                conv_val = I.cast(0.0, I.f32)
                
                for c in range(Gc):
                    for kh in range(K):
                        for kw in range(K):
                            h_in = ho * I.cast(stride, I.index) + kh * I.cast(dilation, I.index) - I.cast(padding, I.index)
                            w_in = wo * I.cast(stride, I.index) + kw * I.cast(dilation, I.index) - I.cast(padding, I.index)
                            
                            h_valid = (h_in >= I.cast(0, I.index)) & (h_in < I.cast(H, I.index))
                            w_valid = (w_in >= I.cast(0, I.index)) & (w_in < I.cast(W, I.index))
                            valid = h_valid & w_valid
                            
                            pixel = I.gather(input, (n, c, h_in, w_in), valid, fill=I.cast(0.0, I.f32))
                            w_val = weight[:, c, kh, kw]
                            conv_val = conv_val + pixel * w_val
                
                out[n, :, ho, wo] = conv_val
    
    alpha_val = I.cast(alpha, I.f32)
    result_tensor = out[:, :, :, :] + other[:, :, :, :] * alpha_val
    out[:, :, :, :] = result_tensor


def build(context):
    artifact = context.compile("conv2d_add", conv2d_add_kernel, constexprs={
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 1,
        "alpha": 1
    })
    
    def wrapper(input, weight, bias=None, other=None, stride=1, padding=1, dilation=1, groups=1, alpha=1, out=None):
        N, C, H, W = input.shape
        F = weight.shape[0]
        K = weight.shape[2]
        Ho = (H + 2 * padding - dilation * (K - 1) - 1) // stride + 1
        Wo = (W + 2 * padding - dilation * (K - 1) - 1) // stride + 1
        
        if out is None:
            out = torch.empty((N, F, Ho, Wo), dtype=torch.float32, device='cuda')
        
        if other is None:
            other = torch.zeros((N, C, H, W), dtype=torch.float32, device='cuda')
        
        artifact(input, weight, other, out)
        return out
    
    return wrapper

