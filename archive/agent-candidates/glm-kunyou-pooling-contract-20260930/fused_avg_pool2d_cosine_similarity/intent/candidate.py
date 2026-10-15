import torch
import intent
import intent.language as I


@intent.kernel
def cosine_sim_kernel(x1: I.In[I.f32, (16, 64, 32, 32)],
                      x2: I.In[I.f32, (16, 64, 32, 32)],
                      eps: I.Constexpr[float],
                      cos_sim: I.Out[I.f32, (16, 1, 32, 32)]):
    B, C, H, W = 16, 64, 32, 32
    
    # Reshape to (B, H, W, C) for contraction over channel
    x1_reshaped = I.reshape(x1, (B, H, W, C))
    x2_reshaped = I.reshape(x2, (B, H, W, C))
    
    # Dot product along channel axis: sum over last axis of x1 * x2
    # Contract: lhs[b, h, w, c] * rhs[b, h, w, c] -> sum over c
    # reduce=((3, 3),) contracts lhs axis 3 with rhs axis 3
    # batch=((0, 0), (1, 1), (2, 2)) keeps batch and spatial dims
    dot_prod = I.contract(
        x1_reshaped,
        x2_reshaped,
        reduce=((3, 3),),
        batch=((0, 0), (1, 1), (2, 2)),
        acc_dtype=I.f32
    )
    
    # Norm of x1 along channel: sqrt(sum(x1^2, axis=1))
    x1_sq = x1 * x1
    norm_x1_sq = I.reduce.sum(x1_sq, axis=1, acc_dtype=I.f32)
    norm_x1 = I.sqrt(norm_x1_sq)
    
    # Norm of x2 along channel
    x2_sq = x2 * x2
    norm_x2_sq = I.reduce.sum(x2_sq, axis=1, acc_dtype=I.f32)
    norm_x2 = I.sqrt(norm_x2_sq)
    
    # Denominator: max(norm_x1, eps) * max(norm_x2, eps)
    norm_x1_safe = I.maximum(norm_x1, I.cast(eps, I.f32))
    norm_x2_safe = I.maximum(norm_x2, I.cast(eps, I.f32))
    denominator = norm_x1_safe * norm_x2_safe
    
    # Cosine similarity
    cos_sim_flat = dot_prod / denominator
    
    # Reshape to (B, 1, H, W) - insert singleton channel dimension
    cos_sim[:, :, :, :] = I.reshape(cos_sim_flat, (B, 1, H, W))


@intent.kernel
def pool2d_avg_kernel(cos_sim: I.In[I.f32, (16, 1, 32, 32)],
                      kernel_size: I.Constexpr[int],
                      stride: I.Constexpr[int],
                      padding: I.Constexpr[int],
                      out: I.Out[I.f32, (16, 1, 16, 16)]):
    B, C, H, W = 16, 1, 32, 32
    
    out_H = (H + 2 * padding - kernel_size) // stride + 1
    out_W = (W + 2 * padding - kernel_size) // stride + 1
    
    count_val = kernel_size * kernel_size
    
    for ph in I.indices(I.domain(0, out_H)):
        for pw in I.indices(I.domain(0, out_W)):
            h_start = ph * stride - padding
            w_start = pw * stride - padding
            
            accum = I.full((B, C), 0.0, I.f32)
            
            for kh in I.indices(I.domain(0, kernel_size)):
                for kw in I.indices(I.domain(0, kernel_size)):
                    ih = h_start + kh
                    iw = w_start + kw
                    
                    valid_h = (ih >= 0) & (ih < H)
                    valid_w = (iw >= 0) & (iw < W)
                    valid = valid_h & valid_w
                    
                    accum = accum + I.select(valid, cos_sim[:, :, ih, iw], I.zeros((B, C), I.f32))
            
            out[:, :, ph, pw] = accum / count_val


def build(context):
    cos_sim_artifact = context.compile("cosine_sim_kernel", cosine_sim_kernel, constexprs={'eps': 1e-08})
    pool_artifact = context.compile("pool2d_avg_kernel", pool2d_avg_kernel, constexprs={'kernel_size': 3, 'stride': 2, 'padding': 1})
    
    def fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size, stride=None, padding=0, eps=1e-08):
        B, C, H, W = x1.shape
        
        # Compute cosine similarity along channel axis 1
        # Input: (B, C, H, W) -> cosine similarity -> (B, H, W) -> (B, 1, H, W)
        cos_sim = torch.empty((B, 1, H, W), device=x1.device, dtype=x1.dtype)
        cos_sim_artifact(x1, x2, eps, cos_sim)
        
        # Apply 2D average pooling
        if stride is None:
            stride = kernel_size
        
        out_H = (H + 2 * padding - kernel_size) // stride + 1
        out_W = (W + 2 * padding - kernel_size) // stride + 1
        
        out = torch.empty((B, 1, out_H, out_W), device=x1.device, dtype=x1.dtype)
        pool_artifact(cos_sim, kernel_size, stride, padding, out)
        
        return out
    
    return fused_avg_pool2d_cosine_similarity

