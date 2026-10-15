import torch
import intent
import intent.language as I


@intent.kernel
def im2col_kernel(
    input: I.In[I.f32, (1, 64, 64, 64)],
    output: I.Out[I.f32, (3844, 576)]
):
    """Transform input image to column matrix for convolution.
    Input shape: (1, 64, 64, 64)
    Output shape: (3844, 576) where 3844 = 62*62, 576 = 64*3*3
    
    Each row corresponds to one output position (oh, ow).
    Each column corresponds to one kernel element (c, kh, kw).
    """
    C, H, W = 64, 64, 64
    kH, kW = 3, 3
    H_out, W_out = 62, 62
    stride = 1
    pad_h, pad_w = 0, 0
    
    # Iterate over all output positions
    positions = I.domain(0, H_out * W_out)
    for pos in I.parallel(positions):
        oh = pos // W_out
        ow = pos % W_out
        
        # Iterate over all kernel elements
        k_elements = I.domain(0, C * kH * kW)
        for kidx in I.parallel(k_elements):
            c = kidx // (kH * kW)
            kh = (kidx % (kH * kW)) // kW
            kw = (kidx % (kH * kW)) % kW
            
            in_h = oh * stride + kh - pad_h
            in_w = ow * stride + kw - pad_w
            
            # Bounds check with padding
            valid_h = (in_h >= 0) & (in_h < H)
            valid_w = (in_w >= 0) & (in_w < W)
            valid = valid_h & valid_w
            
            value = I.select(valid, input[0, c, in_h, in_w], I.cast(0.0, I.f32))
            output[pos, kidx] = value


@intent.kernel
def conv_sigmoid_bias_kernel(
    im2col_out: I.In[I.f32, (3844, 576)],
    weight: I.In[I.f32, (128, 64, 3, 3)],
    bias: I.In[I.f32, (128,)],
    output: I.Out[I.f32, (1, 128, 62, 62)]
):
    """Compute convolution via matmul, add bias, apply sigmoid.
    
    im2col_out: (3844, 576) - im2col transformed input
    weight: (128, 64, 3, 3) - convolution filters
    bias: (128,) - bias tensor
    output: (1, 128, 62, 62) - final result after sigmoid
    """
    H_out, W_out, C_out = 62, 62, 128
    
    # Reshape weight for matmul: (128, 64, 3, 3) -> (128, 576)
    weight_mat = I.reshape(weight, (128, 576))
    
    # Convolution as matrix multiply: im2col @ weight.T
    # im2col_out: (3844, 576), weight_mat: (128, 576)
    # Result after transpose: (3844, 576) @ (576, 128) = (3844, 128)
    conv_result = I.matmul(im2col_out, weight_mat, transpose_rhs=True, acc_dtype=I.f32)
    
    # Reshape result: (3844, 128) -> (1, 128, 62, 62)
    out_b = I.domain(0, 1)
    out_c = I.domain(0, C_out)
    out_h = I.domain(0, H_out)
    out_w = I.domain(0, W_out)
    
    for b in I.parallel(out_b):
        for oc in I.parallel(out_c):
            for oh in I.parallel(out_h):
                for ow in I.parallel(out_w):
                    s_idx = oh * W_out + ow
                    value = conv_result[s_idx, oc]
                    value = value + bias[oc]
                    value = I.sigmoid(value)
                    output[b, oc, oh, ow] = value


@intent.kernel
def conv_sigmoid_nobias_kernel(
    im2col_out: I.In[I.f32, (3844, 576)],
    weight: I.In[I.f32, (128, 64, 3, 3)],
    output: I.Out[I.f32, (1, 128, 62, 62)]
):
    """Compute convolution via matmul, apply sigmoid (no bias).
    
    im2col_out: (3844, 576) - im2col transformed input
    weight: (128, 64, 3, 3) - convolution filters
    output: (1, 128, 62, 62) - final result after sigmoid
    """
    H_out, W_out, C_out = 62, 62, 128
    
    # Reshape weight for matmul: (128, 64, 3, 3) -> (128, 576)
    weight_mat = I.reshape(weight, (128, 576))
    
    # Convolution as matrix multiply: im2col @ weight.T
    conv_result = I.matmul(im2col_out, weight_mat, transpose_rhs=True, acc_dtype=I.f32)
    
    # Reshape result: (3844, 128) -> (1, 128, 62, 62)
    out_b = I.domain(0, 1)
    out_c = I.domain(0, C_out)
    out_h = I.domain(0, H_out)
    out_w = I.domain(0, W_out)
    
    for b in I.parallel(out_b):
        for oc in I.parallel(out_c):
            for oh in I.parallel(out_h):
                for ow in I.parallel(out_w):
                    s_idx = oh * W_out + ow
                    value = conv_result[s_idx, oc]
                    value = I.sigmoid(value)
                    output[b, oc, oh, ow] = value


def build(context):
    im2col_art = context.compile("im2col", im2col_kernel)
    conv_bias_art = context.compile("conv_sigmoid_bias", conv_sigmoid_bias_kernel)
    conv_nobias_art = context.compile("conv_sigmoid_nobias", conv_sigmoid_nobias_kernel)
    
    def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
        # Fixed profile validation
        assert input.shape == (1, 64, 64, 64), f"Expected (1, 64, 64, 64), got {input.shape}"
        assert weight.shape == (128, 64, 3, 3), f"Expected (128, 64, 3, 3), got {weight.shape}"
        assert stride == 1, f"Expected stride=1, got {stride}"
        assert padding == 0, f"Expected padding=0, got {padding}"
        assert dilation == 1, f"Expected dilation=1, got {dilation}"
        assert groups == 1, f"Expected groups=1, got {groups}"
        
        # Allocate intermediate and output tensors
        im2col_out = torch.empty((3844, 576), dtype=torch.float32, device='cuda')
        
        if out is None:
            out = torch.empty((1, 128, 62, 62), dtype=torch.float32, device='cuda')
        
        # Stage 1: im2col transformation
        im2col_art(input, im2col_out)
        
        # Stage 2: convolution + bias + sigmoid (or without bias)
        if bias is not None:
            conv_bias_art(im2col_out, weight, bias, out)
        else:
            conv_nobias_art(im2col_out, weight, out)
        
        return out
    
    return sigmoid_conv2d

