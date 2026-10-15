import torch
import intent
import intent.language as I


@intent.kernel
def bn_relu_kernel(x: I.In[I.f32, ("N", "C", "H", "W")],
                   running_mean: I.In[I.f32, ("C",)],
                   running_var: I.In[I.f32, ("C",)],
                   bn_weight: I.In[I.f32, ("C",)],
                   bn_bias: I.In[I.f32, ("C",)],
                   y: I.Out[I.f32, ("N", "C", "H", "W")]):
    """Apply batch normalization (with running stats) and ReLU.
    
    BN formula: y = bn_weight * (x - running_mean) / sqrt(running_var + eps) + bn_bias
    ReLU: y = max(0, y)
    """
    N, C, H, W = x.shape
    for n in I.parallel(I.domain(0, N)):
        for c in I.parallel(I.domain(0, C)):
            for h in I.domain(0, H):
                for w in I.domain(0, W):
                    mean = running_mean[c]
                    var = running_var[c]
                    inv_std = I.rsqrt(var + I.cast(1e-05, I.f32))
                    
                    val = x[n, c, h, w]
                    normalized = (val - mean) * inv_std
                    bn_val = normalized * bn_weight[c] + bn_bias[c]
                    y[n, c, h, w] = I.maximum(bn_val, I.cast(0.0, I.f32))


def build(context):
    bn_relu_artifact = context.compile("bn_relu", bn_relu_kernel)
    
    def relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0,
                                dilation=1, groups=1, running_mean=None,
                                running_var=None, bn_weight=None, bn_bias=None,
                                training=False, momentum=0.1, eps=1e-05,
                                inplace=False):
        # Apply conv2d using torch (optimized library kernel)
        conv_out = torch.nn.functional.conv2d(
            input, weight, bias=bias, stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )
        
        # Apply batch norm with running stats and ReLU
        output = torch.empty_like(conv_out)
        bn_relu_artifact(conv_out, running_mean, running_var, bn_weight,
                         bn_bias, output)
        return output
    
    return relu_batch_norm_conv2d

