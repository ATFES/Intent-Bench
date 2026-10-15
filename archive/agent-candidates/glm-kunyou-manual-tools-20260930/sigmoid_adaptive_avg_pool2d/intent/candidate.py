import torch
import intent
import intent.language as I


@intent.kernel
def sigmoid_adaptive_avg_pool2d_kernel(
    x: I.In[I.f32, ("N", "C", "H", "W")],
    y: I.Out[I.f32, ("N", "C", "OH", "OW")],
):
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    OH = y.shape[2]
    OW = y.shape[3]

    for n in I.parallel(I.domain(0, N)):
        for c in I.parallel(I.domain(0, C)):
            for oh in I.parallel(I.domain(0, OH)):
                for ow in I.parallel(I.domain(0, OW)):
                    # Adaptive pooling window: start = floor(i*H/OH), end = ceil((i+1)*H/OH)
                    h0 = (oh * H) // OH
                    h1 = -(-((oh + 1) * H) // OH)
                    w0 = (ow * W) // OW
                    w1 = -(-((ow + 1) * W) // OW)

                    rows = I.domain(h0, h1)
                    cols = I.domain(w0, w1)
                    patch = x[n, c, rows, cols]
                    count = I.cast((h1 - h0) * (w1 - w0), I.f32)
                    total = I.reduce.sum(patch, axis=(0, 1), acc_dtype=I.f32)
                    avg = total / count
                    y[n, c, oh, ow] = I.sigmoid(avg)


def build(context):
    artifact = context.compile(
        "sigmoid_adaptive_avg_pool2d_kernel",
        sigmoid_adaptive_avg_pool2d_kernel,
    )

    def sigmoid_adaptive_avg_pool2d(input, output_size):
        if isinstance(output_size, int):
            oh = ow = output_size
        else:
            oh, ow = output_size
        n, c, h, w = input.shape
        out = torch.empty((n, c, oh, ow), dtype=torch.float32, device="cuda")
        artifact(input, out)
        return out

    return sigmoid_adaptive_avg_pool2d

