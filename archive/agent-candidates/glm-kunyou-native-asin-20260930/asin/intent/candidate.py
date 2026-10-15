import torch
import intent
import intent.language as I

@intent.kernel
def asin_kernel(x: I.In[I.f16, (1048576,)],
                y: I.Out[I.f16, (1048576,)]):
    values = x[:]
    result = I.asin(values)
    y[:] = result

def build(context):
    artifact = context.compile("asin_kernel", asin_kernel)

    def wrapper(input_tensor, out=None):
        if out is None:
            output = torch.empty_like(input_tensor)
        else:
            output = out
        artifact(input_tensor, output)
        return output

    return wrapper

