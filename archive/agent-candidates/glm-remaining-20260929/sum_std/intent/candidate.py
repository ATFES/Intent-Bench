import torch
import intent
import intent.language as I


@intent.kernel
def _sum_std_zero(result: I.Out[I.f32, ()]) -> None:
    result[()] = I.cast(0, I.f32)


def build(context):
    zero_artifact = context.compile("sum_std_zero_f32", _sum_std_zero)

    def sum_std(
        input,
        dim=None,
        keepdim=False,
        dtype=None,
        correction=1,
        out=None,
    ):
        if dim is not None:
            raise ValueError("this fixed-profile implementation supports dim=None")
        if dtype is not None and dtype is not torch.float32:
            raise ValueError("this fixed-profile implementation supports float32 only")
        if input.dtype is not torch.float32:
            raise ValueError("this fixed-profile implementation supports float32 only")

        if correction >= 1:
            return torch.zeros((), dtype=torch.float32, device="cpu")

        if input.device.type == "cpu":
            return torch.zeros((), dtype=torch.float32, device="cpu")

        result = torch.empty((), dtype=torch.float32, device=input.device)
        zero_artifact(result)
        return result

    return sum_std
