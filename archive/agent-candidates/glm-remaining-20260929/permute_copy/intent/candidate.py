import torch

import intent
import intent.language as I


@intent.kernel
def permute_copy_reverse_4d(
    input: I.In[
        I.f32,
        (32, 32, 32, 32),
    ],
    output: I.Out[
        I.f32,
        (32, 32, 32, 32),
        I.constraints(noalias=True),
    ],
) -> None:
    for output_0 in I.parallel(I.domain(0, 32)):
        for output_1 in I.parallel(I.domain(0, 32)):
            for output_2 in I.parallel(I.domain(0, 32)):
                for output_3 in I.parallel(I.domain(0, 32)):
                    output[output_0, output_1, output_2, output_3] = input[
                        output_3, output_2, output_1, output_0
                    ]


def build(context):
    kernel = context.compile("permute_copy_reverse_4d", permute_copy_reverse_4d)

    def permute_copy(input: torch.Tensor, dims: list = None) -> torch.Tensor:
        if dims is None:
            dims = [3, 2, 1, 0]
        if tuple(dims) != (3, 2, 1, 0):
            raise ValueError("compiled permutation does not match dims")

        output = torch.empty(
            (32, 32, 32, 32),
            dtype=torch.float32,
            device=input.device,
        )
        kernel(input, output)
        return output

    return permute_copy
