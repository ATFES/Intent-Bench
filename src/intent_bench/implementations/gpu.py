from ..model import PreparedCall


def prepare(name, provider, x):
    import torch
    output = torch.empty_like(x)
    if provider == "triton":
        from . import triton_ops
        import triton
        kernel = getattr(triton_ops, name)
        run = lambda: kernel[(x.shape[0],)](x, output, x.shape[1], x.stride(0), BLOCK=triton.next_power_of_2(x.shape[1]))
    else:
        import cuda.tile as ct
        from . import cutile_ops
        kernel = getattr(cutile_ops, name)
        block = 1 << (x.shape[1] - 1).bit_length()
        run = lambda: ct.launch(torch.cuda.current_stream(x.device), (x.shape[0],), kernel, (x, output, block))
    def host_run():
        nonlocal output
        output=torch.empty_like(x)
        run()
    return PreparedCall(run, lambda: output, device=str(x.device),host_run=host_run)
