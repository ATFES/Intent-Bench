"""Bind the CNNL softmax and weighted RMSNorm library entries to one queue."""

import ctypes as C


class RowOperator:
    def __init__(self, name, args, stream):
        import torch

        self.lib = C.CDLL("libcnnl.so")
        self.handle = C.c_void_p()
        self.descs = []
        self.owners = list(args)
        self.check("cnnlCreate", C.byref(self.handle))
        try:
            self.check("cnnlSetQueue", self.handle, C.c_void_p(stream.mlu_stream))
            x = args[0]
            self.out = torch.empty_like(x)

            def desc(shape, dtype):
                d = C.c_void_p()
                self.check("cnnlCreateTensorDescriptor", C.byref(d))
                self.descs.append(d)
                code = {torch.float16: 1, torch.float32: 2}[dtype]
                self.check(
                    "cnnlSetTensorDescriptor",
                    d,
                    4,
                    code,
                    len(shape),
                    (C.c_int * len(shape))(*shape),
                )
                return d

            ptr = lambda v: C.c_void_p(v.data_ptr())
            m, n = x.shape
            if name == "softmax":
                d = desc((m, 1, n), x.dtype)
                self.api = "cnnlSoftmaxForward"
                self.args = (self.handle, 1, 2, None, d, ptr(x), None, d, ptr(self.out))
            else:
                _, gain, inv, eps = args
                if inv != 1 / n:
                    raise ValueError(
                        "CNNL RMSNorm requires reciprocal normalization width"
                    )
                d, scale, stats = (
                    desc((m, n), x.dtype),
                    desc((n,), gain.dtype),
                    desc((m,), torch.float32),
                )
                size = C.c_size_t()
                self.check(
                    "cnnlGetRmsNormOpWorkspaceSize", self.handle, 1, d, C.byref(size)
                )
                workspace = torch.empty(size.value, dtype=torch.uint8, device=x.device)
                rms = torch.empty(m, dtype=torch.float32, device=x.device)
                self.owners.extend((workspace, rms))
                self.api = "cnnlRmsNormForward"
                self.args = (
                    self.handle,
                    1,
                    d,
                    ptr(x),
                    scale,
                    ptr(gain),
                    None,
                    C.c_float(eps),
                    ptr(workspace),
                    size,
                    d,
                    ptr(self.out),
                    stats,
                    ptr(rms),
                )
        except BaseException:
            self.close()
            raise

    def check(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            self.lib.cnnlGetErrorString.restype = C.c_char_p
            raise RuntimeError(
                f"{name}: {self.lib.cnnlGetErrorString(status).decode()} ({status})"
            )

    def launch(self):
        self.check(self.api, *self.args)

    def close(self):
        for d in self.descs:
            self.lib.cnnlDestroyTensorDescriptor(d)
        self.lib.cnnlDestroy(self.handle)
