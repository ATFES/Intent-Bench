import triton
import triton.language as tl

@triton.jit
def traffic(x,sinks,S:tl.constexpr,WRITE:tl.constexpr,B:tl.constexpr):
    pid=tl.program_id(0)
    idx=(pid*B+tl.arange(0,B))*S
    if WRITE:
        tl.store(x+idx,1.0)
    else:
        v = tl.load(x+idx, volatile=True)
        tl.inline_asm_elementwise("", constraints="=f,f", args=[v], dtype=tl.float32,
                                  is_pure=False, pack=1)
@triton.jit
def empty(sink):
    tl.store(sink,0.)
