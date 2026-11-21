import triton
import triton.language as tl

@triton.jit
def traffic(x,sinks,S:tl.constexpr,WRITE:tl.constexpr,B:tl.constexpr):
    pid=tl.program_id(0)
    idx=(pid*B+tl.arange(0,B))*S
    if WRITE:
        tl.store(x+idx,1.0)
    else:
        tl.store(sinks+pid,tl.sum(tl.load(x+idx),0))
@triton.jit
def empty(sink):
    tl.store(sink,0.)
