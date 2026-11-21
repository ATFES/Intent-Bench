from collections import defaultdict
from pathlib import Path
import ctypes
import statistics
import os
import time
import subprocess

def probe(backend,root):
    if backend=="cuda":
        return cuda(root)
    return mlu(root)

def cuda(root):
    import torch
    import triton
    from .calibration_tiles import traffic, empty
    d=torch.device("cuda",torch.cuda.current_device())
    p=torch.cuda.get_device_properties(d)
    x=torch.ones(1<<26,device=d)
    sinks=torch.empty(1<<16,device=d)
    def elapsed(fn):
        fn();torch.cuda.synchronize()
        stream=torch.cuda.Stream()
        start=torch.cuda.Event(enable_timing=True,external=True)
        stop=torch.cuda.Event(enable_timing=True,external=True)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            start.record();fn();stop.record()
        ns=[]
        for _ in range(9):
            graph.replay();torch.cuda.synchronize();ns.append(start.elapsed_time(stop)*1e6)
        return ns
    data={}
    for stride in (1,16):
        n=x.numel()//stride
        for write in (False,True):
            ns=elapsed(lambda:traffic[(n//1024,)](x,sinks,stride,write,1024))
            data[("write_" if write else "read_")+str(stride)]=[n*4/t for t in ns]
    data["launch"]=elapsed(lambda:empty[(1,)](sinks))
    torch.cuda.synchronize();data["sync"]=[]
    for _ in range(9):
        t=time.perf_counter_ns();torch.cuda.synchronize();data["sync"].append(time.perf_counter_ns()-t)
    return data,"triton "+triton.__version__+" CUDA "+torch.version.cuda,p.name,f"sm{p.major}{p.minor}", "one CUDA stream, 256 MiB storage, FP32 strides 1/16, graph events, idle stream for host sync; unmanaged clocks"

def mlu(root):
    import torch,torch_mlu
    from .implementations.mlu_rows import timer
    from .model import Measurement
    src=Path(__file__).with_name("native")/"calibrate_mlu.mlu"
    cc=os.path.join(os.environ["NEUWARE_HOME"],"bin","cncc")
    lib=root/"traffic.so"
    subprocess.run([cc,"--bang-mlu-arch=mtp_372","-O2","-shared","-fPIC",str(src),"-o",str(lib),"-lcnrt"],check=True)
    mod=ctypes.CDLL(str(lib));call=mod.enqueue
    call.argtypes=[ctypes.c_void_p,ctypes.c_void_p,*([ctypes.c_int]*4),ctypes.c_void_p]
    x=torch.ones(1<<24,device="mlu");sink=torch.empty_like(x);stream=torch.mlu.current_stream()
    settings=Measurement(samples=9,warmup=3)
    def elapsed(words,step,mode,trips):
        def run():
            code=call(x.data_ptr(),sink.data_ptr(),words,step,mode,trips,stream.mlu_stream)
            if code:raise RuntimeError(f"CNRT {code}")
        return [v*1e6 for v in timer(run,stream,settings)]
    data={"launch":elapsed(0,1,0,0)}
    for stride in (1,16):
        for mode,label in [(1,"read_"),(2,"write_")]:
            words=2048;trips=32
            ns=elapsed(words,stride,mode,trips)
            data[label+str(stride)]=[16*words*4*trips/t for t in ns]
    for mode,label in [(1,"in"),(2,"out")]:
        # Two sizes at equal copy count separate transfer intercept and bandwidth.
        small=elapsed(64,1,mode,256);large=elapsed(2048,1,mode,256)
        rates=[];latencies=[]
        for a,b,l in zip(small,large,data["launch"]):
            slope=(b-a)/(256*(2048-64)*4)
            if slope<=0:raise ValueError("DMA slope unresolved")
            rates.append(1/slope);latencies.append(max(0,(a-l)/256-64*4*slope))
        data["dma_"+label+"_bw"]=rates;data["dma_"+label+"_latency"]=latencies
    data["compute"]=[2048*512/v for v in elapsed(2048,1,4,512)]
    data["sync"]=[v/256 for v in elapsed(0,1,3,256)]
    return data,subprocess.check_output([cc,"--version"],text=True).strip().splitlines()[0],torch.mlu.get_device_name(0),"mtp_372","16 block tasks; CNRT notifier; per-participant DMA fit, 256 writes/reads at 256 and 8192 bytes; sync_io barrier; unmanaged clocks"
