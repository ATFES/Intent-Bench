from collections import defaultdict
from pathlib import Path
import ctypes
import statistics
import os
import time
import subprocess
import json

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
    (root/"cache.json").write_text(json.dumps({"bytes":p.L2_cache_size, "source":"CUDA L2 device property"}))
    x=torch.ones(1<<28,device=d)
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
    for tier,words in (("private",16384),("mid",1<<18),("shared",1<<21),("dram",x.numel())):
        for stride in (1,16):
            n=words//stride
            for write in (False,True):
                ns=elapsed(lambda:traffic[(n//1024,)](x,sinks,stride,write,1024))
                data[("write_" if write else "read_")+str(stride)+"_"+tier+"_bw"]=[n*4/t for t in ns]
    data["launch"]=elapsed(lambda:empty[(1,)](sinks))
    torch.cuda.synchronize();data["sync"]=[]
    for _ in range(9):
        t=time.perf_counter_ns();torch.cuda.synchronize();data["sync"].append(time.perf_counter_ns()-t)
    return data,"triton "+triton.__version__+" CUDA "+torch.version.cuda,p.name,f"sm{p.major}{p.minor}", "one CUDA stream, 64 KiB, 1 MiB, 8 MiB, 1 GiB footprints, FP32 strides 1/16, graph events, idle stream for host sync; unmanaged clocks"

def mlu(root):
    import torch,torch_mlu
    from .implementations.mlu_rows import timer
    from .model import Measurement
    src=Path(__file__).with_name("native")/"calibrate_mlu.mlu"
    cc=os.path.join(os.environ["NEUWARE_HOME"],"bin","cncc")
    lib=root/"traffic.so"
    subprocess.run([cc,"--bang-mlu-arch=mtp_372","-O2","-shared","-fPIC",str(src),"-o",str(lib),"-lcnrt"],check=True)
    mod=ctypes.CDLL(str(lib));call=mod.enqueue;call_tasks=mod.enqueue_tasks
    call.argtypes=[ctypes.c_void_p,ctypes.c_void_p,*([ctypes.c_int]*4),ctypes.c_void_p]
    call_tasks.argtypes=[ctypes.c_void_p,ctypes.c_void_p,*([ctypes.c_int]*5),ctypes.c_void_p]
    x=torch.ones(1<<26,device="mlu");sink=torch.empty_like(x);stream=torch.mlu.current_stream()
    size=ctypes.c_int()
    cnrt=ctypes.CDLL(os.path.join(os.environ["NEUWARE_HOME"],"lib64","libcnrt.so"))
    status=cnrt.cnrtDeviceGetAttribute(ctypes.byref(size),0x301,torch.mlu.current_device())
    (root/"cache.json").write_text(json.dumps({"bytes":size.value if status==0 else None,
        "source":"cnrtAttrMaxL2CacheSize", "status":status}))
    settings=Measurement(samples=9,warmup=3)
    def elapsed(words,step,mode,trips,tasks=16):
        def run():
            code=call_tasks(x.data_ptr(),sink.data_ptr(),words,step,mode,trips,tasks,stream.mlu_stream)
            if code:raise RuntimeError(f"CNRT {code}")
        return [v*1e6 for v in timer(run,stream,settings)]
    data={"launch":elapsed(0,1,0,0)}
    for tier,words_total in (("private",16384),("mid",1<<18),("shared",1<<21),("dram",x.numel())):
        for stride in (1,16):
            words=min(2048,words_total//(16*stride))
            trips=words_total//(16*words*stride)
            for mode,label in [(1,"read_"),(2,"write_")]:
                ns=elapsed(words,stride,mode,trips)
                data[label+str(stride)+"_"+tier+"_bw"]=[16*words*4*trips/t for t in ns]
    for mode,label in [(1,"in"),(2,"out")]:
        sizes=(64,128,256,512,1024,2048)
        samples=[elapsed(size,1,mode,256) for size in sizes]
        xs=[4*size for size in sizes]
        (root/f"dma-{label}.json").write_text(json.dumps({"bytes_per_transfer":xs,
            "transfers":256,"launch_ns":data["launch"],"elapsed_ns":samples},indent=2))
        xm=statistics.mean(xs)
        rates=[];latencies=[];errors=[]
        for start,*times in zip(data["launch"],*samples):
            ys=[(t-start)/256 for t in times]; ym=statistics.mean(ys)
            slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
            if slope<=0:raise ValueError("DMA slope unresolved")
            intercept=max(0,ym-slope*xm)
            rates.append(1/slope);latencies.append(intercept)
            errors.append((sum((y-intercept-slope*x)**2 for x,y in zip(xs,ys))/len(xs))**0.5)
        data["dma_"+label+"_bw"]=rates;data["dma_"+label+"_latency"]=latencies
        data["dma_"+label+"_rmse"]=errors
    data["compute"]=[2048*512/v for v in elapsed(2048,1,4,512)]
    # Difference across trip counts removes launch, input DMA and final drain.
    for mode,name in ((6,"max"),(7,"sum"),(8,"exp"),(9,"div_rn_vv"),(13,"div_rn_vs"),(14,"div_rn_sv")):
        for width in (64,512,2048):
            short=elapsed(width,1,mode,16)
            long=elapsed(width,1,mode,128)
            data[f"compute_{name}_{width}"]=[(b-a)/112 for a,b in zip(short,long)]
            if min(data[f"compute_{name}_{width}"])<=0:
                raise ValueError(f"unresolved {name} timing at width {width}")
    data["sync"]=[max(0, v-start)/256 for v,start in zip(elapsed(0,1,3,256),data["launch"])]
    first=elapsed(128,1,5,1)
    data["first_io"]= [
        max(0, time_ns-launch-sync-in_lat-out_lat-512/in_bw-512/out_bw)
        for time_ns,launch,sync,in_lat,out_lat,in_bw,out_bw in zip(
            first,data["launch"],data["sync"],data["dma_in_latency"],
            data["dma_out_latency"],data["dma_in_bw"],data["dma_out_bw"])
    ]
    # Synchronous DMA-compute round-trip latency (16 tasks) and single-task streaming DMA bandwidth.
    rt_sizes=(64,256,1024,2048)
    rt_xs=[4*s for s in rt_sizes]
    rt_xm=statistics.mean(rt_xs)
    def diff_step(words,mode,tasks):
        short=elapsed(words,1,mode,16,tasks)
        long=elapsed(words,1,mode,128,tasks)
        return [(b-a)/112 for a,b in zip(short,long)]
    for tasks,want_lat,want_bw in ((16,True,False),(1,False,True)):
        base_c=[diff_step(s,12,tasks) for s in rt_sizes]
        for mode,label in ((10,"in"),(11,"out")):
            raw=[diff_step(s,mode,tasks) for s in rt_sizes]
            net=[[max(0.0,v-c) for v,c in zip(rs,cs)] for rs,cs in zip(raw,base_c)]
            rates=[];latencies=[]
            for ys in zip(*net):
                ym=statistics.mean(ys)
                slope=sum((xv-rt_xm)*(yv-ym) for xv,yv in zip(rt_xs,ys))/sum((xv-rt_xm)**2 for xv in rt_xs)
                if slope<=0:raise ValueError(f"sync DMA {label} slope unresolved at tasks={tasks}")
                rates.append(1/slope)
                latencies.append(max(0.0,ym-slope*rt_xm))
            if want_lat: data[f"dma_{label}_sync_latency"]=latencies
            if want_bw: data[f"dma_{label}_single_bw"]=rates
    return data,subprocess.check_output([cc,"--version"],text=True).strip().splitlines()[0],torch.mlu.get_device_name(0),"mtp_372","16 block tasks; CNRT notifier; 64 KiB, 1 MiB, 8 MiB and 256 MiB footprints; six-size DMA fit at 256..8192 bytes, 256 copies; sync_io barrier; differential 128-16 sync DMA-compute round-trip fit at 256..8192 bytes (16 tasks and 1 task); unmanaged clocks"
