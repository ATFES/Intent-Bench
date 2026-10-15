"""Standalone Weft artifact ABI; no Intent runtime or metadata is required."""
import base64
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from cnnl_worker import serve


class Worker:
    def __init__(self,request):
        root=Path(__file__).parent;self.request=request
        profile=request['profile'];artifact=request['artifact'];kernel,=artifact['kernels']
        self.inputs=[np.load(root/f'input{i}.npy') for i in range(request['arity'])]
        self.outputs=[np.empty(item['shape'],dtype=item['dtype']) for item in request['outputs']]
        self.arrays=self.inputs+self.outputs
        self.pointers=(C.c_void_p*len(self.arrays))(*(a.ctypes.data for a in self.arrays))
        invocation=[];before=[];after=[]
        for index,(spec,arg) in enumerate(zip(request['bindings'],kernel['arguments'],strict=True)):
            if 'workspace' in spec:
                name=f'workspace{index}';size=spec['workspace'];before.append(f'void *{name}=aligned_alloc(16,({size}+15)/16*16);if(!{name})abort();')
                after.append(f'free({name});');value=name
            else:value=f'p[{spec["buffer"]}]'
            invocation.append(f'({arg["c_type"]}){value}')
        invocation.extend(str(request['dimensions'][d]) for d in kernel['shape_parameters'])
        source=artifact['intrinsic_c']+'\n#include <time.h>\n#include <stdlib.h>\n'
        source+='void bench_run(void **p){'+''.join(before)+kernel['symbol']+'('+','.join(invocation)+');'+''.join(after)+'}\n'
        source+='double bench_sample(void **p,volatile unsigned char *cache,size_t bytes){for(size_t i=0;i<bytes;i+=64)cache[i]+=1;struct timespec a,b;clock_gettime(CLOCK_MONOTONIC,&a);bench_run(p);clock_gettime(CLOCK_MONOTONIC,&b);return (b.tv_sec-a.tv_sec)*1e3+(b.tv_nsec-a.tv_nsec)*1e-6;}\n'
        (root/'source.c').write_text(source)
        command=[*profile['cc'],'-O3','-shared','-fPIC','-march='+profile['march'],'-mabi='+profile['abi'],
                 *profile.get('cflags',[]),str(root/'source.c'),'-lm','-o',str(root/'source.so')]
        start=time.perf_counter();proc=subprocess.run(command,text=True,capture_output=True)
        if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
        self.compile_seconds=time.perf_counter()-start
        if profile.get('cpus'):os.sched_setaffinity(0,profile['cpus'])
        self.library=C.CDLL(str(root/'source.so'))
        self.library.bench_run.argtypes=[C.POINTER(C.c_void_p)];self.library.bench_run.restype=None
        self.library.bench_sample.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_size_t];self.library.bench_sample.restype=C.c_double
        self.cache=np.zeros(64*1024*1024,dtype='uint8')

    def run(self):self.library.bench_run(self.pointers)

    def result(self):
        items=[{'shape':list(a.shape),'dtype':str(a.dtype),'data':base64.b64encode(a.tobytes()).decode()} for a in self.outputs]
        return items[0] if len(items)==1 else {'items':items}

    def measure(self,samples,warmup,cache='flush'):
        for _ in range(warmup):self.run()
        return [self.library.bench_sample(self.pointers,self.cache.ctypes.data,self.cache.nbytes if cache=='flush' else 0) for _ in range(samples)]

    def close(self):pass


if __name__=='__main__':serve(Worker)
