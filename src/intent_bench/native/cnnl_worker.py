"""Independent CNNL process. No Intent, torch or SSH work occurs inside its timer."""
import base64
import ctypes as C
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np


class Worker:
    def __init__(self, request):
        self.buffers=[];self.context=C.c_void_p();self.queue=C.c_void_p();self.notifiers=[]
        root=Path(__file__).parent;sdk=Path(request.get("neuware","/usr/local/neuware"))
        start=time.perf_counter()
        build=subprocess.run(["c++","-std=c++17","-O3","-shared","-fPIC",str(root/"cnnl_ops.cpp"),
            "-I"+str(sdk/"include"),"-L"+str(sdk/"lib64"),"-Wl,-rpath,"+str(sdk/"lib64"),"-lcnnl","-lcnrt","-o",str(root/"ops.so")],text=True,capture_output=True)
        if build.returncode:raise RuntimeError(build.stdout+build.stderr)
        self.compile_seconds=time.perf_counter()-start
        self.runtime=C.CDLL(str(sdk/"lib64/libcnrt.so"));self.lib=C.CDLL(str(root/"ops.so"))
        signatures={"cnrtSetDevice":[C.c_int],"cnrtMalloc":[C.POINTER(C.c_void_p),C.c_size_t],
          "cnrtMemcpy":[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int],"cnrtFree":[C.c_void_p],
          "cnrtQueueCreate":[C.POINTER(C.c_void_p)],"cnrtQueueSync":[C.c_void_p],"cnrtQueueDestroy":[C.c_void_p],
          "cnrtNotifierCreate":[C.POINTER(C.c_void_p)],"cnrtNotifierDestroy":[C.c_void_p],
          "cnrtPlaceNotifier":[C.c_void_p,C.c_void_p],"cnrtNotifierDuration":[C.c_void_p,C.c_void_p,C.POINTER(C.c_float)]}
        for name,signature in signatures.items():
            fn=getattr(self.runtime,name);fn.argtypes=signature;fn.restype=C.c_int
        self.lib.source_create.argtypes=[C.POINTER(C.c_void_p),C.c_void_p]+[C.c_int]*4
        self.lib.source_create.restype=C.c_int
        self.lib.source_run.argtypes=[C.c_void_p]*4;self.lib.source_run.restype=C.c_int
        self.lib.source_destroy.argtypes=[C.c_void_p];self.lib.source_destroy.restype=None
        self.lib.cnnlGetErrorString.argtypes=[C.c_int];self.lib.cnnlGetErrorString.restype=C.c_char_p
        self.cnrt("cnrtSetDevice",request.get("device",0));self.cnrt("cnrtQueueCreate",C.byref(self.queue))
        self.inputs=[np.load(root/f"input{i}.npy") for i in range(request["arity"])]
        op=request["operation"];rows,columns=self.inputs[0].shape;depth=0
        if op=="matmul":depth=columns;columns=self.inputs[1].shape[1]
        self.output=np.empty((rows,columns),dtype="float16")
        for value in [*self.inputs,self.output]:
            pointer=C.c_void_p();self.cnrt("cnrtMalloc",C.byref(pointer),value.nbytes);self.buffers.append(pointer)
            if value is not self.output:self.cnrt("cnrtMemcpy",pointer,value.ctypes.data,value.nbytes,0)
        self.cnnl(self.lib.source_create(C.byref(self.context),self.queue,{"relu":0,"softmax":1,"matmul":2,"layer_norm":3,"gelu":4}[op],rows,columns,depth))
        for _ in range(2):
            event=C.c_void_p();self.cnrt("cnrtNotifierCreate",C.byref(event));self.notifiers.append(event)

    def cnrt(self,name,*args):
        code=getattr(self.runtime,name)(*args)
        if code:raise RuntimeError(f"{name}: CNRT error {code}")

    def cnnl(self,code):
        if code:raise RuntimeError(self.lib.cnnlGetErrorString(code).decode())

    def enqueue(self):
        self.cnnl(self.lib.source_run(self.context,self.buffers[0],self.buffers[1] if len(self.inputs)==2 else None,self.buffers[-1]))

    def run(self):
        self.enqueue();self.cnrt("cnrtQueueSync",self.queue)

    def result(self):
        self.cnrt("cnrtMemcpy",self.output.ctypes.data,self.buffers[-1],self.output.nbytes,2)
        return {"shape":list(self.output.shape),"dtype":str(self.output.dtype),"data":base64.b64encode(self.output.tobytes()).decode()}

    def measure(self,samples,warmup):
        for _ in range(warmup):self.run()
        times=[]
        for _ in range(samples):
            self.cnrt("cnrtPlaceNotifier",self.notifiers[0],self.queue)
            self.enqueue()
            self.cnrt("cnrtPlaceNotifier",self.notifiers[1],self.queue)
            self.cnrt("cnrtQueueSync",self.queue)
            microseconds=C.c_float()
            self.cnrt("cnrtNotifierDuration",*self.notifiers,C.byref(microseconds))
            times.append(microseconds.value*1e-3)
        return times

    def close(self):
        if self.queue.value:self.cnrt("cnrtQueueSync",self.queue)
        if self.context.value:self.lib.source_destroy(self.context)
        for event in self.notifiers:self.cnrt("cnrtNotifierDestroy",event)
        for pointer in self.buffers:self.cnrt("cnrtFree",pointer)
        if self.queue.value:self.cnrt("cnrtQueueDestroy",self.queue)


def serve(factory):
    # SDK diagnostics may write directly to fd 1. Reserve a separate fd for RPC.
    protocol=os.fdopen(os.dup(1),"w",buffering=1)
    os.dup2(2,1)
    sys.stdout=protocol
    worker=None
    try:
        worker=factory(json.loads(Path(sys.argv[1]).read_text()))
        print(json.dumps({"ready":True,"compile_seconds":worker.compile_seconds,
                          'environment':{'machine':platform.node(),'platform':platform.platform(),'python':platform.python_version()}}),flush=True)
        for line in sys.stdin:
            command=json.loads(line)
            try:
                action=command["action"]
                if action=="close":break
                if action=='run':value=worker.run()
                elif action=='result':value=worker.result()
                elif action=='packed':value=worker.packed()
                elif action=='measure':value=worker.measure(**{k:v for k,v in command.items() if k!='action'})
                else:raise ValueError('unknown worker action: '+action)
                print(json.dumps({"value":value}),flush=True)
            except Exception:
                print(json.dumps({"error":traceback.format_exc()}),flush=True)
    except Exception:
        print(json.dumps({"error":traceback.format_exc()}),flush=True)
    finally:
        if worker:worker.close()


if __name__=="__main__":serve(Worker)
