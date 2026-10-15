"""Intent's own Weft runtime is used only by the selected Intent implementation."""
import base64
import ctypes as C
import os
from pathlib import Path
import time
import numpy as np
from cnnl_worker import serve


class Worker:
    def __init__(self,request):
        from intent.runtime.weft import Buffer,NativeProgram,compile_artifact
        root=Path(__file__).parent;profile=request['profile'];start=time.perf_counter()
        artifact=compile_artifact(root/'program',cc=tuple(profile['cc']),cflags=tuple(profile.get('cflags',[])))
        if profile.get('cpus'):os.sched_setaffinity(0,profile['cpus'])
        self.program=NativeProgram(artifact)
        dtypes={'float32':'f32','int32':'i32','int8':'i8','uint8':'u8'}
        arguments=[]
        for value in request['arguments']:
            if isinstance(value,dict):
                array=np.load(root/f"input{value['input']}.npy")
                arguments.append(Buffer(bytearray(array.tobytes()),shape=array.shape,dtype=dtypes[str(array.dtype)]))
            else:arguments.append(value)
        self.call=self.program.prepare(tuple(arguments));self.call.compile();self.call.choose()
        self.compile_seconds=time.perf_counter()-start
        self.cache=bytearray(64*1024*1024)
        self.evict=self.program.library.bind('intent_weft_evict',[C.c_void_p,C.c_int64],None)
        self.cache_pointer=C.addressof(C.c_char.from_buffer(self.cache))

    def run(self):self.call.launch()

    def result(self):
        dtypes={'f32':'float32','i32':'int32','i8':'int8','u8':'uint8'}
        def encode(value):
            if isinstance(value,tuple):return {'items':[encode(v) for v in value]}
            return {'shape':list(value.shape),'dtype':dtypes[value.dtype],'data':base64.b64encode(value.storage).decode()}
        return encode(self.call.result())

    def measure(self,samples,warmup,cache='flush'):
        for _ in range(warmup):self.run()
        prepare=(lambda:self.evict(self.cache_pointer,len(self.cache))) if cache=='flush' else None
        return [self.call.benchmark(samples=1,prepare=prepare) for _ in range(samples)]

    def close(self):self.program.close()


if __name__=='__main__':serve(Worker)
