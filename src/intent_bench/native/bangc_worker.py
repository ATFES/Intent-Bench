"""Native timing of an explicit Intent kernel sequence; transport stays outside."""
import base64
import ctypes as C
import json
from pathlib import Path
import time
import numpy as np

from cnnl_worker import serve


class Worker:
    def __init__(self,request):
        from intent.compiler.artifact import GeneratedProgram
        from intent.targets.bangc import BangCTarget
        from intent.runtime.bangc.buffer import DeviceBuffer
        self.root=Path(__file__).parent/'program'
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.buffers={};self.initial={};self.programs=[];self.calls=[];self.notifiers=[]
        self.mutable=set();start=time.perf_counter()
        for name,spec in self.manifest['buffers'].items():
            buffer=DeviceBuffer((spec['elements'],),spec['dtype'],device=request['device'],neuware=request['neuware'])
            self.buffers[name]=buffer
            if 'file' in spec:
                data=(self.root/spec['file']).read_bytes();buffer.copy_from_host(data);self.initial[name]=data
        def argument(spec):
            if not isinstance(spec,dict):return spec
            return self.buffers[spec['buffer']].view(tuple(spec['shape']),tuple(spec['strides']),offset=spec['offset'])
        for step in self.manifest['steps']:
            generated=GeneratedProgram.load(self.root/step['artifact'])
            target=BangCTarget.from_program(generated,device=request['device'],neuware=request['neuware'])
            program=generated.materialize(target=target).runtime;self.programs.append(program)
            self.calls.append(program.prepare(tuple(argument(v) for v in step['arguments']),explicit_outputs=True))
            self.mutable.update(step['arguments'][p.position]['buffer'] for p in program.interface.mutable_inputs)
        # Native compilation and selection finish before the RPC ready response.
        from intent.runtime.bangc.program import launch_calls
        launch_calls(tuple(self.calls))
        self.owner=self.calls[0].program;self.runtime=self.owner.runtime;self.queue=self.owner.queue
        self.compile_seconds=time.perf_counter()-start
        for _ in range(2):
            notifier=C.c_void_p();self.runtime.invoke('cnrtNotifierCreate',C.byref(notifier));self.notifiers.append(notifier)

    def reset(self):
        for name in self.mutable:
            if name in self.initial:self.buffers[name].copy_from_host(self.initial[name])

    def enqueue(self):
        for call in self.calls:call.enqueue(self.queue)

    def run(self):
        self.reset();self.enqueue();self.runtime.invoke('cnrtQueueSync',self.queue)

    def result(self):
        dtypes={'f16':('float16','float16'),'bf16':('uint16','bfloat16'),'f32':('float32','float32'),
                'i32':('int32','int32'),'i64':('int64','int64'),'bool':('bool','bool')}
        def encode(spec):
            if 'items' in spec:return {'items':[encode(item) for item in spec['items']]}
            info=self.manifest['buffers'][spec['buffer']];dtype,torch_dtype=dtypes[info['dtype']]
            itemsize=np.dtype(dtype).itemsize;data=self.buffers[spec['buffer']].to_host()
            value=np.ndarray(spec['shape'],dtype=dtype,buffer=data,offset=spec['offset']*itemsize,
                             strides=tuple(x*itemsize for x in spec['strides']))
            return {'shape':spec['shape'],'torch_dtype':torch_dtype,'data':base64.b64encode(value.tobytes()).decode()}
        return encode(self.manifest['outputs'])

    def measure(self,samples,warmup):
        for _ in range(warmup):self.run()
        times=[]
        for _ in range(samples):
            self.reset()
            self.runtime.invoke('cnrtPlaceNotifier',self.notifiers[0],self.queue)
            self.enqueue()
            self.runtime.invoke('cnrtPlaceNotifier',self.notifiers[1],self.queue)
            self.runtime.invoke('cnrtQueueSync',self.queue)
            elapsed=C.c_float();self.runtime.invoke('cnrtNotifierDuration',*self.notifiers,C.byref(elapsed))
            times.append(elapsed.value*1e-3)
        return times

    def close(self):
        for notifier in self.notifiers:self.runtime.invoke('cnrtNotifierDestroy',notifier)
        for program in self.programs:program.close()
        for buffer in self.buffers.values():buffer.close()


if __name__=='__main__':serve(Worker)
