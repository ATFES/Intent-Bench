"""Authored MLU sequences bind through the product JIT on the executing host."""
import torch
from intent.language.annotations import ConstexprSpec, ViewKind, ViewSpec

DTYPES = {torch.float16:"f16", torch.bfloat16:"bf16", torch.float32:"f32",
          torch.int8:"i8", torch.uint8:"u8", torch.int32:"i32", torch.int64:"i64", torch.bool:"bool"}
TORCH_DTYPES = {value:key for key,value in DTYPES.items()}


class LocalSequence:
    def __init__(self, context):
        from intent.runtime.bangc import Queue
        self.context = context
        env = context.environment
        self.device = int(env.device.split(":")[1])
        self.neuware = env.options.get("neuware", "/usr/local/neuware")
        self.queue = Queue(device=self.device, neuware=self.neuware)
        self.calls, self.buffers, self.initial, self.outputs = [], {}, {}, []

    def _buffer(self, value):
        from intent.runtime.bangc import DeviceBuffer, DeviceView
        if isinstance(value, (DeviceBuffer, DeviceView)):
            return value
        if not isinstance(value, torch.Tensor):
            return value
        if value.device.type != "cpu":
            raise ValueError("MLU workload input preparation must produce CPU tensors or MLU buffers")
        storage = value.untyped_storage()
        key = storage.data_ptr(), storage.nbytes()
        if key not in self.buffers:
            raw = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
            data = raw.numpy()
            self.buffers[key] = DeviceBuffer.from_host(data, shape=(storage.nbytes()//value.element_size(),),
                                                       dtype=DTYPES[value.dtype], device=self.device,
                                                       neuware=self.neuware)
            self.initial[key] = data.copy()
        return self.buffers[key].view(tuple(value.shape), tuple(value.stride()), offset=value.storage_offset())

    def add(self, definition, arguments, *, constexprs=None):
        parameters = [p for p in definition.signature.parameters.values()
                      if not isinstance(p.annotation, ConstexprSpec)]
        inputs = [p for p in parameters if not (isinstance(p.annotation, ViewSpec)
                                               and p.annotation.kind is ViewKind.OUT)]
        outs = [p for p in parameters if isinstance(p.annotation, ViewSpec)
                and p.annotation.kind is ViewKind.OUT]
        supplied = {p.name:self._buffer(arguments[p.name]) for p in inputs}
        explicit = any(p.name in arguments for p in outs)
        output_values = tuple(self._buffer(arguments[p.name]) for p in outs) if explicit else None
        kernel = self.context.compile(definition, constexprs=constexprs)
        call = kernel.prepare(*(supplied[p.name] for p in inputs), outputs=output_values)
        self.calls.append(call)
        values = call.result()
        values = () if not outs else (values,) if len(outs) == 1 else values
        if not explicit:
            self.outputs.extend(values)
        return {**{p.name:supplied[p.name] for p in inputs if isinstance(p.annotation, ViewSpec)
                   and p.annotation.kind is ViewKind.INOUT},
                **dict(zip((p.name for p in outs), values, strict=True))}

    def compile(self):
        for call in self.calls:
            call.compile()

    def reset(self):
        for key, data in self.initial.items():
            self.buffers[key].copy_from_host(data)

    def enqueue(self):
        for call in self.calls:
            call.enqueue(self.queue)

    def launch(self):
        self.enqueue()
        self.queue.synchronize()

    def close(self):
        self.queue.close()
        for kernel in self.context.kernels:
            kernel.close()
        for buffer in self.buffers.values():
            buffer.close()
        for value in self.outputs:
            value.close()
