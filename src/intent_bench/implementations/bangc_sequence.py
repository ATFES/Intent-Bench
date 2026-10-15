from dataclasses import replace
import torch
import intent
from ..corpus._common.measurement import report_stage

DTYPES = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32", torch.int32: "i32", torch.int64: "i64", torch.bool: "bool"}
TORCH_DTYPES = {value:key for key,value in DTYPES.items()}

class RemoteSequence:

    def __init__(self, context):
        self.context = context
        self.steps = []
        self.allocated_outputs = set()

    def add(self, definition, arguments, *, constexprs=None, target=None):
        views = {name: value for (name, value) in arguments.items() if isinstance(value, torch.Tensor)}
        target = replace(target or self.context.target, shapes={name: tuple(value.shape) for (name, value) in views.items()}, strides={name: tuple(value.stride()) for (name, value) in views.items()})
        report_stage('generated_compilation')
        program = intent.generate(definition, target=target, compiler=self.context.compiler, constexprs=constexprs, tuning_config=self.context.tuning_config, options=self.context.compile_options)
        bound = dict(arguments)
        interface = program.interface
        dimensions = {}
        for parameter in interface.views:
            value = bound.get(parameter.name)
            if value is not None:
                for (identity, extent) in zip(parameter.dimensions, value.shape):
                    if identity > 0:
                        dimensions[identity] = extent
        outputs = {}
        for parameter in interface.parameters:
            name = parameter.name
            if isinstance(parameter, intent.ViewParameter) and parameter.output and (name not in bound):
                shape = tuple((fixed if fixed >= 0 else dimensions[identity] for (fixed, identity) in zip(parameter.shape, parameter.dimensions)))
                bound[name] = torch.empty(shape, dtype=TORCH_DTYPES[parameter.dtype.name], device='cpu')
                storage = bound[name].untyped_storage()
                self.allocated_outputs.add((str(bound[name].device), storage.data_ptr(), storage.nbytes()))
            if name not in bound:
                raise ValueError(f'missing BANG C invocation argument {name}')
            if isinstance(parameter, intent.ViewParameter) and parameter.writable:
                outputs[name] = bound[name]
        self.steps.append((program, bound, interface))
        report_stage('adapter_preparation')
        return outputs
