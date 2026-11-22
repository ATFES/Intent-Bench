"""Bind the packaged V1 normalization kernels on an additional execution backend."""
from contextlib import ExitStack

from .intent import backend_options, make_kernel
from .outputs import native_result
from .prepared import intent_sequence, native_selection
from ..model import PreparedCall
from ..workloads.programs import _device


def prepare(symbol, backend, case, payload, env):
    from ..loading import intent_environment
    intent = intent_environment(env)
    from intent.examples.kernels.normalization.fused_add_rms_norm import fused_add_rms_norm
    from intent.examples.kernels.normalization.dropout_residual_rms_norm import (
        dropout_residual_rms_norm_backward_data, dropout_residual_rms_norm_forward,
    )
    from intent.examples.kernels.backward.layer_norm import layer_norm_backward_rows, layer_norm_backward_reduce

    device = _device(env,backend)
    with ExitStack() as owners:
        buffers = intent.HostBuffers(backend,device,backend_options(env,backend))
        owners.callback(buffers.close)
        arguments = payload.arguments
        if device.startswith("cuda:"):
            from ..trees import to_device
            arguments = to_device(arguments,device)
        args = tuple(buffers.input(v) if hasattr(v,"shape") else v for v in arguments)
        steps, members = [], []

        def bind(definition, values):
            kernel=make_kernel(definition,env,backend)
            owners.callback(kernel.close)
            call=kernel.prepare(*values)
            members.append((kernel,call))
            steps.append(call)
            result=call.result()
            buffers.retain_outputs(result)
            return result

        if symbol=="fused_add_rms_norm":
            outputs=bind(fused_add_rms_norm,args)
        elif symbol=="dropout_residual_rms_norm":
            forward=bind(dropout_residual_rms_norm_forward,(*args[:3],*args[5:]))
            backward=bind(dropout_residual_rms_norm_backward_data,args)
            outputs=(*forward,*backward)
        elif symbol=="layer_norm_backward":
            x,dy,w,mean,inv,scale=args
            shape=(128,x.shape[1])
            wp=buffers.workspace(x,shape,dtype="bf16")
            bp=buffers.workspace(x,shape,dtype="bf16")
            dx=bind(layer_norm_backward_rows,(x,dy,w,mean,inv,wp,bp,scale))
            kernel=members[0][0]
            steps[:0]=[kernel.prepare_fill(wp,0),kernel.prepare_fill(bp,0)]
            dw,db=bind(layer_norm_backward_reduce,(wp,bp))
            outputs=(dx,dw,db)
        else:
            raise ValueError(f"unknown legacy normalization program {symbol}")
        sequence=intent.PreparedSequence(steps,outputs=outputs,owners=(buffers,))
        lifetime=owners.pop_all()
        if backend=="bangc":
            from .bangc import prepare_sequence
            call=prepare_sequence(sequence,outputs,environment=env,close=lifetime.close,owners=lifetime)
        elif device=="cpu":
            call=intent_sequence(sequence,lambda:native_result(outputs),release=lifetime.close)
        else:
            call=PreparedCall(sequence.launch,lambda:native_result(outputs),device=device,
                              compile=sequence.compile,close=lifetime.close)
        call.tuning_report=lambda:{"members":[native_selection(step,kernel) for kernel,step in members]}
        return call
