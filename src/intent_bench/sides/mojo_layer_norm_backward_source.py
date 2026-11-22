from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    x,dy,weight,mean,rstd,_ = context.inputs
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_layer_norm_backward')
    source_state = {}

    def source_launch():
        source_state['output'] = runtime.layer_norm_backward(x, dy, weight, mean, rstd)
    return checked_call(PreparedCall(source_launch, lambda : source_state['output']), (Tolerance(atol=0.05), Tolerance(atol=0.125), Tolerance(atol=0.125)), cuda_graph=False, device_type='cpu', note='既有B4096-N4096 bf16 LayerNorm backward；rows partial后接reduce，完整返回dx/dweight/dbias，PyTorch CPU reference同算法。', owners=locals())


def native_cpu(context):
    from intent_bench.implementations.aten_cpu import prepare as aten
    call=aten("layer_norm_backward",context.inputs,context.environment)
    call.tolerance=(Tolerance(atol=0.05),Tolerance(atol=0.125),Tolerance(atol=0.125))
    return call


def native_device(context):
    import torch
    device=context.environment.device
    if device.startswith("mlu:"):
        import torch_mlu
    elif not device.startswith("cuda:"):
        raise ValueError("the native accelerator implementation requires a CUDA or MLU device")
    x,dy,weight,mean,inv,scale=context.inputs
    if scale != 1/x.shape[1]:
        raise ValueError("ATen requires inverse_features to match the input")
    x,dy,weight,mean,inv=(t.to(device) for t in (x,dy,weight,mean,inv))
    mean,inv=mean.reshape(-1,1),inv.reshape(-1,1)
    gain=torch.empty_like(weight,dtype=torch.float32)
    bias=torch.zeros_like(gain)
    outputs=(torch.empty_like(x),torch.empty_like(gain),torch.empty_like(gain))
    wide_x,wide_dy,wide_dx=(torch.empty_like(x,dtype=torch.float32) for _ in range(3))
    def launch():
        gain.copy_(weight)
        wide_x.copy_(x)
        wide_dy.copy_(dy)
        torch.ops.aten.native_layer_norm_backward.out(
            wide_dy,wide_x,[x.shape[1]],mean,inv,gain,bias,[True,True,True],
            out0=wide_dx,out1=outputs[1],out2=outputs[2])
        outputs[0].copy_(wide_dx)
    call=PreparedCall(launch,lambda:outputs,device=device,
                      tolerance=(Tolerance(atol=.05),Tolerance(atol=.125),Tolerance(atol=.125)),
                      metadata={"library":"ATen native_layer_norm_backward, FP32 computation and all BF16 conversions timed"})
    if device.startswith("mlu:"):
        from intent_bench.implementations.mlu_rows import timer
        queue=torch.mlu.current_stream(device)
        call.native_measure=lambda settings:timer(launch,queue,settings)
        call.synchronize=queue.synchronize
        call.timing="cnrt_notifier"
    return call
