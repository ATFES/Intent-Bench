import os
import torch
from ..loading import intent_environment,load
from ..model import PreparedCall
from ..trees import clone,restore

DEFINITIONS={
 'scaled_fp8_gemm':'intent.examples.kernels.contraction.block_scaled:scaled_fp8_matmul',
 'flash_attention_bias':'intent.examples.kernels.streaming.attention:flash_attention_bias_fwd',
 'flash_varlen_attention':'intent.examples.kernels.streaming.attention:flash_varlen_attention_fwd',
 'flash_attention_inline':'intent.examples.kernels.variants.streaming:flash_attention_inline_fwd',
 'flash_attention_select':'intent.examples.kernels.variants.streaming:flash_attention_select_fwd',
 'flash_attention_full_causal_stream':'intent.examples.kernels.variants.streaming:flash_attention_full_causal_stream_fwd',
 'chunk_retention':'intent.examples.kernels.streaming.linear_attention:chunk_retention_fwd',
 'mamba3_siso_step':'intent.examples.kernels.streaming.mamba:mamba3_siso_step',
 'mamba3_siso_forward':'intent.examples.kernels.streaming.mamba:mamba3_siso_forward',
}


def prepare(name,args,env):
    intent=intent_environment(env)
    from .intent import make_kernel
    artifacts=[]
    def compile(reference,constexprs=None):
        artifact=make_kernel(load(reference),env,'mojo',constexprs=constexprs)
        artifacts.append(artifact)
        return artifact
    if name=='histogram':
        from intent.examples.programs.composition import Histogram
        prefix='intent.examples.kernels.statistics.histogram:'
        chunk=8192
        program=Histogram(compile(prefix+'histogram_partials',{'CHUNK':chunk}),
                          compile(prefix+'histogram_merge'),chunk)
        invocation=program.prepare(*args)
        calls=[invocation.partials,invocation.merge];result=invocation.result
    elif name=='fused_linear_cross_entropy':
        hidden,weight,labels=args;m=hidden.shape[0];n=weight.shape[0];chunk=min(m,1024)
        scratch=torch.empty((chunk,n),dtype=torch.bfloat16);loss=torch.empty(m);mean=torch.empty(1)
        prefix='intent.examples.kernels.loss.fused_linear_cross_entropy:'
        linear=compile(prefix+'linear_logits_chunk');probability=compile(prefix+'cross_entropy_probability_chunk');reduction=compile(prefix+'cross_entropy_mean')
        calls=[]
        for begin in range(0,m,chunk):
            end=min(begin+chunk,m);piece=scratch[:end-begin]
            calls.append(linear.prepare(hidden[begin:end],weight,outputs=(piece,)))
            calls.append(probability.prepare(piece,labels[begin:end],outputs=(loss[begin:end],)))
        calls.append(reduction.prepare(loss,outputs=(mean,)))
        result=lambda:(loss,mean,scratch)
    elif name=='paged_gqa_f16_reduction':
        q,k,v,offsets,indices,lengths,splits,scale=args
        partials=compile('intent.examples.kernels.streaming.paged_attention:splitk_paged_gqa_decode_partials',
                         {'PAGE_SIZE':k.shape[1],'HEAD_GROUP':q.shape[1]//k.shape[2],'SPLITS':8})
        reducer=compile('intent.examples.kernels.streaming.splitk_reduce:splitk_attention_bf16_to_f16_reduce')
        first=partials.prepare(*args);lse,partial=first.result();second=reducer.prepare(partial,lse)
        calls=[first,second];result=second.result
    else:
        constants={}
        if name in {'flash_varlen_attention','flash_attention_inline','flash_attention_select','flash_attention_full_causal_stream'}:constants['CAUSAL']=True
        if name.startswith('mamba3'):
            constants['HEAD_GROUP']=args[2].shape[-2]//args[0].shape[-2]
        artifact=compile(DEFINITIONS[name],constants or None)
        call=artifact.prepare(*args);calls=[call]
        result=(lambda:(call.result(),tuple(args[-5:]))) if name=='mamba3_siso_forward' else call.result
    initial=clone(args)
    from .native_sampling import sample
    reset=lambda:restore(args,initial)
    def run():
        for call in calls:call.launch()
    def native_compile():
        for call in calls:call.compile()
    def close():
        for artifact in artifacts:
            artifact.close()
    if len(calls) > 1:
        from .prepared import intent_sequence
        prepared = intent_sequence(calls, result, reset=reset,
                                   metadata={'artifacts': [str(a.cache_directory) for kernel in artifacts for a in kernel.artifacts]})
        release_timer = prepared.close
        def release():
            release_timer()
            close()
        prepared.close = release
        return prepared
    return PreparedCall(run,result,reset=reset,compile=native_compile,close=close,
                        native_measure=(lambda settings:sample(calls[0],settings,reset)) if len(calls)==1 else None,
                        timing='cpu_native_monotonic' if len(calls)==1 else None,
                        metadata={'artifacts':[str(a.cache_directory) for kernel in artifacts for a in kernel.artifacts]})
