from ..loading import intent_environment,load
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
        calls=[invocation];result=invocation.result
    elif name=='fused_linear_cross_entropy':
        from intent.examples.programs.composition import LinearCrossEntropy
        prefix='intent.examples.kernels.loss.fused_linear_cross_entropy:'
        program=LinearCrossEntropy(compile(prefix+'linear_logits_chunk'),
            compile(prefix+'cross_entropy_probability_chunk'),compile(prefix+'cross_entropy_mean'))
        invocation=program.prepare(*args)
        calls=[invocation];result=invocation.result
    elif name=='paged_gqa_f16_reduction':
        q,k,v,offsets,indices,lengths,splits,scale=args
        partials=compile('intent.examples.kernels.streaming.paged_attention:splitk_paged_gqa_decode_partials',
                         {'PAGE_SIZE':k.shape[1],'HEAD_GROUP':q.shape[1]//k.shape[2],'SPLITS':8})
        reducer=compile('intent.examples.kernels.streaming.splitk_reduce:splitk_attention_bf16_to_f16_reduce')
        from intent.examples.programs.composition import PagedDecode
        invocation=PagedDecode(partials,reducer).prepare(*args)
        calls=[invocation];result=invocation.result
    else:
        constants={}
        if name in {'flash_varlen_attention','flash_attention_inline','flash_attention_select','flash_attention_full_causal_stream'}:constants['CAUSAL']=True
        if name.startswith('mamba3'):
            constants['HEAD_GROUP']=args[2].shape[-2]//args[0].shape[-2]
        artifact=compile(DEFINITIONS[name],constants or None)
        call=artifact.prepare(*args);calls=[call]
        result=(lambda:(call.result(),tuple(args[-5:]))) if name=='mamba3_siso_forward' else call.result
    initial=clone(args)
    reset=lambda:restore(args,initial)
    def close():
        for artifact in artifacts:
            artifact.close()
    from .prepared import intent_sequence
    return intent_sequence(calls, result, reset=reset, release=close,
                           metadata={'artifacts': [str(a.cache_directory) for kernel in artifacts for a in kernel.artifacts]})
