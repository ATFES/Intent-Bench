"""Mature streaming kernels bound to the installed author programs' inputs."""
from ..corpus._common.measurement import functional_launch


def prepare(name, implementation, arguments, environment):
    import torch
    from ..corpus import PROJECT_ROOT
    from ..corpus._common.loading import load_module

    device = str(arguments[0].device)
    owners = arguments
    if implementation == "oracle":
        from ..corpus.cpu.baselines.pytorch import cpu_runtime
        if name == "attention":
            run = lambda: cpu_runtime.flash_attention_bf16_fwd(*arguments)
        elif name == "mamba_chunk_state":
            run = lambda: cpu_runtime.mamba_chunk_state_bf16_fwd(*arguments)
        elif name == "paged_decode":
            def run():
                output = cpu_runtime._paged_gqa_decode(*arguments[:6], arguments[-1]).half()
                # The author publishes NaN when no split contributes any token.
                return output.masked_fill(arguments[5][:, None, None] == 0, float("nan"))
        elif name == "gated_delta":
            q, k, v, gate, beta, scale = arguments
            group = v.shape[2] // q.shape[2]
            q, k = (value.cpu().repeat_interleave(group, dim=2) for value in (q, k))
            def run():
                output, state = cpu_runtime.recurrent_gated_delta_fwd(
                    q, k, v.cpu(), gate.cpu(), beta.cpu(), scale)
                return output.to(device), state.to(device)
        elif name == "causal_linear_attention":
            run = lambda: tuple(value.to(device) for value in
                cpu_runtime.causal_linear_attention_f32(*(value.cpu() for value in arguments)))
        elif name == "fp8_gemm":
            run = lambda: cpu_runtime.block_scaled_matmul(*(value.cpu() for value in arguments)).to(device)
        else:
            raise NotImplementedError(name)
        library = "Independent numerical oracle"
    elif name == "causal_linear_attention" and implementation == "triton":
        source = load_module(PROJECT_ROOT / "corpus/gpu/baselines/triton/tritonbench/attention/linear/chunk_linear_attn.py",
                             "author_chunk_linear_attention")
        q, k, v = (value.unsqueeze(1) for value in arguments)
        def run():
            output, state = source.chunk_linear_attn(q, k, v, scale=1.0,
                output_final_state=True, normalize=False)
            return output.squeeze(1), state.squeeze(1)
        owners = arguments, q, k, v
        library = "TritonBench chunk linear attention, both forward kernels and FP32 final state"
    elif name == "fp8_gemm" and implementation == "torch":
        from ..corpus.cpu.baselines.pytorch import cpu_runtime
        run = lambda: cpu_runtime.block_scaled_matmul(*arguments)
        library = "ATen scale decoding and cuBLAS FP32 GEMM, including all conversions"
    elif name == "fp8_gemm" and implementation == "cutile":
        from ..implementations.context import Context
        from ..source_support import official_source
        context = Context(environment.compiler, PROJECT_ROOT, None, "cutile", environment=environment)
        source = official_source(context,
            "corpus/gpu/baselines/cutile/cutile-python/gemm/block_scaled/BlockScaledMatMul_runtime.py", "author_fp8_gemm")
        lhs, lhs_scale, rhs, rhs_scale = arguments
        a, b = lhs.flatten(1), rhs.flatten(0, 1)
        a_scale = lhs_scale.view(torch.float8_e8m0fnu)
        b_scale = rhs_scale.view(torch.float8_e8m0fnu)
        run = lambda: source.cutile_block_scaled_matmul(a, a_scale, b, b_scale, compiler_timeout=60)
        owners = arguments, a, b, a_scale, b_scale
        library = "NVIDIA cuTile block-scaled E4M3 GEMM, original E8M0 scale bytes"
    elif name in {"attention", "gated_delta"} and implementation == "cutile":
        from ..implementations.context import Context
        from ..source_support import tilegym_source
        context = Context(environment.compiler, PROJECT_ROOT, None, "cutile", environment=environment)
        if name == "attention":
            module = tilegym_source(context,
                "corpus/gpu/baselines/cutile/tilegym/attention/dense/attention.py", "author_attention")
            q, k, v, scale = arguments
            run = lambda: module.tile_fmha(q, k, v, scaling=scale, is_causal=True)
            library = "NVIDIA TileGym causal grouped-query FMHA"
        else:
            module = tilegym_source(context,
                "corpus/gpu/baselines/cutile/tilegym/scan/gated_delta_recurrent/recurrent_gated_delta_rule.py",
                "author_gated_delta", needs_utils=True)
            q, k, v, gate, beta, scale = arguments
            if scale != q.shape[-1]**-0.5:
                raise ValueError("TileGym gated delta uses inverse-square-root query scaling")
            run = lambda: module.recurrent_gated_delta_rule(q, k, v, gate, beta,
                initial_state=None, output_final_state=True, compiler_timeout=60)
            library = "NVIDIA TileGym recurrent gated delta, including final state"
    elif name == "mamba_chunk_state" and implementation == "triton":
        load_module(PROJECT_ROOT / "corpus/gpu/baselines/triton/state-spaces-mamba/mamba_ssm/ops/triton/ssd_chunk_state_runtime.py",
                    "author_mamba_chunk_state_runtime")
        from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd
        run = lambda: _chunk_state_fwd(*arguments, states_in_fp32=True)
        library = "state-spaces/mamba _chunk_state_fwd, FP32 states"
    elif name == "paged_decode" and implementation == "triton":
        runtime = load_module(PROJECT_ROOT / "corpus/gpu/baselines/triton/vllm/attention/paged_decode/paged_gqa_decode_runtime.py",
                              "author_paged_gqa_runtime")
        source = runtime.RUNTIME.load_source(
            PROJECT_ROOT / "corpus/gpu/baselines/triton/vllm/attention/paged_decode/triton_decode_attention.py",
            "author_vllm_paged_gqa")
        q, key, value, offsets, indices, lengths, split_offsets, scale = arguments
        batch, heads, dimension = q.shape
        boundaries = offsets.cpu().tolist()
        counts = [end-start for start, end in zip(boundaries, boundaries[1:])]
        if len(set(counts)) != 1:
            raise NotImplementedError("this vLLM binding requires a rectangular page table")
        table = indices.reshape(batch, counts[0])
        splits = (split_offsets.numel()-1) // batch
        output = torch.empty_like(q)
        lse = torch.empty((batch, heads), dtype=torch.float32, device=device)
        workspace = torch.empty((batch, heads, splits, dimension+1), dtype=torch.float32, device=device)
        unit_scale = torch.ones((), dtype=torch.float32, device=device)
        def run():
            source.decode_attention_fwd(q, key, value, output, lse, table, lengths,
                workspace, splits, scale, page_size=key.shape[1],
                k_scale=unit_scale, v_scale=unit_scale)
            return output
        owners = arguments, output, lse, workspace, table, unit_scale
        library = "vLLM split-K paged GQA decode, complete partial and merge kernels"
    else:
        raise NotImplementedError((name, implementation))
    call = functional_launch(run)
    call.device = device
    call.owners = owners
    call.metadata["library"] = library
    return call
