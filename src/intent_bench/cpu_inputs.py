"""Independent input factories extracted from authored CPU cases.

Input geometry and numeric contracts are owned here; Intent kernels are resolved
only when an Intent implementation is selected.
"""
import torch
import math
from .structured import InputBundle, KernelSymbol
from .implementations.context import Tolerance, IntegerTolerance, SimilarityTolerance

def inputs_batched_row_affine():
    shape = (17, 257, 4093)
    x = torch.randn(shape, dtype=torch.float32)
    scale = torch.randn(shape[:2], dtype=torch.float32)
    bias = torch.randn(shape[:2], dtype=torch.float32)
    return InputBundle((x, scale, bias), 'intent.examples.kernels.pointwise.batched_affine:batched_row_affine', 'corpus/cpu/baselines/mojo/modular/pointwise/batched_affine/batched_affine_runtime.py', Tolerance(2e-06), None, ())

def inputs_integer_log2_floor():
    values = torch.randint(1, 1 << 30, (262144,), dtype=torch.int32)
    return InputBundle((values,), 'intent.examples.kernels.pointwise.while_loop:integer_log2_floor', 'integer_log2_floor', Tolerance(atol=0.0), None, ())

def inputs_paired_sum_product():
    x = torch.randn((1048576,), dtype=torch.float32)
    y = torch.randn_like(x)
    return InputBundle((x, y), 'intent.examples.kernels.pointwise.record:paired_sum_product', 'paired_sum_product', Tolerance(atol=2e-06), None, ())

def inputs_weighted_rms_norm():
    x = torch.randn((8192, 4096), dtype=torch.float32)
    weight = torch.randn((4096,), dtype=torch.float32)
    return InputBundle((x, weight, 1.0 / 4096, 1e-06), 'intent.examples.kernels.normalization.rms_norm:weighted_rms_norm', 'corpus/cpu/baselines/mojo/modular/normalization/rms_norm/rms_norm_runtime.py', Tolerance(5e-05), None, ())

def inputs_stable_softmax():
    x = torch.randn((8192, 8192), dtype=torch.float32)
    return InputBundle((x,), 'intent.examples.kernels.normalization.softmax:stable_softmax', 'corpus/cpu/baselines/mojo/modular/normalization/softmax/softmax_runtime.py', Tolerance(1e-06, 1e-05), None, ())

def inputs_weighted_layer_norm():
    x = torch.randn((8192, 4096), dtype=torch.float32)
    weight = torch.randn((4096,), dtype=torch.float32)
    bias = torch.randn((4096,), dtype=torch.float32)
    return InputBundle((x, weight, bias, 1.0 / 4096, 1e-06), 'intent.examples.kernels.normalization.layer_norm:weighted_layer_norm', 'corpus/cpu/baselines/mojo/modular/normalization/layer_norm/layer_norm_runtime.py', Tolerance(5e-05, 1e-05), None, ())

def inputs_flaggems_logsumexp():
    x = torch.randn((8192, 8192), dtype=torch.float32)
    return InputBundle((x,), 'intent.examples.kernels.normalization.logsumexp:row_logsumexp', 'logsumexp', Tolerance(atol=0.0001, rtol=1e-05), None, ())

def inputs_fused_softmax():
    x = torch.randn((8192, 8192), dtype=torch.float16)
    return InputBundle((x,), 'intent.examples.kernels.normalization.softmax:stable_softmax_f16', 'softmax', Tolerance(0.01, 0.01), None, ())

def inputs_chunked_softmax():
    x = torch.randn((8192, 32768), dtype=torch.bfloat16)
    return InputBundle((x,), 'intent.examples.kernels.normalization.softmax:chunked_softmax_bf16', 'softmax', Tolerance(0.02, 0.01), None, ())

def inputs_layer_norm_f16():
    x = torch.randn((8192, 4096), dtype=torch.float16)
    weight = torch.randn((4096,), dtype=torch.float16)
    bias = torch.randn((4096,), dtype=torch.float16)
    return InputBundle((x, weight, bias, 1.0 / 4096, 1e-05), 'intent.examples.kernels.normalization.layer_norm:layer_norm_f16', 'layer_norm', Tolerance(0.01), None, ())

def inputs_layer_norm_bf16():
    x = torch.randn((8192, 4096), dtype=torch.bfloat16)
    weight = torch.randn((4096,), dtype=torch.bfloat16)
    bias = torch.randn((4096,), dtype=torch.bfloat16)
    return InputBundle((x, weight, bias, 1.0 / 4096, 1e-05), 'intent.examples.kernels.normalization.layer_norm:layer_norm_bf16', 'layer_norm', Tolerance(0.02, 0.01), None, ())

def inputs_rms_norm_f32():
    x = torch.randn((8192, 4096), dtype=torch.float32)
    return InputBundle((x, 1.0 / 4096, 1e-05), 'intent.examples.kernels.normalization.rms_norm:rms_norm_f32', 'rms_norm', Tolerance(0.01, 0.01), None, ())

def inputs_rms_norm_bf16():
    x = torch.randn((8192, 4096), dtype=torch.bfloat16)
    weight = torch.randn((4096,), dtype=torch.bfloat16)
    return InputBundle((x, weight, 1.0 / 4096, 1e-06), 'intent.examples.kernels.normalization.rms_norm:rms_norm_bf16', 'weighted_rms_norm', Tolerance(0.02, 0.01), None, ())

def inputs_flaggems_softmax_backward():
    probabilities = torch.randn((4096, 4097), dtype=torch.float32)
    upstream = torch.randn_like(probabilities)
    return InputBundle((probabilities, upstream), 'intent.examples.kernels.backward.softmax:softmax_backward', 'softmax_backward', Tolerance(0.0001, 1e-05), None, ())

def inputs_fused_add_rms_norm():
    shape = (8192, 4096)
    x = torch.randn(shape, dtype=torch.bfloat16) * 0.5
    residual = torch.randn_like(x) * 0.5
    weight = torch.randn((4096,), dtype=torch.bfloat16)
    return InputBundle((x, residual, weight, 1.0 / 4096, 1e-06, 1.0), 'intent.examples.kernels.normalization.fused_add_rms_norm:fused_add_rms_norm', 'fused_add_rms_norm', (Tolerance(atol=0.05), Tolerance(atol=0.05)), None, ())

def inputs_dense_gemm_f32():
    a = torch.randn((1024, 1024), dtype=torch.float32)
    b = torch.randn((1024, 1024), dtype=torch.float32)
    return InputBundle((a, b), 'intent.examples.kernels.contraction.gemm:gemm_f32', 'corpus/cpu/baselines/mojo/modular/contraction/gemm/gemm_runtime.py', Tolerance(0.0002, 1e-05), None, ())

def inputs_dense_gemm():
    a = torch.randn((4096, 4096), dtype=torch.float16)
    b = torch.randn((4096, 14336), dtype=torch.float16)
    return InputBundle((a, b), 'intent.examples.kernels.contraction.gemm:gemm', 'matmul', Tolerance(0.01, 0.01), {'ACTIVATION': KernelSymbol('intent.examples.kernels.contraction.gemm:Activation', 'NONE')}, ())

def inputs_tilegym_dense_gemm():
    a = torch.randn((8192, 4096), dtype=torch.bfloat16)
    b = torch.randn((4096, 11008), dtype=torch.bfloat16)
    return InputBundle((a, b), 'intent.examples.kernels.contraction.gemm:bf16_gemm', 'matmul', Tolerance(0.05, 0.02), None, ())

def inputs_batched_gemm():
    a = torch.randn((32, 512, 1024), dtype=torch.bfloat16)
    b = torch.randn((32, 1024, 512), dtype=torch.bfloat16)
    return InputBundle((a, b), 'intent.examples.kernels.contraction.batched_gemm:batched_gemm_nn', 'matmul', Tolerance(0.05, 0.02), None, ())

def inputs_batched_gemm_tn():
    logical_a = torch.randn((32, 512, 1024), dtype=torch.bfloat16)
    logical_a /= math.sqrt(1024)
    logical_b = torch.randn((32, 1024, 512), dtype=torch.bfloat16)
    a = logical_a.transpose(-1, -2).contiguous()
    return InputBundle((a, logical_b), 'intent.examples.kernels.contraction.batched_gemm:batched_gemm_tn', 'batched_gemm_tn', Tolerance(atol=0.05), None, ())

def inputs_batched_gemm_nt():
    logical_a = torch.randn((32, 512, 1024), dtype=torch.bfloat16)
    logical_a /= math.sqrt(1024)
    logical_b = torch.randn((32, 1024, 512), dtype=torch.bfloat16)
    b = logical_b.transpose(-1, -2).contiguous()
    return InputBundle((logical_a, b), 'intent.examples.kernels.contraction.batched_gemm:batched_gemm_nt', 'batched_gemm_nt', Tolerance(atol=0.05), None, ())

def inputs_batched_gemm_tt():
    logical_a = torch.randn((32, 512, 1024), dtype=torch.bfloat16)
    logical_a /= math.sqrt(1024)
    logical_b = torch.randn((32, 1024, 512), dtype=torch.bfloat16)
    a = logical_a.transpose(-1, -2).contiguous()
    b = logical_b.transpose(-1, -2).contiguous()
    return InputBundle((a, b), 'intent.examples.kernels.contraction.batched_gemm:batched_gemm_tt', 'batched_gemm_tt', Tolerance(atol=0.05), None, ())

def inputs_mla_head_query_projection():
    source = torch.randn((1, 512, 8, 128), dtype=torch.float16) * 0.1
    weight = torch.randn((8, 512, 128), dtype=torch.float16) * 0.05
    return InputBundle((source, weight), 'intent.examples.kernels.contraction.mla:mla_head_projection', 'mla_head_projection', Tolerance(atol=0.03), None, ())

def inputs_mla_head_value_projection():
    source = torch.randn((1, 512, 8, 512), dtype=torch.float16) * 0.1
    weight = torch.randn((8, 128, 512), dtype=torch.float16) * 0.05
    return InputBundle((source, weight), 'intent.examples.kernels.contraction.mla:mla_head_projection', 'mla_head_projection', Tolerance(atol=0.03), None, ())

def inputs_qkv_projection():
    tokens = hidden = projection = 4096
    x = torch.randn((tokens, hidden), dtype=torch.float16)
    weights = tuple((torch.randn((hidden, projection), dtype=torch.float16) for _ in range(3)))
    packed_weights = torch.stack(weights)
    return InputBundle((x, packed_weights), 'intent.examples.kernels.contraction.qkv:fused_qkv_projection', 'qkv_projection', Tolerance(atol=0.02, rtol=0.01), None, ())

def inputs_gated_dual_gemm():
    x = torch.randn((2048, 4096), dtype=torch.float16)
    x /= math.sqrt(4096)
    gate_weight = torch.randn((4096, 4096), dtype=torch.float16)
    value_weight = torch.randn_like(gate_weight)
    return InputBundle((x, gate_weight, value_weight), 'intent.examples.kernels.contraction.dual_gemm:gated_dual_gemm', 'gated_dual_gemm', Tolerance(atol=0.05), None, ())

def inputs_fp8_gemm():
    lhs = torch.randn((4096, 4096), dtype=torch.float16).to(torch.float8_e4m3fn)
    rhs = torch.randn((14336, 4096), dtype=torch.float16).to(torch.float8_e4m3fn)
    return InputBundle((lhs, rhs), 'intent.examples.kernels.contraction.weight_only_int4:fp8_e4m3_matmul', 'fp8_matmul', Tolerance(atol=0.5, rtol=0.05), None, (), prequantized_reference='fp8_matmul_accumulator')

def inputs_fp8_e5m2_gemm():
    lhs = torch.randn((1024, 1024), dtype=torch.float16).to(torch.float8_e5m2)
    rhs = torch.randn((1024, 1024), dtype=torch.float16).to(torch.float8_e5m2)
    return InputBundle((lhs, rhs), 'intent.examples.kernels.contraction.weight_only_int4:fp8_e5m2_matmul', 'fp8_matmul', SimilarityTolerance(max_error=0.001), None, ())

def inputs_quantized_gemm():
    lhs = torch.randn((4096, 4096), dtype=torch.float16) / math.sqrt(4096)
    rhs = torch.randn((4096, 14336), dtype=torch.float16)
    bias = torch.randn((14336,), dtype=torch.float32) * 0.25
    residual = torch.randn((4096, 14336), dtype=torch.float16) * 0.25
    scale = torch.linspace(1.0 / 48.0, 1.0 / 24.0, 14336, dtype=torch.float32)
    return InputBundle((lhs, rhs, bias, residual, scale), 'intent.examples.kernels.contraction.gemm:quantized_gemm', 'quantized_gemm', IntegerTolerance(max_abs=2), None, ())

def inputs_weight_only_int4():
    activation = torch.randn((512, 2048), dtype=torch.float16) * 0.05
    logical = torch.randint(-8, 8, (2048, 4096), dtype=torch.int32)
    packed = torch.zeros((256, 4096), dtype=torch.int32)
    for lane in range(8):
        packed |= (logical[lane::8] & 15) << 4 * lane
    scales = 0.01 + 0.02 * torch.rand((32, 4096), dtype=torch.float16)
    return InputBundle((activation, packed, scales), 'intent.examples.kernels.contraction.weight_only_int4:weight_only_int4_matmul', 'weight_only_int4_matmul', Tolerance(atol=0.04), None, ())

def inputs_w4a8_gemm():
    activation = torch.randint(-128, 128, (4096, 4096), dtype=torch.int8)
    packed = torch.randint(0, 256, (14336, 2048), dtype=torch.uint8)
    return InputBundle((activation, packed), 'intent.examples.kernels.contraction.weight_only_int4:w4a8_packed_matmul', 'w4a8_packed_matmul', Tolerance(atol=0.0), None, ())

def inputs_bitnet_int2_decode():
    activation = torch.randint(-8, 8, (1, 4096), dtype=torch.int8)
    logical = torch.randint(0, 2, (4096, 4096), dtype=torch.int8).reshape(4096, 256, 4, 4)
    packed = torch.zeros((4096, 256, 4), dtype=torch.uint8)
    for lane in range(4):
        packed |= logical[:, :, lane, :].to(torch.uint8) << lane * 2
    return InputBundle((activation, packed.flatten(1)), 'intent.examples.kernels.contraction.weight_only_int4:bitnet_int2_matmul', 'bitnet_int2_matmul', Tolerance(atol=0.0), None, ())

def inputs_dequant_bf16_fp4():
    activation = torch.randn((4096, 4096), dtype=torch.bfloat16) * 0.125
    packed = torch.randint(0, 256, (4096, 2048), dtype=torch.uint8)
    return InputBundle((activation, packed), 'intent.examples.kernels.contraction.weight_only_int4:dequant_bf16_fp4_matmul', 'dequant_bf16_fp4_matmul', Tolerance(atol=1.0, rtol=0.02), None, ())

def inputs_deepgemm_fp8_2xacc():
    lhs_values = torch.randn((4096, 4096), dtype=torch.bfloat16).view(4096, 32, 128)
    lhs_max = lhs_values.abs().float().amax(dim=2).clamp_min(0.0001)
    lhs = (lhs_values * (448.0 / lhs_max[:, :, None])).to(torch.float8_e4m3fn)
    rhs_values = torch.randn((4096, 4096), dtype=torch.bfloat16).view(32, 128, 32, 128)
    rhs_max = rhs_values.abs().float().amax(dim=(1, 3), keepdim=True).clamp_min(0.0001)
    rhs = (rhs_values * (448.0 / rhs_max)).to(torch.float8_e4m3fn).view(4096, 32, 128)
    return InputBundle((lhs, rhs, lhs_max / 448.0, (rhs_max / 448.0).view(32, 32)), 'intent.examples.kernels.contraction.block_scaled:deepgemm_fp8_2xacc', 'deepgemm_fp8_2xacc', Tolerance(atol=1.0, rtol=0.02), None, ())

def inputs_scaled_fp8_gemm():
    lhs = torch.randn((4096, 4096), dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    rhs = torch.randn((4096, 14336), dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    return InputBundle((lhs, rhs, 1.0, 1.0), 'intent.examples.kernels.contraction.block_scaled:scaled_fp8_matmul', 'bench_scaled_fp8_gemm', Tolerance(atol=0.5, rtol=0.05), None, ())

def inputs_vector_dot():
    lhs = torch.randn((1048576,), dtype=torch.float32)
    rhs = torch.randn_like(lhs)
    return InputBundle((lhs, rhs), 'intent.examples.kernels.contraction.vector:vector_dot', 'vector_dot', Tolerance(atol=0.002, rtol=1e-05), None, ())

def inputs_matrix_vector():
    matrix = torch.randn((1024, 1024), dtype=torch.float32)
    vector = torch.randn((1024,), dtype=torch.float32)
    return InputBundle((matrix, vector), 'intent.examples.kernels.contraction.vector:matrix_vector', 'matmul', Tolerance(atol=0.0002, rtol=1e-05), None, ())

def inputs_vector_matrix():
    vector = torch.randn((1024,), dtype=torch.float32)
    matrix = torch.randn((1024, 1024), dtype=torch.float32)
    return InputBundle((vector, matrix), 'intent.examples.kernels.contraction.vector:vector_matrix', 'matmul', Tolerance(atol=0.0002, rtol=1e-05), None, ())

def inputs_vector_outer():
    lhs = torch.randn((1024,), dtype=torch.float32)
    rhs = torch.randn_like(lhs)
    return InputBundle((lhs, rhs), 'intent.examples.kernels.contraction.vector:vector_outer', 'vector_outer', Tolerance(atol=0.0), None, ())

def inputs_mxfp8_gemm():

    def _block_quantize_mxfp8(x, block_size):
        dtype_max = torch.finfo(torch.float8_e4m3fn).max
        x_block = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
        scale = torch.max(x_block.abs(), dim=-1, keepdims=True)[0]
        scale = torch.clamp(scale / dtype_max, min=1e-12)
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
        x_q = (x_block / scale).to(torch.float8_e4m3fn).reshape(x.shape)
        scale = scale.to(torch.float8_e8m0fnu).squeeze(-1)
        return (x_q, scale)
    (m, k, n, block) = (4096, 4096, 14336, 32)
    (lhs_rows, lhs_scale_rows) = _block_quantize_mxfp8(torch.randn((m, k), dtype=torch.float32), block)
    (rhs_rows, rhs_scale_rows) = _block_quantize_mxfp8(torch.randn((n, k), dtype=torch.float32), block)
    lhs = lhs_rows.view(m, k // block, block)
    lhs_scale = lhs_scale_rows.view(torch.uint8)
    rhs = rhs_rows.T.view(k // block, block, n)
    rhs_scale = rhs_scale_rows.T.view(torch.uint8)
    return InputBundle((lhs, lhs_scale, rhs, rhs_scale), 'intent.examples.kernels.contraction.block_scaled:block_scaled_matmul', 'block_scaled_matmul', Tolerance(atol=1.0, rtol=0.02), None, ())

def inputs_block_sparse_gemm():
    dimension = 4096
    lhs = torch.randn((dimension, dimension), dtype=torch.float16)
    rhs = torch.randn_like(lhs)
    mask = (torch.rand((32, 32, 128)) > 0.5).to(torch.uint8)
    return InputBundle((lhs, rhs, mask), 'intent.examples.kernels.contraction.block_sparse:block_sparse_matmul', 'block_sparse_matmul', Tolerance(atol=0.05, rtol=0.02), None, ())

def inputs_causal_attention_f32():
    q = torch.randn((8, 128, 32), dtype=torch.float32)
    (k, v) = (torch.randn_like(q), torch.randn_like(q))
    return InputBundle((q, k, v, 32 ** (-0.5)), 'intent.examples.kernels.streaming.attention_f32:causal_attention_f32', 'corpus/cpu/baselines/mojo/intentdsl/attention/causal/causal_runtime.py', Tolerance(0.0002, 1e-05), None, ())

def inputs_causal_linear_attention_f32():
    q = torch.randn((8, 128, 32), dtype=torch.float32)
    (k, v) = (torch.randn_like(q), torch.randn_like(q))
    return InputBundle((q, k, v), 'intent.examples.kernels.streaming.attention_f32:causal_linear_attention_f32', 'corpus/cpu/baselines/mojo/intentdsl/attention/linear/linear_runtime.py', Tolerance(0.0002, 1e-05), None, ())

def inputs_flash_attention_bf16():
    dimension = 128
    q = torch.randn((2, 32, 4096, dimension), dtype=torch.bfloat16)
    k = torch.randn((2, 8, 4096, dimension), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    return InputBundle((q, k, v, dimension ** (-0.5)), 'intent.examples.kernels.streaming.attention:flash_attention_bf16_fwd', 'flash_attention_bf16_fwd', Tolerance(atol=0.05, rtol=0.02), {'HEAD_GROUP': 4, 'CAUSAL': True}, ())

def inputs_flash_attention_f16():
    q = torch.randn((4, 32, 4096, 128), dtype=torch.float16)
    (k, v) = (torch.randn_like(q), torch.randn_like(q))
    return InputBundle((q, k, v, 128 ** (-0.5)), 'intent.examples.kernels.streaming.attention:flash_attention_fwd', 'flash_attention_bf16_fwd', Tolerance(atol=0.02, rtol=0.02), {'CAUSAL': True}, ())

def inputs_flash_gqa_attention_f16():
    q = torch.randn((4, 32, 4096, 128), dtype=torch.float16)
    k = torch.randn((4, 8, 4096, 128), dtype=torch.float16)
    v = torch.randn_like(k)
    return InputBundle((q, k, v, 128 ** (-0.5)), 'intent.examples.kernels.streaming.attention:flash_gqa_attention_fwd', 'flash_attention_bf16_fwd', Tolerance(atol=0.05, rtol=0.02), {'HEAD_GROUP': 4, 'CAUSAL': True}, ())

def inputs_continuous_gqa_decode():
    q = torch.randn((32, 32, 128), dtype=torch.float16)
    k = torch.randn((32, 8192, 8, 128), dtype=torch.float16)
    v = torch.randn_like(k)
    mask = torch.ones((32, 8192, 8), dtype=torch.uint8)
    return InputBundle((q, k, v, mask, 128 ** (-0.5)), 'intent.examples.kernels.streaming.attention:continuous_gqa_decode', 'continuous_gqa_decode', Tolerance(atol=0.05, rtol=0.02), {'HEAD_GROUP': 4}, ())

def inputs_flash_varlen_gqa_prefill():
    lengths = torch.tensor((4096, 3968, 3840, 3712, 3584, 3456, 3328, 3200), dtype=torch.int32)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int()))
    q = torch.randn((29184, 32, 128), dtype=torch.float16)
    k = torch.randn((29184, 8, 128), dtype=torch.float16)
    v = torch.randn_like(k)
    return InputBundle((q, k, v, lengths, offsets, 128 ** (-0.5)), 'intent.examples.kernels.streaming.attention:flash_varlen_gqa_prefill', 'flash_varlen_gqa_prefill', Tolerance(atol=0.05, rtol=0.02), {'HEAD_GROUP': 4}, ())

def inputs_flash_attention_bias():
    q = torch.randn((4, 32, 4096, 128), dtype=torch.float16)
    (k, v) = (torch.randn_like(q), torch.randn_like(q))
    bias = torch.randn((4, 32, 4096), dtype=torch.float32)
    return InputBundle((q, k, v, bias, 128 ** (-0.5)), 'intent.examples.kernels.streaming.attention:flash_attention_bias_fwd', 'bench_flash_attention_bias', Tolerance(atol=0.05, rtol=0.02), None, ())

def inputs_flash_varlen_attention():
    VARLEN_BATCH = 8
    VARLEN_TOTAL_TOKENS = 29114
    lengths = torch.full((VARLEN_BATCH,), VARLEN_TOTAL_TOKENS // VARLEN_BATCH, dtype=torch.int32)
    lengths[:VARLEN_TOTAL_TOKENS % VARLEN_BATCH] += 1
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int()))
    q = torch.randn((VARLEN_TOTAL_TOKENS, 128), dtype=torch.float16)
    (k, v) = (torch.randn_like(q), torch.randn_like(q))
    return InputBundle((q, k, v, lengths, offsets, 128 ** (-0.5)), 'intent.examples.kernels.streaming.attention:flash_varlen_attention_fwd', 'bench_flash_varlen_attention', Tolerance(atol=0.05, rtol=0.02), {'CAUSAL': True}, ())

def inputs_attention_sink_prefill():
    q = torch.randn((1, 32, 4096, 128), dtype=torch.bfloat16)
    k = torch.randn((1, 8, 4096, 128), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sinks = torch.randn((32,), dtype=torch.bfloat16)
    return InputBundle((q, k, v, sinks, 128 ** (-0.5)), 'intent.examples.kernels.streaming.attention_specialized:attention_sink_prefill', 'attention_sink_prefill', Tolerance(atol=0.05, rtol=0.02), {'HEAD_GROUP': 4}, ())

def inputs_gemma_gqa_prefill():
    q = torch.randn((2, 32, 4096, 128), dtype=torch.bfloat16)
    k = torch.randn((2, 8, 4096, 128), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    return InputBundle((q, k, v, 128 ** (-0.5)), 'intent.examples.kernels.streaming.attention_specialized:gemma_gqa_prefill', 'gemma_gqa_prefill', Tolerance(atol=0.05, rtol=0.02), {'HEAD_GROUP': 4, 'WINDOW': 1024, 'SOFT_CAP': 50.0}, ())

def inputs_sliding_window_gqa_prefill():
    q = torch.randn((2, 32, 4096, 128), dtype=torch.float16)
    k = torch.randn((2, 8, 4096, 128), dtype=torch.float16)
    v = torch.randn_like(k)
    return InputBundle((q, k, v, 128 ** (-0.5)), 'intent.examples.kernels.streaming.attention_specialized:sliding_window_gqa_prefill', 'sliding_window_gqa_prefill', Tolerance(atol=0.05, rtol=0.02), {'HEAD_GROUP': 4, 'WINDOW': 1024}, ())

def inputs_block_causal_attention():
    q = torch.randn((2, 4096, 16, 128), dtype=torch.float16)
    (k, v) = (torch.randn_like(q), torch.randn_like(q))
    return InputBundle((q, k, v, 128 ** (-0.5)), 'intent.examples.kernels.streaming.attention_specialized:block_causal_attention_fwd', 'block_causal_attention', Tolerance(atol=0.05, rtol=0.02), {'BLOCK': 64}, ())

def inputs_varlen_block_causal_attention():
    offsets = torch.tensor((0, 4096, 7936, 11520, 14848), dtype=torch.int32)
    q = torch.randn((14848, 16, 128), dtype=torch.float16)
    (k, v) = (torch.randn_like(q), torch.randn_like(q))
    return InputBundle((q, k, v, offsets, 128 ** (-0.5)), 'intent.examples.kernels.streaming.attention_specialized:varlen_block_causal_attention_fwd', 'varlen_block_causal_attention', Tolerance(atol=0.05, rtol=0.02), {'BLOCK': 64}, ())

def inputs_native_sparse_attention():
    q = torch.randn((2, 4096, 32, 128), dtype=torch.float16)
    k = torch.randn((2, 4096, 4, 128), dtype=torch.float16)
    v = torch.randn_like(k)
    blocks = torch.arange(64, dtype=torch.int32).reshape(1, 1, 1, 64).expand(2, 4096, 4, 64).contiguous()
    return InputBundle((q, k, v, blocks, 128 ** (-0.5)), 'intent.examples.kernels.streaming.attention_specialized:native_sparse_attention_fwd', 'native_sparse_attention', Tolerance(atol=0.05, rtol=0.02), {'HEAD_GROUP': 8, 'BLOCK': 64}, ())

def inputs_varlen_gqa_decode_with_sink_logits():
    lengths = 4096 - torch.arange(16, dtype=torch.int32) * 64
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int()))
    q = torch.randn((16, 32, 64), dtype=torch.float16)
    k = torch.randn((57856, 8, 64), dtype=torch.float16)
    v = torch.randn_like(k)
    sink = torch.zeros((32,), dtype=torch.float32)
    return InputBundle((q, k, v, offsets, sink, 64 ** (-0.5)), 'intent.examples.kernels.streaming.attention:varlen_gqa_decode_with_sink_logits', 'varlen_gqa_decode_with_sink_logits', (Tolerance(atol=0.05, rtol=0.02), Tolerance(atol=0.05, rtol=0.02)), {'HEAD_GROUP': 4, 'MAX_BLOCKS': 64}, ())

def inputs_paged_gqa_decode_attention():
    lengths = torch.tensor((8191, 7937, 7683, 7429, 7175, 6921, 6667, 6413), dtype=torch.int32)
    counts = (lengths + 63) // 64
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), counts.cumsum(0).int()))
    pages = int(offsets[-1])
    indices = torch.randperm(pages).to(torch.int32)
    q = torch.randn((8, 32, 128), dtype=torch.float16) * 0.5
    key = torch.randn((pages, 64, 8, 128), dtype=torch.float16) * 0.5
    value = torch.randn_like(key) * 0.5
    return InputBundle((q, key, value, offsets, indices, lengths, 128 ** (-0.5)), 'intent.examples.kernels.streaming.paged_attention:paged_gqa_decode_attention', 'paged_gqa_decode_attention', Tolerance(atol=0.03), {'PAGE_SIZE': 64, 'HEAD_GROUP': 4}, ())

def inputs_gelu():
    x = torch.randn((8192, 4096), dtype=torch.float16)
    return InputBundle((x,), 'intent.examples.kernels.activation.pointwise:gelu_tanh', 'gelu', Tolerance(atol=0.01, rtol=0.01), None, ())

def inputs_relu():
    x = torch.randn((8192, 4096), dtype=torch.float16)
    return InputBundle((x,), 'intent.examples.kernels.activation.pointwise:relu_forward', 'relu', Tolerance(atol=0.0), None, ())

def inputs_flaggems_addcmul():
    shape = (64, 128, 4096)
    x = torch.randn(shape, dtype=torch.bfloat16)
    scale = torch.randn(shape[:2], dtype=torch.bfloat16)
    bias = torch.randn(shape[:2], dtype=torch.bfloat16)
    return InputBundle((x, scale, bias), 'intent.examples.kernels.activation.pointwise:addcmul_broadcast_bf16', 'addcmul', Tolerance(atol=0.02, rtol=0.01), None, ())

def inputs_swiglu():
    gate = torch.randn((8192, 14336), dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    return InputBundle((gate, up), 'intent.examples.kernels.activation.swiglu:swiglu_forward', 'swiglu', Tolerance(atol=0.02, rtol=0.01), None, ())

def inputs_swiglu_backward():
    shape = (4096, 4096)
    dc = torch.randn(shape, dtype=torch.bfloat16) * 0.5
    a = torch.randn(shape, dtype=torch.bfloat16) * 0.5
    b = torch.randn(shape, dtype=torch.bfloat16) * 0.5
    return InputBundle((dc, a, b), 'intent.examples.kernels.backward.swiglu:swiglu_backward', 'swiglu_backward', (Tolerance(atol=0.05), Tolerance(atol=0.05)), None, ())

def inputs_silu_and_mul_packed():
    SWIGLU_FEATURES = 14336
    packed = torch.randn((4096, 2 * SWIGLU_FEATURES), dtype=torch.bfloat16)
    return InputBundle((packed,), 'intent.examples.kernels.activation.swiglu:silu_and_mul_packed', 'silu_and_mul_packed', Tolerance(atol=0.02, rtol=0.01), None, ())

def inputs_flaggems_transpose_copy():
    x = torch.randn((4093, 8191), dtype=torch.float16)
    return InputBundle((x,), 'intent.examples.kernels.layout.transpose:matrix_transpose', 'transpose', Tolerance(0.0), None, ())

def inputs_embedding_lookup():
    table = torch.randn((32768, 4096), dtype=torch.bfloat16)
    indices = torch.randint(0, 32768, (8, 2048), dtype=torch.int64).reshape(-1)
    return InputBundle((table, indices), 'intent.examples.kernels.backward.embedding:embedding_forward_lookup_bf16', 'index_select', Tolerance(atol=0.0), None, ())

def inputs_embedding_lookup_f32():
    table = torch.randn((8192, 1021), dtype=torch.float32)
    indices = torch.randint(0, 8192, (8192,), dtype=torch.int32)
    return InputBundle((table, indices), 'intent.examples.kernels.backward.embedding:embedding_forward_lookup', 'embedding_forward_lookup', Tolerance(atol=0.0), None, ())

def inputs_index_select():
    source = torch.randn((65536, 4096), dtype=torch.float16)
    indices = torch.arange(0, 32768 * 2, 2, dtype=torch.int64)
    return InputBundle((source, indices), 'intent.examples.kernels.indexing.relations:index_select_rows', 'index_select', Tolerance(atol=0.0), None, ())

def inputs_scalar_table_lookup():
    labels = torch.randint(0, 65536, (65536,), dtype=torch.int32)
    table = torch.randn((65536, 128), dtype=torch.float32)
    return InputBundle((labels, table), 'intent.examples.kernels.indexing.relations:scalar_table_lookup', 'scalar_table_lookup', Tolerance(atol=0.0), None, ())

def inputs_shifted_row_copy():
    OFFSET_FEATURES = 1024
    OFFSET_ROWS = 4096
    x = torch.randn((OFFSET_ROWS, OFFSET_FEATURES), dtype=torch.float32)
    return InputBundle((x,), 'intent.examples.kernels.indexing.relations:shifted_row_copy', 'shifted_row_copy', Tolerance(atol=0.0), None, ())

def inputs_roll_rows_forward():
    x = torch.randn((8192, 4096), dtype=torch.float16).reshape(-1)
    return InputBundle((x,), 'intent.examples.kernels.indexing.relations:roll_rows_forward', 'roll_rows_forward', Tolerance(atol=0.0), {'ROW_WIDTH': 4096}, ())

def inputs_grouped_query_head_add():
    GQA_KEY_HEADS = 8
    GQA_QUERY_HEADS = 32
    GQA_TOKENS = 4096
    query = torch.randn((GQA_QUERY_HEADS, GQA_TOKENS), dtype=torch.float16)
    key = torch.randn((GQA_KEY_HEADS, GQA_TOKENS), dtype=torch.float16)
    return InputBundle((query, key), 'intent.examples.kernels.indexing.relations:grouped_query_head_add', 'grouped_query_head_add', Tolerance(atol=0.0), None, ())

def inputs_alternating_signed_indices():
    SELECT_COLUMNS = 8191
    SELECT_ROWS = 4093
    shape_source = torch.empty((SELECT_ROWS, SELECT_COLUMNS), dtype=torch.float32)
    return InputBundle((shape_source,), 'intent.examples.kernels.pointwise.select:alternating_signed_indices', 'alternating_signed_indices', Tolerance(atol=0.0), None, ())

def inputs_csr_spmv():
    rows = columns = 32768
    nonzeros_per_row = 32
    nonzeros = rows * nonzeros_per_row
    row_offsets = torch.arange(rows + 1, dtype=torch.int32) * nonzeros_per_row
    column_indices = torch.randint(0, columns, (nonzeros,), dtype=torch.int32)
    values = torch.randn((nonzeros,), dtype=torch.float32) * 0.1
    vector = torch.randn((columns,), dtype=torch.float32)
    return InputBundle((row_offsets, column_indices, values, vector), 'intent.examples.kernels.sparse.csr_spmv:csr_spmv', 'csr_spmv', Tolerance(atol=2e-05), None, ())

def inputs_csr_spmm():
    rows = columns = 8192
    nonzeros_per_row = 32
    features = 128
    nonzeros = rows * nonzeros_per_row
    row_offsets = torch.arange(rows + 1, dtype=torch.int32) * nonzeros_per_row
    column_indices = torch.randint(0, columns, (nonzeros,), dtype=torch.int32)
    values = torch.randn((nonzeros,), dtype=torch.float32) * 0.1
    dense = torch.randn((columns, features), dtype=torch.float32)
    return InputBundle((row_offsets, column_indices, values, dense), 'intent.examples.kernels.sparse.csr_spmm:csr_spmm', 'csr_spmm', Tolerance(atol=3e-05), None, ())

def inputs_roi_align_center_sample():
    (batch_size, channels, height, width) = (8, 64, 128, 128)
    rois_count = 2048
    (pooled_height, pooled_width) = (7, 7)
    feature = torch.randn((batch_size, channels, height, width), dtype=torch.float32)
    batch = torch.randint(0, batch_size, (rois_count, 1)).float()
    upper_left = torch.rand((rois_count, 2)) * (height - 16)
    size = torch.rand((rois_count, 2)) * 48 + 8
    lower_right = torch.minimum(upper_left + size, torch.tensor([width - 1, height - 1]))
    rois = torch.cat((batch, upper_left, lower_right), dim=1)
    return InputBundle((feature, rois), 'intent.examples.kernels.vision.roi_align:roi_align_center_sample', 'roi_align_center_sample', Tolerance(atol=0.0001), None, ())

def inputs_max_pool2d():
    x = torch.randn((8, 32, 128, 128), dtype=torch.float16)
    return InputBundle((x,), 'intent.examples.kernels.vision.max_pool:max_pool2d', 'max_pool2d', Tolerance(atol=0.0), None, ())

def inputs_flaggems_max_pool2d_with_indices():
    x = torch.randn((8, 32, 128, 128), dtype=torch.float16)
    return InputBundle((x,), 'intent.examples.kernels.vision.max_pool_with_indices:max_pool2d_with_indices', 'max_pool2d_with_indices', (Tolerance(atol=0.0), Tolerance(atol=0.0)), None, ())

def inputs_greedy_nms():
    upper_left = torch.rand((32, 1024, 2), dtype=torch.float32) * 0.8
    size = torch.rand_like(upper_left) * 0.2 + 0.01
    boxes = torch.cat((upper_left, upper_left + size), dim=2)
    threshold = 0.5
    return InputBundle((boxes, threshold), 'intent.examples.kernels.vision.nms:greedy_nms', 'greedy_nms', Tolerance(atol=0.0), None, ())

def inputs_rotary_embedding_flat():
    values = torch.randn((65536, 128), dtype=torch.float16)
    cosine = torch.randn((2048, 64), dtype=torch.float16)
    sine = torch.randn_like(cosine)
    return InputBundle((values, cosine, sine), 'intent.examples.kernels.position.rope:rotary_embedding_flat', 'rotary_embedding_flat', Tolerance(atol=0.01), {'HEADS': 32}, ())

def inputs_rotary_embedding_bf16():
    (batch, sequence, heads, dimension) = (4, 4096, 32, 128)
    values = torch.randn((batch, sequence, heads, dimension), dtype=torch.bfloat16)
    positions = torch.arange(sequence, dtype=torch.float32)
    inverse = 1.0 / 10000 ** (torch.arange(0, dimension, 2, dtype=torch.float32) / dimension)
    angles = torch.outer(positions, inverse)
    (cosine, sine) = (angles.cos(), angles.sin())
    return InputBundle((values, cosine, sine), 'intent.examples.kernels.position.rope:rotary_embedding_bf16', 'rotary_embedding_bf16', Tolerance(atol=0.005, rtol=0.05), None, ())

def inputs_flaggems_conv1d():
    x = torch.randn((64, 16384), dtype=torch.float16)
    weight = torch.randn((5,), dtype=torch.float16)
    return InputBundle((x, weight), 'intent.examples.kernels.convolution.direct:conv1d_same', 'conv1d_same', Tolerance(atol=0.02, rtol=0.01), None, ())

def inputs_causal_conv1d():
    CAUSAL_CONV_BATCH = 8
    CAUSAL_CONV_CHANNELS = 2048
    CAUSAL_CONV_LENGTH = 4096
    CAUSAL_CONV_WIDTH = 4
    x = torch.randn((CAUSAL_CONV_BATCH, CAUSAL_CONV_CHANNELS, CAUSAL_CONV_LENGTH), dtype=torch.float16) * 0.1
    weight = torch.randn((CAUSAL_CONV_CHANNELS, CAUSAL_CONV_WIDTH), dtype=torch.float16) * 0.1
    bias = torch.randn((CAUSAL_CONV_CHANNELS,), dtype=torch.float16) * 0.1
    return InputBundle((x, weight, bias), 'intent.examples.kernels.convolution.direct:causal_depthwise_conv1d', 'causal_conv1d', Tolerance(atol=0.003), {'SILU': True}, ())

def inputs_causal_conv1d_bf16():
    x = torch.randn((4, 4096, 4096), dtype=torch.bfloat16).transpose(1, 2)
    weight = torch.randn((4096, 4), dtype=torch.bfloat16)
    bias = torch.randn((4096,), dtype=torch.bfloat16)
    return InputBundle((x, weight, bias), 'intent.examples.kernels.convolution.direct:causal_depthwise_conv1d_bf16', 'causal_conv1d', Tolerance(atol=0.05, rtol=0.05), {'SILU': True}, ())

def inputs_conv2d():
    CONV2D_BATCH = 16
    CONV2D_FILTER_HEIGHT = 3
    CONV2D_FILTER_WIDTH = 3
    CONV2D_HEIGHT = 256
    CONV2D_WIDTH = 256
    x = torch.randn((CONV2D_BATCH, CONV2D_HEIGHT, CONV2D_WIDTH), dtype=torch.float16) * 0.1
    weight = torch.randn((CONV2D_FILTER_HEIGHT, CONV2D_FILTER_WIDTH), dtype=torch.float16) * 0.1
    return InputBundle((x, weight), 'intent.examples.kernels.convolution.direct:conv2d_same', 'conv2d', Tolerance(atol=0.003), None, ())

def inputs_conv2d_nhwc():
    x = torch.randn((32, 128, 128, 256), dtype=torch.float16)
    weight = torch.randn((3, 3, 256, 512), dtype=torch.float16)
    return InputBundle((x, weight), 'intent.examples.kernels.convolution.direct:conv2d_nhwc', 'conv2d_nhwc', Tolerance(atol=0.02, rtol=0.02), None, ())

def inputs_bitonic_sort():
    values = torch.randn((4096, 256), dtype=torch.float32)
    return InputBundle((values,), 'intent.examples.kernels.sorting.bitonic:bitonic_sort_rows', 'bitonic_sort', Tolerance(atol=0.0), None, ())

def inputs_insertion_top_k():
    logits = torch.randn((1024, 4093), dtype=torch.float32)
    return InputBundle((logits,), 'intent.examples.kernels.sampling.top_k:insertion_top_k', 'insertion_top_k', (Tolerance(atol=0.0), Tolerance(atol=0.0)), None, ())

def inputs_smith_waterman():
    query = torch.randint(0, 20, (128, 128), dtype=torch.int32)
    reference = torch.randint(0, 20, (128, 128), dtype=torch.int32)
    return InputBundle((query, reference), 'intent.examples.kernels.dynamic_programming.smith_waterman:smith_waterman_score', 'smith_waterman', Tolerance(atol=0.0), None, ())

def inputs_viterbi():
    emissions = torch.randn((64, 256, 64), dtype=torch.float32) * 0.1
    transitions = torch.randn((64, 64), dtype=torch.float32) * 0.1
    return InputBundle((emissions, transitions), 'intent.examples.kernels.dynamic_programming.viterbi:viterbi_decode', 'viterbi', (Tolerance(atol=0.0), Tolerance(atol=2e-05)), None, ())

def inputs_radix2_fft():
    real = torch.randn((1024, 1024), dtype=torch.float32)
    imag = torch.randn_like(real)
    lanes = torch.arange(512, dtype=torch.float32)
    twiddle_real = torch.empty((10, 512), dtype=torch.float32)
    twiddle_imag = torch.empty_like(twiddle_real)
    for stage in range(10):
        angle = -2.0 * math.pi * lanes / (1 << stage + 1)
        twiddle_real[stage] = torch.cos(angle)
        twiddle_imag[stage] = torch.sin(angle)
    return InputBundle((real, imag, twiddle_real, twiddle_imag), 'intent.examples.kernels.spectral.fft:radix2_fft', 'radix2_fft', (Tolerance(atol=0.0003), Tolerance(atol=0.0003)), None, ())

def inputs_flaggems_cumsum():
    x = torch.randn((8192, 8192), dtype=torch.float32)
    return InputBundle((x,), 'intent.examples.kernels.streaming.ordered_prefix:row_cumsum_f32', 'cumsum', Tolerance(atol=0.002, rtol=1e-05), None, ())

def inputs_ordered_prefix():
    x = torch.randn((512, 17, 31), dtype=torch.float32) * 0.01
    return InputBundle((x,), 'intent.examples.kernels.streaming.ordered_prefix:ordered_product_prefix', 'ordered_prefix', Tolerance(atol=2e-05), None, ())

def inputs_compact_nonzero():
    values = torch.randn((64, 4096), dtype=torch.float32)
    values[torch.rand_like(values) < 0.7] = 0.0
    return InputBundle((values,), 'intent.examples.kernels.compaction.nonzero:compact_nonzero_rows', 'compact_nonzero', (Tolerance(atol=0.0), Tolerance(atol=0.0)), None, ())

def inputs_sorted_nucleus_cutoff():
    ranks = torch.arange(4093, dtype=torch.float32)
    probabilities = torch.softmax(-ranks / 256.0, dim=0)[None, :].expand(1024, -1).contiguous()
    return InputBundle((probabilities, 0.9), 'intent.examples.kernels.sampling.nucleus:sorted_nucleus_cutoff', 'nucleus', (Tolerance(atol=0.0002), Tolerance(atol=0.0)), None, ())

def inputs_histogram():
    samples = torch.randint(0, 256, (8 * 1024 * 1024,), dtype=torch.int32).to(torch.float32)
    return InputBundle((samples,), 'intent.examples.kernels.statistics.histogram:histogram_256', 'histogram', Tolerance(atol=0.0), None, ())

def inputs_flash_cross_entropy_bf16():
    logits = torch.randn((8192, 32768), dtype=torch.bfloat16)
    labels = torch.randint(0, 32768, (8192,), dtype=torch.int64)
    return InputBundle((logits, labels), 'intent.examples.kernels.loss.cross_entropy:flash_cross_entropy_bf16', 'flash_cross_entropy_bf16', (Tolerance(atol=0.05, rtol=0.01), Tolerance(atol=0.05, rtol=0.01)), {'IGNORE_INDEX': -100, 'Z_LOSS_SCALE': 0.0001}, ())

def inputs_dropout():
    DROPOUT_FEATURES = 4096
    DROPOUT_ROWS = 8192
    DROPOUT_SEED = 12345
    DROP_PROBABILITY = 0.1

    def _mix_dropout_seed(seed: int) -> int:
        mixed = int(seed) * 2654435761 & 4294967295
        if mixed >= 2147483648:
            mixed -= 4294967296
        return mixed
    x = torch.randn((DROPOUT_ROWS, DROPOUT_FEATURES), dtype=torch.float16)
    probability = DROP_PROBABILITY
    mixed_seed = _mix_dropout_seed(DROPOUT_SEED)
    inverse_keep = 1.0 / (1.0 - probability)
    return InputBundle((x, mixed_seed, probability, inverse_keep), 'intent.examples.kernels.regularization.dropout:xor_shift_dropout', 'dropout', Tolerance(atol=0.0), None, ())

def inputs_barrier_option_paths():
    return InputBundle((17, 100.0, 95.0, 150.0, 0.0002, 0.01), 'intent.examples.kernels.simulation.monte_carlo:barrier_option_paths', 'barrier_option_paths', Tolerance(atol=0.0003), None, ())

def inputs_boolean_reduction():
    BOOLEAN_COLUMNS = 4093
    BOOLEAN_ROWS = 8192
    shape_source = torch.empty((BOOLEAN_ROWS, BOOLEAN_COLUMNS), dtype=torch.float32)
    return InputBundle((shape_source,), 'intent.examples.kernels.reduction.boolean:row_boolean_reduction', 'row_boolean_reduction', Tolerance(atol=0.0), None, ())

def inputs_kmeans_assign():
    CLUSTERS = 64
    FEATURES = 64
    POINTS = 65536
    points = torch.randn((POINTS, FEATURES), dtype=torch.float16) * 0.1
    centroids = torch.randn((CLUSTERS, FEATURES), dtype=torch.float16) * 0.1
    return InputBundle((points, centroids), 'intent.examples.kernels.clustering.kmeans:kmeans_assign', 'kmeans_assign', (Tolerance(atol=0.0), Tolerance(atol=0.0002)), None, ())

def inputs_jagged_mean():
    lengths = torch.arange(512, dtype=torch.int32) % 128 + 1
    offsets = torch.empty(513, dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(lengths, dim=0)
    values = torch.randn((33024, 128), dtype=torch.float32)
    return InputBundle((values, offsets), 'intent.examples.kernels.ragged.jagged_mean:jagged_mean', 'jagged_mean', Tolerance(atol=2e-05, rtol=1e-05), None, ())

def inputs_ragged_grouped_gemm():
    GROUPED_K = 4096
    GROUPED_N = 4096
    GROUPED_ROWS = 8192
    GROUPS = 8
    x = torch.randn((GROUPED_ROWS, GROUPED_K), dtype=torch.float16)
    x /= math.sqrt(GROUPED_K)
    offsets = torch.tensor([0, GROUPED_ROWS // 32, GROUPED_ROWS // 16, GROUPED_ROWS // 8, GROUPED_ROWS // 4, GROUPED_ROWS // 2, 3 * GROUPED_ROWS // 4, 7 * GROUPED_ROWS // 8, GROUPED_ROWS], dtype=torch.int32)
    weight = torch.randn((GROUPS, GROUPED_K, GROUPED_N), dtype=torch.float16)
    return InputBundle((x, offsets, weight), 'intent.examples.kernels.ragged.grouped_gemm:ragged_grouped_gemm', 'ragged_grouped_gemm', Tolerance(atol=0.05), None, ())

def inputs_ragged_grouped_gemm_bf16():
    rows = (256, 512, 1024, 2048)
    x = torch.cat(tuple((torch.randn((count, 4096), dtype=torch.bfloat16) for count in rows)))
    weight = torch.stack(tuple((torch.randn((4096, 4096), dtype=torch.bfloat16) for _ in rows)))
    offsets = torch.tensor((0, 256, 768, 1792, 3840), dtype=torch.int32)
    return InputBundle((x, offsets, weight), 'intent.examples.kernels.ragged.grouped_gemm:ragged_grouped_gemm_bf16', 'ragged_grouped_gemm', Tolerance(atol=0.05, rtol=0.02), None, ())

def inputs_moe_expert_projection():
    (tokens, hidden, output, experts) = (2048, 4096, 14336, 8)
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16)
    weight = torch.randn((experts, output, hidden), dtype=torch.bfloat16)
    ids = torch.stack((torch.arange(tokens) % experts, (torch.arange(tokens) + 1) % experts), dim=1).to(torch.int32)
    member_routes = torch.argsort(ids.reshape(-1), stable=True).to(torch.int32)
    counts = torch.bincount(ids.reshape(-1), minlength=experts)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))).to(torch.int32)
    return InputBundle((x, offsets, member_routes, weight), 'intent.examples.kernels.ragged.grouped_gemm:routed_expert_projection_bf16', 'routed_expert_projection_bf16', Tolerance(atol=0.1, rtol=0.05), {'TOP_K': 2}, ())

def inputs_nested_jagged_mean_pool():
    DOCUMENTS = 256
    NESTED_FEATURES = 128
    NESTED_TOKENS = 65536
    SENTENCES = 4096

    def _nested_pool_inputs():
        document_lengths = torch.tensor([8 if index % 2 == 0 else 24 for index in range(DOCUMENTS)], dtype=torch.int32)
        sentence_lengths = torch.tensor([8 if index % 2 == 0 else 24 for index in range(SENTENCES)], dtype=torch.int32)
        document_offsets = torch.zeros(DOCUMENTS + 1, dtype=torch.int32)
        sentence_offsets = torch.zeros(SENTENCES + 1, dtype=torch.int32)
        document_offsets[1:] = document_lengths.cumsum(dim=0)
        sentence_offsets[1:] = sentence_lengths.cumsum(dim=0)
        values = torch.randn((NESTED_TOKENS, NESTED_FEATURES), dtype=torch.float32)
        return (document_offsets, sentence_offsets, values)
    (document_offsets, sentence_offsets, values) = _nested_pool_inputs()
    return InputBundle((document_offsets, sentence_offsets, values), 'intent.examples.kernels.ragged.nested_pool:nested_jagged_mean_pool', 'nested_jagged_mean_pool', (Tolerance(atol=2e-05), Tolerance(atol=2e-05)), None, ())

def inputs_nested_jagged_mean_pool_identity():
    DOCUMENTS = 256
    NESTED_FEATURES = 128
    NESTED_TOKENS = 65536
    SENTENCES = 4096

    def _nested_pool_inputs():
        document_lengths = torch.tensor([8 if index % 2 == 0 else 24 for index in range(DOCUMENTS)], dtype=torch.int32)
        sentence_lengths = torch.tensor([8 if index % 2 == 0 else 24 for index in range(SENTENCES)], dtype=torch.int32)
        document_offsets = torch.zeros(DOCUMENTS + 1, dtype=torch.int32)
        sentence_offsets = torch.zeros(SENTENCES + 1, dtype=torch.int32)
        document_offsets[1:] = document_lengths.cumsum(dim=0)
        sentence_offsets[1:] = sentence_lengths.cumsum(dim=0)
        values = torch.randn((NESTED_TOKENS, NESTED_FEATURES), dtype=torch.float32)
        return (document_offsets, sentence_offsets, values)
    (document_offsets, sentence_offsets, values) = _nested_pool_inputs()
    document_indices = torch.arange(SENTENCES, dtype=torch.int32)
    token_indices = torch.arange(NESTED_TOKENS, dtype=torch.int32)
    return InputBundle((document_offsets, sentence_offsets, document_indices, token_indices, values), 'intent.examples.kernels.variants.decomposition:nested_jagged_mean_pool_identity', 'nested_jagged_mean_pool_indexed', (Tolerance(atol=2e-05), Tolerance(atol=2e-05)), None, ())

def inputs_selective_state_scan():
    SELECTIVE_SCAN_BATCH = 128
    SELECTIVE_SCAN_LENGTH = 4096
    x = torch.randn((SELECTIVE_SCAN_BATCH, SELECTIVE_SCAN_LENGTH), dtype=torch.float32) * 0.05
    decay = 0.9 + 0.09 * torch.rand_like(x)
    drive = torch.randn_like(x) * 0.05
    return InputBundle((x, decay, drive), 'intent.examples.kernels.streaming.selective_scan:selective_state_scan', 'selective_state_scan', Tolerance(atol=2e-05), None, ())

def inputs_streamed_online_softmax_f16():
    x = torch.randn((8192, 8192), dtype=torch.float16)
    return InputBundle((x,), 'intent.examples.kernels.streaming.online_softmax:streamed_online_softmax_f16', 'streamed_online_softmax_f16', Tolerance(atol=0.01, rtol=0.01), None, ())

def inputs_streamed_online_softmax():
    x = torch.randn((8192, 8192), dtype=torch.float32)
    return InputBundle((x,), 'intent.examples.kernels.streaming.online_softmax:streamed_online_softmax', 'softmax', Tolerance(atol=1e-05), None, ())

def inputs_recurrent_gated_delta_fwd():
    (batch, sequence, heads, key_dimension, value_dimension) = (2, 2048, 8, 128, 128)
    query = torch.randn((batch, sequence, heads, key_dimension), dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn((batch, sequence, heads, value_dimension), dtype=torch.bfloat16) * 0.1
    gate = -torch.rand((batch, sequence, heads), dtype=torch.bfloat16) * 0.5
    beta = torch.sigmoid(torch.randn_like(gate))
    scale = 1.0 / math.sqrt(key_dimension)
    return InputBundle((query, key, value, gate, beta, scale), 'intent.examples.kernels.streaming.gated_delta:recurrent_gated_delta_fwd', 'recurrent_gated_delta_fwd', (Tolerance(atol=0.1, rtol=0.05), Tolerance(atol=0.1, rtol=0.05)), {'HEAD_GROUP': 1}, ())

def inputs_mamba_chunk_scan():
    (batch, sequence, heads, groups, dimension, state, chunk) = (1, 2048, 32, 8, 64, 128, 256)
    chunks = sequence // chunk
    cb = torch.randn((batch, chunks, groups, chunk, chunk), dtype=torch.float16) * 0.05
    x = torch.randn((batch, sequence, heads, dimension), dtype=torch.float16) * 0.1
    dt = torch.rand((batch, heads, chunks, chunk), dtype=torch.float16) * 0.1
    dA = -torch.rand_like(dt) * 0.1
    state_matrix = torch.randn((batch, sequence, groups, state), dtype=torch.float16) * 0.05
    previous = torch.randn((batch, chunks, heads, dimension, state), dtype=torch.float16) * 0.05
    residual_scale = torch.randn((heads,), dtype=torch.float16) * 0.1
    return InputBundle((cb, x, dt, dA, state_matrix, previous, residual_scale), 'intent.examples.kernels.streaming.selective_scan:mamba_chunk_scan_fwd', 'mamba_chunk_scan_fwd', Tolerance(atol=0.1, rtol=0.05), {'HEADS_PER_GROUP': heads // groups}, ())

def inputs_mamba_chunk_state():
    (batch, sequence, heads, groups, dimension, state, chunk) = (1, 2048, 32, 8, 64, 128, 256)
    chunks = sequence // chunk
    state_basis = torch.randn((batch, sequence, groups, state), dtype=torch.float16)
    x = torch.randn((batch, sequence, heads, dimension), dtype=torch.float16)
    dt = torch.randn((batch, heads, chunks, chunk), dtype=torch.float16)
    cumulative_decay = torch.cumsum(-torch.rand_like(dt) * 0.1, dim=-1)
    return InputBundle((state_basis, x, dt, cumulative_decay), 'intent.examples.kernels.streaming.mamba:mamba_chunk_state_fwd', 'mamba_chunk_state_fwd', Tolerance(atol=0.1, rtol=0.05), {'HEAD_GROUP': heads // groups}, ())

def inputs_linear_attention_forward():
    (batch, sequence, heads, dimension) = (1, 2048, 16, 128)
    q = torch.randn((batch, sequence, heads, dimension), dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(dimension)
    return InputBundle((q, k, v, scale), 'intent.examples.kernels.streaming.linear_attention:fused_chunk_linear_attention_fwd', 'linear_attention_forward', (Tolerance(atol=0.1, rtol=0.05), Tolerance(atol=0.1, rtol=0.05)), None, ())

def inputs_mamba_chunk_state_bf16():
    x = torch.randn((1, 2048, 32, 64), dtype=torch.bfloat16)
    basis = torch.randn((1, 2048, 8, 128), dtype=torch.bfloat16)
    dt = torch.rand((1, 32, 8, 256), dtype=torch.float32) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    return InputBundle((basis, x, dt, decay), 'intent.examples.kernels.streaming.mamba:mamba_chunk_state_bf16_fwd', 'mamba_chunk_state_bf16_fwd', Tolerance(atol=0.02, rtol=0.02), {'HEAD_GROUP': 4}, ())

def inputs_mamba_state_passing():
    states = torch.randn((1, 8, 32, 8192), dtype=torch.float32) * 0.01
    decay = -torch.rand((1, 32, 8), dtype=torch.float32) * 0.1
    initial = torch.zeros((1, 32, 8192), dtype=torch.float32)
    return InputBundle((states, decay, initial), 'intent.examples.kernels.streaming.mamba:mamba_state_passing_fwd', 'mamba_state_passing_fwd', (Tolerance(atol=1e-05, rtol=1e-05), Tolerance(atol=1e-05, rtol=1e-05)), None, ())

def inputs_mamba_chunk_scan_bf16():
    x = torch.randn((1, 2048, 32, 64), dtype=torch.bfloat16)
    state_matrix = torch.randn((1, 2048, 8, 128), dtype=torch.bfloat16)
    cb = torch.randn((1, 8, 8, 256, 256), dtype=torch.bfloat16) * 0.01
    dt = torch.rand((1, 32, 8, 256), dtype=torch.float32) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    previous = torch.randn((1, 8, 32, 64, 128), dtype=torch.float32) * 0.01
    residual = torch.randn((32,), dtype=torch.float32) * 0.01
    return InputBundle((cb, x, dt, decay, state_matrix, previous, residual), 'intent.examples.kernels.streaming.selective_scan:mamba_chunk_scan_bf16_fwd', 'mamba_chunk_scan_fwd', Tolerance(atol=0.1, rtol=0.05), {'HEADS_PER_GROUP': 4}, ())

def inputs_mamba3_siso_step():
    query = torch.randn((32, 4, 32), dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn((32, 16, 64), dtype=torch.bfloat16) * 0.1
    adt = -torch.rand((32, 16), dtype=torch.float32) * 0.1
    (dt, trap) = (torch.rand_like(adt) * 0.1, torch.rand_like(adt) * 0.1)
    query_bias = torch.randn((16, 32), dtype=torch.bfloat16) * 0.01
    key_bias = torch.randn_like(query_bias) * 0.01
    angles = torch.randn((32, 16, 16), dtype=torch.float32) * 0.01
    residual = torch.randn((16,), dtype=torch.float32) * 0.01
    gate = torch.randn_like(value) * 0.1
    states = tuple((torch.zeros(shape, dtype=torch.float32) for shape in ((32, 16, 16), (32, 16, 64, 32), (32, 16, 32), (32, 16, 64))))
    return InputBundle((query, key, value, adt, dt, trap, query_bias, key_bias, angles, residual, gate, *states), 'intent.examples.kernels.streaming.mamba:mamba3_siso_step', 'bench_mamba3_step', (Tolerance(0.002, 0.02), Tolerance(1e-05, 1e-05), Tolerance(0.0001, 0.0001), Tolerance(0.002, 0.02)), {'HEAD_GROUP': 4}, ())

def inputs_mamba3_siso_forward():
    query = torch.randn((1, 2048, 4, 32), dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn((1, 2048, 16, 64), dtype=torch.bfloat16) * 0.1
    adt = -torch.rand((1, 16, 2048), dtype=torch.float32) * 0.1
    (dt, trap) = (torch.rand_like(adt) * 0.1, torch.rand_like(adt) * 0.1)
    query_bias = torch.randn((16, 32), dtype=torch.bfloat16) * 0.01
    key_bias = torch.randn_like(query_bias) * 0.01
    angles = torch.randn((1, 2048, 16, 16), dtype=torch.float32) * 0.01
    residual = torch.randn((16,), dtype=torch.float32) * 0.01
    gate = torch.randn_like(value) * 0.1
    stores = (torch.empty((1, 2048, 16, 32), dtype=torch.bfloat16), torch.empty((1, 2048, 16, 32), dtype=torch.bfloat16), *(torch.empty((1, 16, 2048), dtype=torch.float32) for _ in range(3)))
    return InputBundle((query, key, value, adt, dt, trap, query_bias, key_bias, angles, residual, gate, *stores), 'intent.examples.kernels.streaming.mamba:mamba3_siso_forward', 'bench_mamba3_forward', Tolerance(0.002, 0.02), {'HEAD_GROUP': 4}, (11, 12, 13, 14, 15))

def inputs_linear_attention_backward():
    q = torch.randn((1, 2048, 16, 128), dtype=torch.float16)
    (k, v, gradient) = (torch.randn_like(q), torch.randn_like(q), torch.randn_like(q))
    tolerance = Tolerance(atol=0.1, rtol=0.05)
    return InputBundle((q, k, v, gradient, 128 ** (-0.5)), 'intent.examples.kernels.streaming.linear_attention:fused_chunk_linear_attention_bwd', 'linear_attention_backward', (tolerance, tolerance, tolerance), None, ())

def inputs_chunk_retention():
    q = torch.randn((1, 2048, 16, 128), dtype=torch.float16)
    (k, v) = (torch.randn_like(q), torch.randn_like(q))
    return InputBundle((q, k, v, 128 ** (-0.5)), 'intent.examples.kernels.streaming.linear_attention:chunk_retention_fwd', 'bench_retention', Tolerance(0.05, 0.02), None, ())

def inputs_mhc_post():
    (tokens, hidden, streams) = (4096, 2560, 4)
    layer_output = torch.randn((tokens, hidden), dtype=torch.bfloat16)
    residual = torch.randn((tokens, streams, hidden), dtype=torch.bfloat16)
    post_mix = torch.randn((tokens, streams), dtype=torch.float32)
    residual_mix = torch.randn((tokens, streams, streams), dtype=torch.float32)
    return InputBundle((residual, layer_output, post_mix, residual_mix), 'intent.examples.kernels.routing.mhc:mhc_apply_residual', 'mhc_apply_residual', Tolerance(atol=0.05, rtol=0.02), None, ())

def inputs_flaggems_fp8_mqa_logits():
    (queries, keys, heads, dimension) = (256, 4096, 32, 128)
    q = (torch.randn((queries, heads, dimension), dtype=torch.float16) * 0.25).to(torch.float8_e4m3fn)
    kv = (torch.randn((keys, dimension), dtype=torch.float16) * 0.25).to(torch.float8_e4m3fn)
    kv_scale = 0.5 + torch.rand((keys,), dtype=torch.float32)
    head_weight = torch.rand((queries, heads), dtype=torch.float32) * 0.1
    indices = torch.arange(queries, dtype=torch.int32)
    key_start = indices % 17
    key_end = keys - indices % 19
    return InputBundle((q, kv, kv_scale, head_weight, key_start, key_end), 'intent.examples.kernels.routing.mqa_logits:fp8_mqa_logits', 'fp8_mqa_logits', Tolerance(atol=0.02, rtol=0.02), None, ())

def inputs_swiglu_forward_helper():
    SWIGLU_FEATURES = 14336
    SWIGLU_TOKENS = 4096
    shape = (SWIGLU_TOKENS, SWIGLU_FEATURES)
    gate = torch.randn(shape, dtype=torch.bfloat16) * 0.5
    up = torch.randn_like(gate) * 0.5
    return InputBundle((gate, up), 'intent.examples.kernels.variants.activation:swiglu_forward_helper', 'swiglu_float_intermediate', Tolerance(atol=0.05), None, ())

def inputs_gemm_loop_interchange():
    GEMM_K = 4096
    GEMM_M = 4096
    GEMM_N = 14336
    a = torch.randn((GEMM_M, GEMM_K), dtype=torch.float16) * 0.1
    b = torch.randn((GEMM_K, GEMM_N), dtype=torch.float16) * 0.1
    return InputBundle((a, b), 'intent.examples.kernels.variants.contraction:gemm_loop_interchange', 'matmul', Tolerance(atol=0.03), {'ACTIVATION': KernelSymbol('intent.examples.kernels.contraction.gemm:Activation', 'NONE')}, ())

def inputs_rotary_embedding_equivalent_index():
    HALF_DIMENSION = 64
    HEAD_DIMENSION = 128
    (heads, tokens) = (32, 2048)
    rows = heads * tokens
    values = torch.randn((rows, HEAD_DIMENSION), dtype=torch.float16)
    cosine = torch.randn((tokens, HALF_DIMENSION), dtype=torch.float16)
    sine = torch.randn_like(cosine)
    return InputBundle((values, cosine, sine), 'intent.examples.kernels.variants.indexing:rotary_embedding_equivalent_index', 'rotary_embedding_flat', Tolerance(atol=0.01), {'HEADS': heads}, ())

def inputs_matrix_transpose_scalar_domains():
    TRANSPOSE_COLUMNS = 8191
    TRANSPOSE_ROWS = 4093
    x = torch.randn((TRANSPOSE_ROWS, TRANSPOSE_COLUMNS), dtype=torch.float16)
    return InputBundle((x,), 'intent.examples.kernels.variants.layout:matrix_transpose_scalar_domains', 'transpose', Tolerance(atol=0.0), None, ())

def inputs_matrix_transpose_product_domains():
    TRANSPOSE_COLUMNS = 8191
    TRANSPOSE_ROWS = 4093
    x = torch.randn((TRANSPOSE_ROWS, TRANSPOSE_COLUMNS), dtype=torch.float16)
    return InputBundle((x,), 'intent.examples.kernels.variants.decomposition:matrix_transpose_product_domains', 'transpose', Tolerance(atol=0.0), None, ())

def inputs_conv2d_reduce_order():
    x = torch.randn((16, 256, 256), dtype=torch.float16) * 0.1
    weight = torch.randn((3, 3), dtype=torch.float16) * 0.1
    return InputBundle((x, weight), 'intent.examples.kernels.variants.contraction:conv2d_reduce_order', 'conv2d', Tolerance(atol=0.003), None, ())

def inputs_weighted_layer_norm_second_moment():
    x = torch.randn((8192, 4096), dtype=torch.float32)
    weight = torch.randn((4096,), dtype=torch.float32)
    bias = torch.randn_like(weight)
    return InputBundle((x, weight, bias, 1.0 / 4096, 1e-05), 'intent.examples.kernels.variants.normalization:weighted_layer_norm_second_moment', 'layer_norm', Tolerance(atol=5e-05), None, ())

def inputs_ordered_prefix_nested():
    x = torch.randn((512, 17, 31), dtype=torch.float32) * 0.1
    return InputBundle((x,), 'intent.examples.kernels.variants.decomposition:ordered_prefix_nested', 'ordered_prefix', Tolerance(atol=2e-05), None, ())

def inputs_stable_softmax_online():
    ONLINE_COLUMNS = 8192
    ONLINE_ROWS = 8192
    x = torch.randn((ONLINE_ROWS, ONLINE_COLUMNS), dtype=torch.float32)
    return InputBundle((x,), 'intent.examples.kernels.variants.normalization:stable_softmax_online', 'softmax', Tolerance(atol=1e-05), None, ())

def inputs_streamed_online_softmax_inline():
    ONLINE_COLUMNS = 8192
    ONLINE_ROWS = 8192
    x = torch.randn((ONLINE_ROWS, ONLINE_COLUMNS), dtype=torch.float32)
    return InputBundle((x,), 'intent.examples.kernels.variants.streaming:streamed_online_softmax_inline', 'softmax', Tolerance(atol=1e-05), None, ())

def inputs_flash_attention_inline():
    q = torch.randn((4, 32, 4096, 128), dtype=torch.float16)
    (k, v) = (torch.randn_like(q), torch.randn_like(q))
    return InputBundle((q, k, v, 128 ** (-0.5)), 'intent.examples.kernels.variants.streaming:flash_attention_inline_fwd', 'bench_flash_attention_causal', Tolerance(atol=0.05, rtol=0.02), {'CAUSAL': True}, ())

def inputs_flash_attention_select():
    q = torch.randn((4, 32, 4096, 128), dtype=torch.float16)
    (k, v) = (torch.randn_like(q), torch.randn_like(q))
    return InputBundle((q, k, v, 128 ** (-0.5)), 'intent.examples.kernels.variants.streaming:flash_attention_select_fwd', 'bench_flash_attention_causal', Tolerance(atol=0.05, rtol=0.02), {'CAUSAL': True}, ())

def inputs_flash_attention_full_causal_stream():
    q = torch.randn((4, 32, 4096, 128), dtype=torch.float16)
    (k, v) = (torch.randn_like(q), torch.randn_like(q))
    return InputBundle((q, k, v, 128 ** (-0.5)), 'intent.examples.kernels.variants.streaming:flash_attention_full_causal_stream_fwd', 'bench_flash_attention_causal', Tolerance(atol=0.05, rtol=0.02), {'CAUSAL': True}, ())

def inputs_paged_mla_decode():
    (batch, query_heads, key_heads) = (32, 128, 1)
    (sequence, value_dimension, rope_dimension) = (8192, 128, 64)
    page_size = 64
    pages_per_sequence = sequence // page_size
    pages = batch * pages_per_sequence
    q_latent = torch.randn((batch, query_heads, value_dimension), dtype=torch.float16)
    q_rope = torch.randn((batch, query_heads, rope_dimension), dtype=torch.float16)
    latent_cache = torch.randn((pages, page_size, key_heads, value_dimension), dtype=torch.float16)
    rope_cache = torch.randn((pages, page_size, key_heads, rope_dimension), dtype=torch.float16)
    page_indices = torch.arange(pages, dtype=torch.int32)
    page_offsets = torch.arange(0, pages + 1, pages_per_sequence, dtype=torch.int32)
    lengths = torch.full((batch,), sequence, dtype=torch.int32)
    scale = (value_dimension + rope_dimension) ** (-0.5)
    return InputBundle((q_latent, q_rope, latent_cache, rope_cache, page_offsets, page_indices, lengths, scale), 'intent.examples.kernels.streaming.mla:paged_mla_decode', 'paged_mla_decode', Tolerance(atol=0.05, rtol=0.05), {'PAGE_SIZE': page_size, 'HEAD_GROUP': query_heads // key_heads}, ())

def inputs_mla_prefill():
    query = torch.randn((1, 128, 2048, 128), dtype=torch.float16)
    query_rope = torch.randn((1, 128, 2048, 64), dtype=torch.float16)
    key = torch.randn((1, 1, 2048, 128), dtype=torch.float16)
    key_rope = torch.randn((1, 1, 2048, 64), dtype=torch.float16)
    value = torch.randn_like(key)
    return InputBundle((query, query_rope, key, key_rope, value, 192 ** (-0.5)), 'intent.examples.kernels.streaming.attention:mla_prefill', 'mla_prefill', Tolerance(atol=0.05, rtol=0.02), {'HEAD_GROUP': 128}, ())

def inputs_absorbed_mla_prefill():
    ABSORBED_MLA_SCALE = 0.07216878364870323
    query = torch.randn((1, 512, 8, 512), dtype=torch.float16) * 0.1
    query_rope = torch.randn((1, 512, 8, 64), dtype=torch.float16) * 0.1
    cache = torch.randn((1, 512, 512), dtype=torch.float16) * 0.1
    cache_rope = torch.randn((1, 512, 64), dtype=torch.float16) * 0.1
    return InputBundle((query, query_rope, cache, cache_rope, ABSORBED_MLA_SCALE), 'intent.examples.kernels.streaming.mla:absorbed_mla_prefill', 'absorbed_mla_prefill', Tolerance(atol=0.04), None, ())

def inputs_absorbed_mla_decode():

    def _absorbed_decode_inputs():
        query = torch.randn((8, 64, 512), dtype=torch.float16)
        query_rope = torch.randn((8, 64, 64), dtype=torch.float16)
        cache = torch.randn((8, 8192, 512), dtype=torch.float16)
        cache_rope = torch.randn((8, 8192, 64), dtype=torch.float16)
        return (query, query_rope, cache, cache_rope, 576 ** (-0.5))
    return InputBundle(_absorbed_decode_inputs(), 'intent.examples.kernels.streaming.mla:absorbed_mla_decode', 'absorbed_mla_decode', Tolerance(atol=0.05, rtol=0.02), None, ())

def inputs_token_sparse_mla_prefill():
    query = torch.randn((64, 8, 512), dtype=torch.bfloat16) * 0.1
    query_rope = torch.randn((64, 8, 64), dtype=torch.bfloat16) * 0.1
    cache = torch.randn((4096, 512), dtype=torch.bfloat16) * 0.1
    cache_rope = torch.randn((4096, 64), dtype=torch.bfloat16) * 0.1
    selected = torch.randint(0, 4096, (64, 256), dtype=torch.int32)
    (selected[:, -2], selected[:, -1]) = (-1, 4096)
    return InputBundle((query, query_rope, cache, cache_rope, selected, 576 ** (-0.5)), 'intent.examples.kernels.streaming.mla:token_sparse_mla_prefill', 'token_sparse_mla_prefill', (Tolerance(atol=0.05), Tolerance(atol=0.005), Tolerance(atol=0.005)), None, ())

def inputs_token_sparse_mla_value_prefill():
    query = torch.randn((1, 64, 2048, 128), dtype=torch.bfloat16)[0].permute(1, 0, 2).contiguous()
    query_rope = torch.randn((1, 64, 2048, 64), dtype=torch.bfloat16)[0].permute(1, 0, 2).contiguous()
    key = torch.randn((4096, 128), dtype=torch.bfloat16)
    value = torch.randn_like(key)
    key_rope = torch.randn((4096, 64), dtype=torch.bfloat16)
    selected = (torch.arange(2048, dtype=torch.int32)[:, None] - torch.arange(512, dtype=torch.int32)[None, :]).clamp_min(0)
    return InputBundle((query, query_rope, key, value, key_rope, selected, 192 ** (-0.5)), 'intent.examples.kernels.streaming.mla:token_sparse_mla_value_prefill', 'token_sparse_mla_value_prefill', Tolerance(atol=0.1, rtol=0.05), None, ())

def inputs_sparse_2to4_gemm():
    from .corpus.cpu.baselines.pytorch.cpu_runtime import make_sparse_2to4_inputs
    m, n, k = 8192, 14336, 8192
    compressed, metadata = make_sparse_2to4_inputs(m, k)
    rhs = torch.randn((k, n), dtype=torch.float16)
    return InputBundle((compressed, metadata, rhs), 'intent.examples.kernels.contraction.sparse_2to4:sparse_2to4_gemm',
                       'sparse_2to4_gemm', Tolerance(atol=5e-2, rtol=2e-2))


def inputs_fused_linear_cross_entropy():
    hidden = torch.randn((2048, 4096), dtype=torch.bfloat16)
    weight = torch.randn((32768, 4096), dtype=torch.bfloat16)
    target = torch.randint(0, 32768, (2048,), dtype=torch.int64)
    return InputBundle((hidden, weight, target), '', 'bench_linear_cross_entropy',
                       (Tolerance(atol=5e-2, rtol=2e-2), Tolerance(atol=5e-2, rtol=2e-2), Tolerance(atol=2e-2, rtol=2e-2)))


def inputs_paged_gqa_f16_reduction():
    q = torch.randn((16, 32, 128), dtype=torch.float16)
    key = torch.randn((8192, 16, 8, 128), dtype=torch.float16)
    value = torch.randn_like(key)
    indices = torch.arange(8192, dtype=torch.int32)
    offsets = torch.arange(0, 8193, 512, dtype=torch.int32)
    splits = torch.arange(0, 8193, 64, dtype=torch.int32)
    lengths = torch.full((16,), 8192, dtype=torch.int32)
    return InputBundle((q, key, value, offsets, indices, lengths, splits, 128 ** -0.5), '',
                       'bench_paged_gqa_f16_reduction', Tolerance(atol=1e-1, rtol=5e-2))


FACTORIES = {
    'fused_linear_cross_entropy': inputs_fused_linear_cross_entropy,
    'paged_gqa_f16_reduction': inputs_paged_gqa_f16_reduction,
    'sparse_2to4_gemm': inputs_sparse_2to4_gemm,
    'batched_row_affine': inputs_batched_row_affine,
    'integer_log2_floor': inputs_integer_log2_floor,
    'paired_sum_product': inputs_paired_sum_product,
    'weighted_rms_norm': inputs_weighted_rms_norm,
    'stable_softmax': inputs_stable_softmax,
    'weighted_layer_norm': inputs_weighted_layer_norm,
    'flaggems_logsumexp': inputs_flaggems_logsumexp,
    'fused_softmax': inputs_fused_softmax,
    'chunked_softmax': inputs_chunked_softmax,
    'layer_norm_f16': inputs_layer_norm_f16,
    'layer_norm_bf16': inputs_layer_norm_bf16,
    'rms_norm_f32': inputs_rms_norm_f32,
    'rms_norm_bf16': inputs_rms_norm_bf16,
    'flaggems_softmax_backward': inputs_flaggems_softmax_backward,
    'fused_add_rms_norm': inputs_fused_add_rms_norm,
    'dense_gemm_f32': inputs_dense_gemm_f32,
    'dense_gemm': inputs_dense_gemm,
    'tilegym_dense_gemm': inputs_tilegym_dense_gemm,
    'batched_gemm': inputs_batched_gemm,
    'batched_gemm_tn': inputs_batched_gemm_tn,
    'batched_gemm_nt': inputs_batched_gemm_nt,
    'batched_gemm_tt': inputs_batched_gemm_tt,
    'mla_head_query_projection': inputs_mla_head_query_projection,
    'mla_head_value_projection': inputs_mla_head_value_projection,
    'qkv_projection': inputs_qkv_projection,
    'gated_dual_gemm': inputs_gated_dual_gemm,
    'fp8_gemm': inputs_fp8_gemm,
    'fp8_e5m2_gemm': inputs_fp8_e5m2_gemm,
    'quantized_gemm': inputs_quantized_gemm,
    'weight_only_int4': inputs_weight_only_int4,
    'w4a8_gemm': inputs_w4a8_gemm,
    'bitnet_int2_decode': inputs_bitnet_int2_decode,
    'dequant_bf16_fp4': inputs_dequant_bf16_fp4,
    'deepgemm_fp8_2xacc': inputs_deepgemm_fp8_2xacc,
    'scaled_fp8_gemm': inputs_scaled_fp8_gemm,
    'vector_dot': inputs_vector_dot,
    'matrix_vector': inputs_matrix_vector,
    'vector_matrix': inputs_vector_matrix,
    'vector_outer': inputs_vector_outer,
    'mxfp8_gemm': inputs_mxfp8_gemm,
    'block_sparse_gemm': inputs_block_sparse_gemm,
    'causal_attention_f32': inputs_causal_attention_f32,
    'causal_linear_attention_f32': inputs_causal_linear_attention_f32,
    'flash_attention_bf16': inputs_flash_attention_bf16,
    'flash_attention_f16': inputs_flash_attention_f16,
    'flash_gqa_attention_f16': inputs_flash_gqa_attention_f16,
    'continuous_gqa_decode': inputs_continuous_gqa_decode,
    'flash_varlen_gqa_prefill': inputs_flash_varlen_gqa_prefill,
    'flash_attention_bias': inputs_flash_attention_bias,
    'flash_varlen_attention': inputs_flash_varlen_attention,
    'attention_sink_prefill': inputs_attention_sink_prefill,
    'gemma_gqa_prefill': inputs_gemma_gqa_prefill,
    'sliding_window_gqa_prefill': inputs_sliding_window_gqa_prefill,
    'block_causal_attention': inputs_block_causal_attention,
    'varlen_block_causal_attention': inputs_varlen_block_causal_attention,
    'native_sparse_attention': inputs_native_sparse_attention,
    'varlen_gqa_decode_with_sink_logits': inputs_varlen_gqa_decode_with_sink_logits,
    'paged_gqa_decode_attention': inputs_paged_gqa_decode_attention,
    'gelu': inputs_gelu,
    'relu': inputs_relu,
    'flaggems_addcmul': inputs_flaggems_addcmul,
    'swiglu': inputs_swiglu,
    'swiglu_backward': inputs_swiglu_backward,
    'silu_and_mul_packed': inputs_silu_and_mul_packed,
    'flaggems_transpose_copy': inputs_flaggems_transpose_copy,
    'embedding_lookup': inputs_embedding_lookup,
    'embedding_lookup_f32': inputs_embedding_lookup_f32,
    'index_select': inputs_index_select,
    'scalar_table_lookup': inputs_scalar_table_lookup,
    'shifted_row_copy': inputs_shifted_row_copy,
    'roll_rows_forward': inputs_roll_rows_forward,
    'grouped_query_head_add': inputs_grouped_query_head_add,
    'alternating_signed_indices': inputs_alternating_signed_indices,
    'csr_spmv': inputs_csr_spmv,
    'csr_spmm': inputs_csr_spmm,
    'roi_align_center_sample': inputs_roi_align_center_sample,
    'max_pool2d': inputs_max_pool2d,
    'flaggems_max_pool2d_with_indices': inputs_flaggems_max_pool2d_with_indices,
    'greedy_nms': inputs_greedy_nms,
    'rotary_embedding_flat': inputs_rotary_embedding_flat,
    'rotary_embedding_bf16': inputs_rotary_embedding_bf16,
    'flaggems_conv1d': inputs_flaggems_conv1d,
    'causal_conv1d': inputs_causal_conv1d,
    'causal_conv1d_bf16': inputs_causal_conv1d_bf16,
    'conv2d': inputs_conv2d,
    'conv2d_nhwc': inputs_conv2d_nhwc,
    'bitonic_sort': inputs_bitonic_sort,
    'insertion_top_k': inputs_insertion_top_k,
    'smith_waterman': inputs_smith_waterman,
    'viterbi': inputs_viterbi,
    'radix2_fft': inputs_radix2_fft,
    'flaggems_cumsum': inputs_flaggems_cumsum,
    'ordered_prefix': inputs_ordered_prefix,
    'compact_nonzero': inputs_compact_nonzero,
    'sorted_nucleus_cutoff': inputs_sorted_nucleus_cutoff,
    'histogram': inputs_histogram,
    'flash_cross_entropy_bf16': inputs_flash_cross_entropy_bf16,
    'dropout': inputs_dropout,
    'barrier_option_paths': inputs_barrier_option_paths,
    'boolean_reduction': inputs_boolean_reduction,
    'kmeans_assign': inputs_kmeans_assign,
    'jagged_mean': inputs_jagged_mean,
    'ragged_grouped_gemm': inputs_ragged_grouped_gemm,
    'ragged_grouped_gemm_bf16': inputs_ragged_grouped_gemm_bf16,
    'moe_expert_projection': inputs_moe_expert_projection,
    'nested_jagged_mean_pool': inputs_nested_jagged_mean_pool,
    'nested_jagged_mean_pool_identity': inputs_nested_jagged_mean_pool_identity,
    'selective_state_scan': inputs_selective_state_scan,
    'streamed_online_softmax_f16': inputs_streamed_online_softmax_f16,
    'streamed_online_softmax': inputs_streamed_online_softmax,
    'recurrent_gated_delta_fwd': inputs_recurrent_gated_delta_fwd,
    'mamba_chunk_scan': inputs_mamba_chunk_scan,
    'mamba_chunk_state': inputs_mamba_chunk_state,
    'linear_attention_forward': inputs_linear_attention_forward,
    'mamba_chunk_state_bf16': inputs_mamba_chunk_state_bf16,
    'mamba_state_passing': inputs_mamba_state_passing,
    'mamba_chunk_scan_bf16': inputs_mamba_chunk_scan_bf16,
    'mamba3_siso_step': inputs_mamba3_siso_step,
    'mamba3_siso_forward': inputs_mamba3_siso_forward,
    'linear_attention_backward': inputs_linear_attention_backward,
    'chunk_retention': inputs_chunk_retention,
    'mhc_post': inputs_mhc_post,
    'flaggems_fp8_mqa_logits': inputs_flaggems_fp8_mqa_logits,
    'swiglu_forward_helper': inputs_swiglu_forward_helper,
    'gemm_loop_interchange': inputs_gemm_loop_interchange,
    'rotary_embedding_equivalent_index': inputs_rotary_embedding_equivalent_index,
    'matrix_transpose_scalar_domains': inputs_matrix_transpose_scalar_domains,
    'matrix_transpose_product_domains': inputs_matrix_transpose_product_domains,
    'conv2d_reduce_order': inputs_conv2d_reduce_order,
    'weighted_layer_norm_second_moment': inputs_weighted_layer_norm_second_moment,
    'ordered_prefix_nested': inputs_ordered_prefix_nested,
    'stable_softmax_online': inputs_stable_softmax_online,
    'streamed_online_softmax_inline': inputs_streamed_online_softmax_inline,
    'flash_attention_inline': inputs_flash_attention_inline,
    'flash_attention_select': inputs_flash_attention_select,
    'flash_attention_full_causal_stream': inputs_flash_attention_full_causal_stream,
    'paged_mla_decode': inputs_paged_mla_decode,
    'mla_prefill': inputs_mla_prefill,
    'absorbed_mla_prefill': inputs_absorbed_mla_prefill,
    'absorbed_mla_decode': inputs_absorbed_mla_decode,
    'token_sparse_mla_prefill': inputs_token_sparse_mla_prefill,
    'token_sparse_mla_value_prefill': inputs_token_sparse_mla_value_prefill,
}
