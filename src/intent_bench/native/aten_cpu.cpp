#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {
struct Call {
  int operation;
  std::vector<at::Tensor> inputs, outputs, workspace;
  std::vector<double> scalars;
  std::string error;
};

at::ScalarType dtype(int kind) {
  switch (kind) {
  case 0: return at::kFloat;
  case 1: return at::kHalf;
  case 2: return at::kBFloat16;
  case 3: return at::kLong;
  case 4: return at::kInt;
  case 5: return at::kChar;
  default: TORCH_CHECK(false, "unsupported baseline input dtype");
  }
}

void initialize(Call &call) {
  auto &a = call.inputs;
  auto &o = call.outputs;
  auto &w = call.workspace;
  switch (call.operation) {
  case 0: case 1: case 4: case 7: case 8: case 10:
    o.push_back(at::empty_like(a[0])); break;
  case 2: o.push_back(at::empty({a[0].size(1), a[0].size(0)}, a[0].options())); break;
  case 3: o.push_back(at::empty({a[1].numel(), a[0].size(1)}, a[0].options())); break;
  case 5: o.push_back(at::empty({a[0].size(0), a[1].size(1)}, a[0].options())); break;
  case 6: o.push_back(at::empty({a[0].size(0), a[0].size(1), a[1].size(2)}, a[0].options())); break;
  case 9:
    o = {a[1], a[2], a[3]};
    w.push_back(at::ones({}, a[0].options()));
    break;
  case 11:
    o = {a[0]};
    w.push_back(at::empty({a[0].size(0)}, a[0].options().dtype(at::kInt)));
    break;
  case 12:
    o = {at::empty_like(a[0]), at::empty_like(a[2]), at::empty_like(a[2])};
    break;
  case 13:
    o.push_back(at::empty({a[0].numel()-1}, a[2].options()));
    w.push_back(at::sparse_csr_tensor(a[0], a[1], a[2],
                                    {a[0].numel()-1, a[3].numel()},
                                    a[2].options().layout(at::kSparseCsr)));
    break;
  case 14: {
    TORCH_CHECK(a[1].numel() == a[2].size(0)+1, "group offsets must match grouped weights");
    TORCH_CHECK(a[1].scalar_type() == at::kInt || a[1].scalar_type() == at::kLong,
                "group offsets require i32 or i64 storage");
    o.push_back(at::empty({a[0].size(0), a[2].size(2)}, a[0].options()));
    break;
  }
  case 15:
    o.push_back(at::empty({a[0].size(0), a[1].size(1)}, a[0].options().dtype(at::kInt)));
    break;
  case 16:
    TORCH_CHECK(a[0].dim()==4 && a[1].dim()==4 && a[2].sizes()==a[1].sizes(),
                "CPU flash attention requires [B,H,T,D] Q/K/V");
    TORCH_CHECK(a[0].size(0)==a[1].size(0) && a[0].size(3)==a[1].size(3) &&
                a[1].size(1)>0 && a[0].size(1)%a[1].size(1)==0,
                "CPU flash attention requires compatible grouped query heads");
    o.push_back(at::empty_like(a[0]));
    break;
  default: TORCH_CHECK(false, "unsupported ATen CPU operation");
  }
  if (call.operation == 4) {
    w.push_back(at::empty({a[0].size(0), 1}, a[0].options().dtype(at::kFloat)));
    w.push_back(at::empty_like(w[0]));
  }
  if (call.operation == 7) {
    w.push_back(at::empty(a[0].sizes(), a[0].options().dtype(at::kFloat)));
    w.push_back(at::empty_like(w[0]));
  }
  if (call.operation == 10) {
    o.push_back(at::empty({a[0].size(1)}, a[0].options().dtype(at::kFloat)));
    o.push_back(at::empty_like(o[1]));
  }
}

void execute(Call &call) {
  auto &a = call.inputs;
  auto &o = call.outputs;
  auto &w = call.workspace;
  auto &s = call.scalars;
  switch (call.operation) {
  case 0: at::clamp_min_out(o[0], a[0], 0); break;
  case 1: at::_softmax_out(o[0], a[0], -1, false); break;
  case 2: o[0].copy_(a[0].t()); break;
  case 3: at::index_select_out(o[0], a[0], 0, a[1]); break;
  case 4: at::native_layer_norm_out(o[0], w[0], w[1], a[0], {a[0].size(-1)}, a[1], a[2], s[1]); break;
  case 5: at::mm_out(o[0], a[0], a[1]); break;
  case 6: at::bmm_out(o[0], a[0], a[1]); break;
  case 7:
    w[0].copy_(a[0]); w[1].copy_(a[1]);
    at::silu_out(w[0], w[0]);
    w[0].mul_(w[1]); o[0].copy_(w[0]);
    break;
  case 8: at::cumsum_out(o[0], a[0], -1); break;
  case 9:
    at::_fused_adamw_({a[1]}, {a[0]}, {a[2]}, {a[3]}, {}, {w[0]},
                      s[0], s[1], s[2], s[6], s[5], false, false);
    break;
  case 10: at::native_batch_norm_out(o[0], o[1], o[2], a[0], a[1], a[2], a[3], a[4], true, s[1], s[0]); break;
  case 11: at::linalg_cholesky_ex_out(o[0], w[0], a[0], false, false); break;
  case 12: at::native_group_norm_backward_out(o[0], o[1], o[2], a[1], a[0], a[3], a[4], a[2],
                                             a[0].size(0), a[0].size(1), a[0].size(2), a[3].size(1),
                                             {true, true, true}); break;
  case 13: at::mv_out(o[0], w[0], a[3]); break;
  case 14: {
    auto boundary = [&](int64_t group) -> int64_t {
      return a[1].scalar_type() == at::kLong ? a[1].const_data_ptr<int64_t>()[group]
                                           : a[1].const_data_ptr<int32_t>()[group];
    };
    TORCH_CHECK(boundary(0) == 0 && boundary(a[2].size(0)) == a[0].size(0),
                "group offsets must cover the complete input");
    for (int64_t group=0; group<a[2].size(0); ++group) {
      int64_t begin = boundary(group), end = boundary(group+1);
      TORCH_CHECK(0 <= begin && begin <= end && end <= a[0].size(0), "invalid group offsets");
      auto output = o[0].slice(0, begin, end);
      at::mm_out(output, a[0].slice(0, begin, end), a[2][group]);
    }
    break;
  }
  case 15: at::_int_mm_out(o[0], a[0], a[1]); o[0].add_(a[2]); break;
  case 16:
    // The public CPU flash-attention API owns output/LSE allocation. Keep its
    // complete native call, including packing and allocation, inside the timer.
    o[0] = std::get<0>(at::_scaled_dot_product_flash_attention_for_cpu(
        a[0], a[1], a[2], 0.0, true, std::nullopt, s[0]));
    break;
  }
}
}

extern "C" void *bench_aten_create(int operation, int workers, int count, void **data,
                                    const int *types, const int64_t *ranks,
                                    const int64_t *sizes, const int64_t *strides,
                                    int scalarCount, const double *scalars) {
  auto call = std::make_unique<Call>();
  call->operation = operation;
  try {
    at::set_num_threads(workers);
    call->scalars.assign(scalars, scalars + scalarCount);
    for (int i = 0; i < count; ++i) {
      call->inputs.push_back(at::from_blob(data[i], at::IntArrayRef(sizes, ranks[i]),
          at::IntArrayRef(strides, ranks[i]), at::TensorOptions().device(at::kCPU).dtype(dtype(types[i]))));
      sizes += ranks[i]; strides += ranks[i];
    }
    initialize(*call);
  } catch (const std::exception &error) { call->error = error.what(); }
  return call.release();
}

extern "C" void bench_aten_run(void *opaque) {
  auto &call = *static_cast<Call *>(opaque);
  try { execute(call); }
  catch (const std::exception &error) { call.error = error.what(); }
}
extern "C" const char *bench_aten_error(void *opaque) { return static_cast<Call *>(opaque)->error.c_str(); }
extern "C" int bench_aten_outputs(void *opaque) { return static_cast<Call *>(opaque)->outputs.size(); }
extern "C" void *bench_aten_output(void *opaque, int output, int64_t *rank, int64_t *sizes, int *type) {
  const auto &tensor = static_cast<Call *>(opaque)->outputs.at(output);
  *rank = tensor.dim();
  for (int64_t i = 0; i < *rank; ++i) sizes[i] = tensor.size(i);
  for (int kind = 0; kind <= 5; ++kind)
    if (tensor.scalar_type() == dtype(kind)) *type = kind;
  return tensor.data_ptr();
}
extern "C" void bench_aten_destroy(void *opaque) { delete static_cast<Call *>(opaque); }
