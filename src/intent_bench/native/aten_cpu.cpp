#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#ifdef INTENT_BENCH_AOTI
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#endif
#include <algorithm>
#include <cstdint>
#include <memory>
#include <limits>
#include <string>
#include <vector>

#ifdef INTENT_BENCH_ONEDNN
extern "C" void *bench_dnnl_matmul_create(int64_t, int64_t, int64_t, int64_t, void *, void *, void *);
extern "C" const char *bench_dnnl_matmul_error(void *);
extern "C" const char *bench_dnnl_matmul_implementation(void *);
extern "C" void bench_dnnl_matmul_run(void *);
extern "C" void bench_dnnl_matmul_destroy(void *);
#endif
#ifdef INTENT_BENCH_VLLM
extern "C" int bench_vllm_moe_align(int32_t *, int32_t *, int32_t *, int32_t *, int32_t *, int32_t *, int, int, int);
#endif

namespace {
struct Call {
  int operation;
  std::vector<at::Tensor> inputs, outputs, workspace;
  std::vector<double> scalars;
  std::string error;
#ifdef INTENT_BENCH_AOTI
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> package;
#endif
#ifdef INTENT_BENCH_ONEDNN
  void *matrix = nullptr;
  ~Call() { bench_dnnl_matmul_destroy(matrix); }
#endif
};

at::ScalarType dtype(int kind) {
  switch (kind) {
  case 0: return at::kFloat;
  case 1: return at::kHalf;
  case 2: return at::kBFloat16;
  case 3: return at::kLong;
  case 4: return at::kInt;
  case 5: return at::kChar;
  case 6: return at::kByte;
  case 7: return at::kFloat8_e4m3fn;
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
  case 6: case 19: case 20: case 21: {
    bool leftTranspose = call.operation == 19 || call.operation == 21;
    bool rightTranspose = call.operation == 20 || call.operation == 21;
    o.push_back(at::empty({a[0].size(0), a[0].size(leftTranspose ? 2 : 1),
                          a[1].size(rightTranspose ? 1 : 2)}, a[0].options()));
    break;
  }
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
  case 17:
    TORCH_CHECK(a[0].dim()==1 && a[0].scalar_type()==at::kFloat,
                "the author histogram requires a one-dimensional FP32 input");
    o.push_back(at::empty({256}, a[0].options().dtype(at::kInt)));
    w.push_back(at::empty(a[0].sizes(), a[0].options().dtype(at::kInt)));
    w.push_back(at::empty(a[0].sizes(), a[0].options().dtype(at::kBool)));
    w.push_back(at::empty_like(w[1]));
    break;
  case 18:
    TORCH_CHECK(a[0].dim()==2 && a[1].dim()==1 && a[1].numel()>0,
                "jagged mean requires values [T,D] and segment offsets [B+1]");
    o.push_back(at::empty({a[1].numel()-1, a[0].size(1)}, a[0].options()));
    break;
  case 22:
    TORCH_CHECK(a[0].dim()==3 && a[1].dim()==2 && a[2].dim()==1 &&
                a[1].size(0)==a[0].size(1) && a[2].size(0)==a[0].size(1),
                "causal convolution requires X[B,D,L], W[D,W] and bias[D]");
    o.push_back(at::empty_like(a[0]));
    for (const auto &input : a)
      w.push_back(at::empty(input.sizes(), input.options().dtype(at::kFloat)));
    break;
  case 23:
    TORCH_CHECK(a[0].dim()==2, "row compaction requires a matrix");
    o.push_back(at::empty(a[0].sizes(), a[0].options().dtype(at::kInt)));
    o.push_back(at::empty({a[0].size(0)}, a[0].options().dtype(at::kInt)));
    break;
  case 24:
    TORCH_CHECK(a.size()==4 && a[0].dim()==3 && a[2].dim()==3 &&
                a[0].scalar_type()==at::kFloat8_e4m3fn && a[2].scalar_type()==at::kFloat8_e4m3fn &&
                a[1].scalar_type()==at::kByte && a[3].scalar_type()==at::kByte &&
                a[0].size(1)==a[2].size(0) && a[0].size(2)==a[2].size(1) &&
                a[1].sizes()==at::IntArrayRef({a[0].size(0), a[0].size(1)}) &&
                a[3].sizes()==at::IntArrayRef({a[2].size(0), a[2].size(2)}),
                "block-scaled GEMM requires E4M3 A[M,G,K], B[G,K,N] and E8M0 byte scales[M,G]/[G,N]");
    o.push_back(at::empty({a[0].size(0), a[2].size(2)}, a[0].options().dtype(at::kFloat)));
    for (const auto &input : a)
      w.push_back(at::empty(input.sizes(), input.options().dtype(at::kFloat)));
    w.push_back(at::empty(a[1].sizes(), a[1].options().dtype(at::kBool)));
    w.push_back(at::empty(a[3].sizes(), a[3].options().dtype(at::kBool)));
    break;
  case 25:
    TORCH_CHECK(a.size()==7 && a[0].dim()==3 && a[1].dim()==4 && a[2].dim()==4 &&
                a[0].size(2)==a[1].size(3) && a[1].sizes().slice(0, 3)==a[2].sizes().slice(0, 3) &&
                a[1].size(1)>0 && a[1].size(2)>0 && a[0].size(1)%a[1].size(2)==0,
                "paged attention requires compatible Q[B,HQ,D] and KV[P,PS,HK,D]");
    TORCH_CHECK(a[3].dim()==1 && a[3].numel()==a[0].size(0)+1 && a[4].dim()==1 &&
                a[5].dim()==1 && a[5].numel()==a[0].size(0) && a[6].dim()==1 &&
                a[3].scalar_type()==at::kInt && a[4].scalar_type()==at::kInt &&
                a[5].scalar_type()==at::kInt && a[6].scalar_type()==at::kInt,
                "paged attention requires INT32 page offsets, indices, lengths and split offsets");
    o.push_back(at::empty({a[0].size(0), a[0].size(1), a[2].size(3)}, a[0].options()));
    break;
#ifdef INTENT_BENCH_ONEDNN
  case 26: {
    TORCH_CHECK(a.size()==4 && a[0].dim()==4 && a[1].dim()==4 && a[2].dim()==4 && a[3].dim()==4 &&
                a[0].scalar_type()==at::kBFloat16 && a[1].scalar_type()==at::kBFloat16 &&
                a[2].scalar_type()==at::kFloat && a[3].scalar_type()==at::kFloat,
                "chunk state requires BF16 basis/X and FP32 dt/decay");
    auto batch=a[1].size(0), heads=a[1].size(2), groups=a[0].size(2);
    auto chunks=a[2].size(2), size=a[2].size(3), width=a[1].size(3), states=a[0].size(3);
    TORCH_CHECK(groups>0 && heads%groups==0 && size>0 && a[0].size(0)==batch &&
                a[0].size(1)==chunks*size && a[1].size(1)==chunks*size &&
                a[2].size(0)==batch && a[2].size(1)==heads && a[3].sizes()==a[2].sizes(),
                "chunk state shapes must describe complete chunks and grouped heads");
    o.push_back(at::empty({batch,chunks,heads,width,states}, a[2].options()));
    w.push_back(at::empty_like(a[2]));
    w.push_back(at::empty({batch,chunks,groups,heads/groups,size,states}, a[2].options()));
    w.push_back(at::empty(w[1].sizes(), a[0].options()));
    w.push_back(at::empty({batch,chunks,heads,width,size}, a[1].options()));
    call.matrix=bench_dnnl_matmul_create(batch*chunks*heads,width,states,size,
        w[3].data_ptr(),w[2].data_ptr(),o[0].data_ptr());
    TORCH_CHECK(call.matrix, "oneDNN matmul preparation: ", bench_dnnl_matmul_error(nullptr));
    break;
  }
#endif
  case 27:
    TORCH_CHECK(a.size() == 3 && a[0].dim() == 3 && a[1].dim() == 3 && a[2].dim() == 3 &&
                a[0].scalar_type() == at::kFloat && a[1].scalar_type() == at::kFloat &&
                a[2].scalar_type() == at::kFloat && a[0].sizes() == a[1].sizes() &&
                a[0].size(0) == a[2].size(0) && a[0].size(1) == a[2].size(1),
                "causal linear attention requires matching FP32 batch/token axes");
    o.push_back(at::empty_like(a[2]));
    o.push_back(at::empty({a[0].size(0), a[0].size(2), a[2].size(2)}, a[0].options()));
    w.push_back(at::empty({a[0].size(0), a[0].size(1), a[0].size(1)}, a[0].options()));
    break;
#ifdef INTENT_BENCH_AOTI
  case 28:
    TORCH_CHECK(a.size() == 2, "compiled CPU comparison requires input and prebound output");
    o.push_back(a[1]);
    break;
#endif
#ifdef INTENT_BENCH_VLLM
  case 29: {
    TORCH_CHECK(a.size() == 1 && a[0].dim() == 2 && a[0].scalar_type() == at::kInt &&
                call.scalars.size() == 3 && call.scalars[1] == 128,
                "vLLM CPU alignment requires INT32 routes and block size 128");
    int experts = call.scalars[0], capacity = call.scalars[2], workers = at::get_num_threads();
    TORCH_CHECK(experts > 0 && capacity >= a[0].numel() + experts * 127,
                "vLLM alignment requires worst-case padded capacity");
    o = {at::empty({capacity}, a[0].options()), at::empty({capacity / 128 + 1}, a[0].options()),
         at::empty({1}, a[0].options())};
    w = {at::empty({workers + 1, experts}, a[0].options()),
         at::empty({experts + 1}, a[0].options()), at::empty({capacity / 128 + 2}, a[0].options())};
    break;
  }
#endif
  case 30:
    TORCH_CHECK(a.size() == 6 && a[0].dim() == 4 && a[1].dim() == 4 &&
                a[1].sizes() == a[2].sizes() && a[0].sizes() == a[3].sizes() &&
                a[3].sizes() == a[4].sizes() && a[5].sizes() == a[0].sizes().slice(0, 3) &&
                a[1].size(1) > 0 && a[0].size(1) % a[1].size(1) == 0 && call.scalars.size() == 1,
                "grouped attention backward requires Q, K, V, output, grad_output, LSE and scale");
    o = {at::empty_like(a[0]), at::empty_like(a[1]), at::empty_like(a[2])};
    break;
  case 31: case 32:
    TORCH_CHECK(a.size() == 4 && a[0].dim() == 2 && a[1].dim() == 1 && a[1].numel() >= 2 &&
                a[1].scalar_type() == at::kLong && a[2].dim() == 2 && a[2].size(1) > 0 &&
                a[2].size(0) == a[0].size(1) && a[3].sizes() == a[2].sizes().slice(0, 1),
                "segmented convolution requires X[T,D], offsets[B+1], W[D,K], bias[D]");
    o = {at::empty_like(a[0]), at::empty({a[1].numel()-1, a[2].size(1), a[0].size(1)}, a[0].options())};
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
  case 6: case 19: case 20: case 21: {
    auto left = call.operation == 19 || call.operation == 21 ? a[0].transpose(-1, -2) : a[0];
    auto right = call.operation == 20 || call.operation == 21 ? a[1].transpose(-1, -2) : a[1];
    at::bmm_out(o[0], left, right);
    break;
  }
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
  case 17: {
    // Match the author's signed truncation and exclusion before counting. A
    // sentinel bin retains the complete input without data-dependent packing.
    w[0].copy_(a[0]);
    at::lt_out(w[1], w[0], 0);
    at::ge_out(w[2], w[0], 256);
    w[1].logical_or_(w[2]);
    w[0].masked_fill_(w[1], 256);
    auto counts = at::bincount(w[0], std::nullopt, 257);
    o[0].copy_(counts.narrow(0, 0, 256));
    break;
  }
  case 18:
    at::segment_reduce_out(o[0], a[0], "mean", std::nullopt, std::nullopt, a[1]);
    break;
  case 22: {
    for (size_t i=0; i<a.size(); ++i) w[i].copy_(a[i]);
    auto convolution = at::conv1d(w[0], w[1].unsqueeze(1), w[2],
        {1}, {a[1].size(1)-1}, {1}, a[0].size(1));
    auto causal = convolution.narrow(2, 0, a[0].size(2));
    at::silu_out(causal, causal);
    o[0].copy_(causal);
    break;
  }
  case 23: {
    auto coordinates = at::nonzero(a[0]);
    auto counts = at::bincount(coordinates.select(1, 0), std::nullopt, a[0].size(0));
    o[0].fill_(-1);
    o[1].copy_(counts);
    auto columns = coordinates.select(1, 1).to(at::kInt);
    const int64_t *sizes = counts.const_data_ptr<int64_t>();
    int64_t begin = 0;
    for (int64_t row=0; row<a[0].size(0); ++row) {
      o[0][row].narrow(0, 0, sizes[row]).copy_(columns.narrow(0, begin, sizes[row]));
      begin += sizes[row];
    }
    break;
  }
  case 24:
    for (size_t i=0; i<a.size(); ++i) w[i].copy_(a[i]);
    // E8M0 byte 255 is NaN; byte zero denotes 2^-127, not zero.
    for (int i : {1, 3}) {
      w[i].sub_(127.0).exp2_();
      at::eq_out(w[4 + i/2], a[i], 255);
      w[i].masked_fill_(w[4 + i/2], std::numeric_limits<float>::quiet_NaN());
    }
    w[0].mul_(w[1].unsqueeze(-1));
    w[2].mul_(w[3].unsqueeze(1));
    at::mm_out(o[0], w[0].flatten(1), w[2].flatten(0, 1));
    break;
  case 25: {
    const int32_t *offsets = a[3].const_data_ptr<int32_t>();
    const int32_t *lengths = a[5].const_data_ptr<int32_t>();
    const int32_t *splitOffsets = a[6].const_data_ptr<int32_t>();
    TORCH_CHECK(a[0].size(0)>0 && a[6].numel()>1 && (a[6].numel()-1)%a[0].size(0)==0,
                "split offsets must describe the same number of partitions for each sequence");
    const int64_t splits = (a[6].numel()-1)/a[0].size(0);
    const int64_t pageSize = a[1].size(1), heads = a[1].size(2);
    for (int64_t batch=0; batch<a[0].size(0); ++batch) {
      int64_t begin = offsets[batch], end = offsets[batch+1], length = lengths[batch];
      TORCH_CHECK(0<=begin && begin<=end && end<=a[4].numel() &&
                  0<=length && length<=(end-begin)*pageSize, "invalid paged sequence bounds");
      TORCH_CHECK(splitOffsets[batch*splits]==begin && splitOffsets[(batch+1)*splits]==end,
                  "split partitions must cover each sequence's page table");
      for (int64_t split=0; split<splits; ++split)
        TORCH_CHECK(splitOffsets[batch*splits+split]<=splitOffsets[batch*splits+split+1],
                    "split partitions must preserve page order");
      if (!length) { o[0][batch].zero_(); continue; }
      auto pages = a[4].narrow(0, begin, (length+pageSize-1)/pageSize);
      auto gather = [&](const at::Tensor &cache) {
        return at::index_select(cache, 0, pages).view({-1, heads, cache.size(3)})
            .narrow(0, 0, length).permute({1, 0, 2}).unsqueeze(0);
      };
      auto key = gather(a[1]), value = gather(a[2]);
      auto query = a[0][batch].unsqueeze(0).unsqueeze(2);
      auto output = std::get<0>(at::_scaled_dot_product_flash_attention_for_cpu(
          query, key, value, 0.0, false, std::nullopt, s[0]));
      o[0][batch].copy_(output.squeeze(0).squeeze(1));
    }
    break;
  }
#ifdef INTENT_BENCH_ONEDNN
  case 26: {
    auto batch=a[1].size(0), heads=a[1].size(2), groups=a[0].size(2);
    auto chunks=a[2].size(2), size=a[2].size(3), width=a[1].size(3), states=a[0].size(3);
    at::sub_out(w[0],a[3].select(3,size-1).unsqueeze(3),a[3]);
    w[0].clamp_max_(0).exp_().mul_(a[2]);
    auto basis=a[0].view({batch,chunks,size,groups,states}).permute({0,1,3,2,4}).unsqueeze(3);
    w[1].copy_(basis);
    auto scale=w[0].permute({0,2,1,3}).reshape({batch,chunks,groups,heads/groups,size,1});
    w[1].mul_(scale);
    w[2].copy_(w[1]); // Preserve the author's BF16 rounding before contraction.
    w[3].copy_(a[1].view({batch,chunks,size,heads,width}).permute({0,1,3,4,2}));
    bench_dnnl_matmul_run(call.matrix);
    TORCH_CHECK(!*bench_dnnl_matmul_error(call.matrix), bench_dnnl_matmul_error(call.matrix));
    break;
  }
#endif
  case 27:
    at::bmm_out(w[0], a[0], a[1].transpose(1, 2));
    w[0].tril_();
    at::bmm_out(o[0], w[0], a[2]);
    at::bmm_out(o[1], a[1].transpose(1, 2), a[2]);
    break;
#ifdef INTENT_BENCH_AOTI
  case 28:
    TORCH_CHECK(call.package, "compiled CPU package has not been loaded");
    call.package->run(a);
    break;
#endif
#ifdef INTENT_BENCH_VLLM
  case 29:
    w[0].zero_();
    o[0].fill_(a[0].numel());
    o[1].fill_(-1);
    o[2].data_ptr<int32_t>()[0] = bench_vllm_moe_align(
        o[0].data_ptr<int32_t>(), o[1].data_ptr<int32_t>(), a[0].data_ptr<int32_t>(),
        w[0].data_ptr<int32_t>(), w[1].data_ptr<int32_t>(), w[2].data_ptr<int32_t>(),
        s[0], a[0].numel(), at::get_num_threads());
    break;
#endif
  case 30: {
    int64_t groups = a[0].size(1) / a[1].size(1);
    auto key = groups == 1 ? a[1] : at::repeat_interleave(a[1], groups, 1);
    auto value = groups == 1 ? a[2] : at::repeat_interleave(a[2], groups, 1);
    auto [dq, dk, dv] = at::_scaled_dot_product_flash_attention_for_cpu_backward(
        a[4], a[0], key, value, a[3], a[5], 0.0, true, std::nullopt, s[0]);
    auto reduceHeads = [&](at::Tensor gradient) {
      if (groups == 1) return gradient;
      return gradient.reshape({gradient.size(0), a[1].size(1), groups,
                               gradient.size(2), gradient.size(3)})
          .sum(2, false, at::kFloat).to(gradient.scalar_type());
    };
    o = {dq, reduceHeads(dk), reduceHeads(dv)};
    break;
  }
  case 31: case 32: {
    int64_t width = a[2].size(1), segments = a[1].numel()-1;
    std::vector<at::Tensor> pieces, histories;
    const auto *offsets = a[1].const_data_ptr<int64_t>();
    TORCH_CHECK(offsets[0] == 0 && offsets[segments] == a[0].size(0),
                "sequence offsets must cover the input");
    for (int64_t i=0; i<segments; ++i) {
      int64_t begin = offsets[i], end = offsets[i+1];
      TORCH_CHECK(begin >= 0 && begin <= end && end <= a[0].size(0), "invalid sequence offset");
      int64_t length = end - begin, history = std::min(length, width);
      auto values = a[0].narrow(0, begin, length);
      if (call.operation == 32) {
        auto padded = at::constant_pad_nd(values.to(at::kFloat), {0, 0, width-1, 0});
        if (length) {
          auto sum = (padded.unfold(0, width, 1) * a[2]).sum(-1) + a[3];
          pieces.push_back((sum * at::sigmoid(sum)).to(a[0].scalar_type()));
        } else {
          pieces.push_back(values);
        }
        auto paddedHistory = at::constant_pad_nd(values, {0, 0, width-history, 0});
        histories.push_back(paddedHistory.narrow(0, paddedHistory.size(0)-width, width));
        continue;
      }
      auto state = o[1].select(0, i);
      state.narrow(0, 0, width-history).zero_();
      state.narrow(0, width-history, history).copy_(a[0].narrow(0, end-history, history));
      if (!length) continue;
      auto channels = values.to(at::kFloat).transpose(0, 1).unsqueeze(0);
      auto convolution = at::conv1d(channels, a[2].unsqueeze(1), a[3],
                                     {1}, {width-1}, {1}, a[0].size(1));
      o[0].narrow(0, begin, length).copy_(at::silu(convolution.narrow(2, 0, length)).squeeze(0).transpose(0, 1));
    }
    if (call.operation == 32) o = {at::cat(pieces), at::stack(histories)};
    break;
  }
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
#ifdef INTENT_BENCH_AOTI
extern "C" void bench_aten_load_package(void *opaque, const char *path) {
  auto &call = *static_cast<Call *>(opaque);
  try {
    call.package = std::make_unique<torch::inductor::AOTIModelPackageLoader>(path);
  } catch (const std::exception &error) { call.error = error.what(); }
}
#endif
extern "C" const char *bench_aten_error(void *opaque) { return static_cast<Call *>(opaque)->error.c_str(); }
extern "C" const char *bench_aten_implementation(void *opaque) {
#ifdef INTENT_BENCH_ONEDNN
  auto &call=*static_cast<Call *>(opaque);
  if (call.matrix) return bench_dnnl_matmul_implementation(call.matrix);
#endif
  return "ATen";
}
extern "C" int bench_aten_outputs(void *opaque) { return static_cast<Call *>(opaque)->outputs.size(); }
extern "C" void *bench_aten_output(void *opaque, int output, int64_t *rank,
                                   const int64_t **sizes, const int64_t **strides, int *type) {
  const auto &tensor = static_cast<Call *>(opaque)->outputs.at(output);
  *rank = tensor.dim();
  *sizes = tensor.sizes().data();
  *strides = tensor.strides().data();
  *type = -1;
  for (int kind = 0; kind <= 7; ++kind)
    if (tensor.scalar_type() == dtype(kind)) *type = kind;
  return tensor.data_ptr();
}
extern "C" void bench_aten_destroy(void *opaque) { delete static_cast<Call *>(opaque); }
