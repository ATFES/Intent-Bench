#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

struct Operation {
  int kind, m, n, k;
  ggml_context *context = nullptr;
  ggml_backend_t backend = nullptr;
  ggml_backend_buffer_t buffer = nullptr;
  ggml_backend_graph_plan_t plan = nullptr;
  ggml_threadpool_t threadpool = nullptr;
  ggml_tensor *weight = nullptr, *input = nullptr, *output = nullptr, *state_output = nullptr;
  ggml_cgraph *graph = nullptr;
  std::vector<float> a, b, result, reference;
  std::vector<unsigned char> packed_weight, packed_input;
  ~Operation() {
    if (plan) ggml_backend_graph_plan_free(backend, plan);
    if (buffer) ggml_backend_buffer_free(buffer);
    if (context) ggml_free(context);
    if (backend) ggml_backend_free(backend);
    if (threadpool) ggml_threadpool_free(threadpool);
  }
  void prepare(int threads) {
    auto parameters = ggml_threadpool_params_default(threads);
    threadpool = ggml_threadpool_new(&parameters);
    if (!threadpool) throw std::runtime_error("GGML threadpool allocation failed");
    ggml_backend_cpu_set_threadpool(backend, threadpool);
    plan = ggml_backend_graph_plan_create(backend, graph);
    if (!plan) throw std::runtime_error("GGML graph plan allocation failed");
  }
  void run() {
    if (kind == 4) {
      ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(b.data(), packed_input.data(), k);
    } else if (kind == 3) {
      const auto *trait = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
      const size_t bytes = ggml_row_size(GGML_TYPE_Q4_K, k);
      for (int row = 0; row < n; ++row)
        trait->vec_dot(k, &result[row], 0, packed_weight.data()+row*bytes, 0, packed_input.data(), 0, 1);
    } else {
      if (ggml_backend_graph_plan_compute(backend, plan) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("GGML graph execution failed");
      ggml_backend_synchronize(backend);
    }
  }
};

static thread_local std::string last_error;
extern "C" const char *bench_error() { return last_error.c_str(); }

extern "C" void *bench_create(int kind, const void *weight_data, const float *b, int m, int n, int k, int threads) {
  try {
    if (m<=0 || n<=0 || k<=0 || threads<=0 || kind<0 || kind>5)
      throw std::runtime_error("invalid GGML workload geometry");
    if ((kind==1 || kind==3 || kind==4 || kind==5) && k%256)
      throw std::runtime_error("Q4_K/Q8_K require K divisible by 256");
    auto op=std::make_unique<Operation>();op->kind=kind;op->m=m;op->n=n;op->k=k;
    const auto *a = static_cast<const float *>(weight_data);
    op->a.resize(n*k);
    if (kind!=5) std::copy(a,a+n*k,op->a.begin());
    op->b.assign(b,b+m*k);op->result.resize(m*n);op->reference.resize(m*n);
    ggml_cpu_init();
    ggml_type type=kind==1 || kind==3 || kind==5 ? GGML_TYPE_Q4_K : GGML_TYPE_F32;
    if (kind==1 || kind==3 || kind==5) {
      op->packed_weight.resize(ggml_row_size(type,k)*n);
      if (kind==5) std::memcpy(op->packed_weight.data(),weight_data,op->packed_weight.size());
      else ggml_quantize_chunk(type,a,op->packed_weight.data(),0,n,k,nullptr);
      for (int row=0;row<n;++row)
        ggml_get_type_traits(type)->to_float(op->packed_weight.data()+row*ggml_row_size(type,k),op->a.data()+row*k,k);
    }
    if (kind==1 || kind==3 || kind==4 || kind==5) {
      op->packed_input.resize(ggml_row_size(GGML_TYPE_Q8_K,k)*m);
      for (int row=0;row<m;++row) {
        auto *encoded=op->packed_input.data()+row*ggml_row_size(GGML_TYPE_Q8_K,k);
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(b+row*k,encoded,k);
        if(kind!=4) {
          // Q8_K is an intermediate dot format; its public to_float slot may be null.
          for(int block=0;block<k/256;++block) {
            float scale;std::memcpy(&scale,encoded+block*292,sizeof(float));
            const auto *values=reinterpret_cast<const int8_t*>(encoded+block*292+4);
            for(int j=0;j<256;++j)op->b[row*k+block*256+j]=scale*values[j];
          }
        }
      }
    }
    if (kind!=4) {
      for (int row=0;row<m;++row) {
        double mean_square=0;
        if (kind==2) for(int j=0;j<k;++j)mean_square+=double(b[row*k+j])*b[row*k+j];
        for(int col=0;col<n;++col) {
          double value=0;
          if(kind==2)value=b[row*k+col]/std::sqrt(mean_square/k+1e-5);
          else for(int j=0;j<k;++j)value+=double(op->a[col*k+j])*op->b[row*k+j];
          op->reference[row*n+col]=float(value);
        }
      }
    }
    if(kind==3 || kind==4)return op.release();
    op->backend=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(op->backend,threads);
    op->context=ggml_init({16*ggml_tensor_overhead()+ggml_graph_overhead_custom(32,false),nullptr,true});
    if(!op->context || !op->backend)throw std::runtime_error("GGML initialization failed");
    op->input=ggml_new_tensor_2d(op->context,GGML_TYPE_F32,k,m);
    if(kind==2)op->output=ggml_rms_norm(op->context,op->input,1e-5f);
    else {
      op->weight=ggml_new_tensor_2d(op->context,type,k,n);
      op->output=ggml_mul_mat(op->context,op->weight,op->input);
    }
    op->graph=ggml_new_graph_custom(op->context,32,false);ggml_build_forward_expand(op->graph,op->output);
    op->buffer=ggml_backend_alloc_ctx_tensors(op->context,op->backend);
    if(!op->buffer)throw std::runtime_error("GGML buffer allocation failed");
    ggml_backend_tensor_set(op->input,b,0,m*k*sizeof(float));
    if(op->weight)ggml_backend_tensor_set(op->weight,type==GGML_TYPE_F32 ? (const void*)a : op->packed_weight.data(),0,ggml_nbytes(op->weight));
    op->prepare(threads);
    return op.release();
  } catch(const std::exception &e){last_error=e.what();return nullptr;}
}

extern "C" void *bench_gated_delta_create(const void *const *data, int batch, int tokens,
                                          int key_heads, int value_heads, int dimension, int threads) {
  try {
    if (batch<=0 || tokens<=0 || key_heads<=0 || value_heads<=0 || dimension<=0 ||
        threads<=0 || value_heads%key_heads)
      throw std::runtime_error("invalid GGML gated delta geometry");
    auto op = std::make_unique<Operation>();
    op->kind = 6;
    ggml_cpu_init();
    op->backend = ggml_backend_cpu_init();
    if (!op->backend) throw std::runtime_error("GGML CPU initialization failed");
    ggml_backend_cpu_set_n_threads(op->backend, threads);
    op->context = ggml_init({64*ggml_tensor_overhead()+ggml_graph_overhead_custom(64,false), nullptr, true});
    if (!op->context) throw std::runtime_error("GGML context allocation failed");
    auto *ctx = op->context;
    ggml_tensor *inputs[5] = {
      ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, dimension, key_heads, tokens, batch),
      ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, dimension, key_heads, tokens, batch),
      ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, dimension, value_heads, tokens, batch),
      ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 1, value_heads, tokens, batch),
      ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 1, value_heads, tokens, batch),
    };
    ggml_tensor *values[5];
    for (int i=0; i<5; ++i) values[i] = ggml_cast(ctx, inputs[i], GGML_TYPE_F32);
    // GGML broadcasts heads cyclically; the author groups adjacent value heads.
    // Expand Q/K within the timed graph, retaining the original input storage.
    if (value_heads != key_heads) {
      for (int i=0; i<2; ++i) {
        auto *grouped = ggml_reshape_4d(ctx, values[i], dimension, 1, key_heads, int64_t(tokens)*batch);
        auto *expanded = ggml_repeat_4d(ctx, grouped, dimension, value_heads/key_heads,
                                        key_heads, int64_t(tokens)*batch);
        values[i] = ggml_reshape_4d(ctx, expanded, dimension, value_heads, tokens, batch);
      }
    }
    auto *initial = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dimension, dimension, value_heads, batch);
    auto *packed = ggml_gated_delta_net(ctx, values[0], values[1], values[2], values[3], values[4], initial, 1);
    const int64_t score_count = int64_t(batch)*tokens*value_heads*dimension;
    auto *scores = ggml_view_1d(ctx, packed, score_count, 0);
    op->output = ggml_cast(ctx, ggml_reshape_4d(ctx, scores, dimension, value_heads, tokens, batch), GGML_TYPE_BF16);
    auto *state = ggml_view_4d(ctx, packed, dimension, dimension, value_heads, batch,
        dimension*sizeof(float), int64_t(dimension)*dimension*sizeof(float),
        int64_t(dimension)*dimension*value_heads*sizeof(float), score_count*sizeof(float));
    // Native GGML stores S transposed. Materialize the public [B,H,K,V] result.
    op->state_output = ggml_cont(ctx, ggml_permute(ctx, state, 1, 0, 2, 3));
    op->graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(op->graph, op->output);
    ggml_build_forward_expand(op->graph, op->state_output);
    op->buffer = ggml_backend_alloc_ctx_tensors(ctx, op->backend);
    if (!op->buffer) throw std::runtime_error("GGML buffer allocation failed");
    for (int i=0; i<5; ++i) ggml_backend_tensor_set(inputs[i], data[i], 0, ggml_nbytes(inputs[i]));
    std::vector<float> zero_state(ggml_nelements(initial), 0.0f);
    ggml_backend_tensor_set(initial, zero_state.data(), 0, ggml_nbytes(initial));
    op->prepare(threads);
    // The native recurrence copies initial state on every invocation.
    return op.release();
  } catch (const std::exception &e) { last_error=e.what(); return nullptr; }
}

extern "C" int bench_run(void *ptr) {
  try{static_cast<Operation*>(ptr)->run();return 0;}
  catch(const std::exception &e){last_error=e.what();return 1;}
}
extern "C" void bench_output(void *ptr, void *out, int reference) {
  auto &op=*static_cast<Operation*>(ptr);
  if(op.kind==6)ggml_backend_tensor_get(op.output,out,0,ggml_nbytes(op.output));
  else if(op.kind==4)std::memcpy(out,op.packed_input.data(),op.packed_input.size());
  else if(reference)std::memcpy(out,op.reference.data(),op.reference.size()*sizeof(float));
  else if(op.kind==3)std::memcpy(out,op.result.data(),op.result.size()*sizeof(float));
  else ggml_backend_tensor_get(op.output,out,0,op.result.size()*sizeof(float));
}
extern "C" void bench_state_output(void *ptr, void *out) {
  auto &op=*static_cast<Operation*>(ptr);
  ggml_backend_tensor_get(op.state_output,out,0,ggml_nbytes(op.state_output));
}
extern "C" double bench_measure(void *ptr, int repeat) {
  auto begin=std::chrono::steady_clock::now();
  for(int i=0;i<repeat;++i)if(bench_run(ptr))return -1;
  return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/repeat;
}
extern "C" void bench_destroy(void *ptr) {delete static_cast<Operation*>(ptr);}
extern "C" void bench_packed_weight(void *ptr, void *out) {
  auto &op=*static_cast<Operation*>(ptr);
  std::memcpy(out,op.packed_weight.data(),op.packed_weight.size());
}
