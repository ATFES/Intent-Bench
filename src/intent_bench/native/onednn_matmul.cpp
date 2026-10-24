#include <dnnl.hpp>
#include <dnnl_version.h>
#include <memory>
#include <string>
#include <unordered_map>

#if DNNL_VERSION_MAJOR < 3
#error "This comparison requires oneDNN 3.x or newer; select its prefix in the Bench environment."
#endif

namespace {
struct Matrix {
  dnnl::engine engine{dnnl::engine::kind::cpu, 0};
  dnnl::stream stream{engine};
  dnnl::matmul operation;
  std::unordered_map<int, dnnl::memory> arguments;
  std::string implementation, error;

  Matrix(int64_t batch, int64_t m, int64_t n, int64_t k,
         void *left, void *right, void *output) {
    using memory = dnnl::memory;
    memory::desc a({batch, m, k}, memory::data_type::bf16, memory::format_tag::abc);
    memory::desc b({batch, k, n}, memory::data_type::bf16, memory::format_tag::abc);
    memory::desc c({batch, m, n}, memory::data_type::f32, memory::format_tag::abc);
    dnnl::matmul::primitive_desc descriptor(engine, a, b, c);
    operation = dnnl::matmul(descriptor);
    implementation = descriptor.impl_info_str();
    arguments = {{DNNL_ARG_SRC, memory(a, engine, left)},
                 {DNNL_ARG_WEIGHTS, memory(b, engine, right)},
                 {DNNL_ARG_DST, memory(c, engine, output)}};
  }
};
thread_local std::string create_error;
}

extern "C" void *bench_dnnl_matmul_create(int64_t batch, int64_t m, int64_t n, int64_t k,
                                          void *left, void *right, void *output) {
  create_error.clear();
  try { return new Matrix(batch, m, n, k, left, right, output); }
  catch (const std::exception &error) { create_error = error.what(); return nullptr; }
}
extern "C" const char *bench_dnnl_matmul_error(void *opaque) {
  return opaque ? static_cast<Matrix *>(opaque)->error.c_str() : create_error.c_str();
}
extern "C" const char *bench_dnnl_matmul_implementation(void *opaque) {
  return static_cast<Matrix *>(opaque)->implementation.c_str();
}
extern "C" void bench_dnnl_matmul_run(void *opaque) {
  auto &matrix = *static_cast<Matrix *>(opaque);
  try {
    matrix.operation.execute(matrix.stream, matrix.arguments);
    matrix.stream.wait();
  } catch (const std::exception &error) { matrix.error = error.what(); }
}
extern "C" void bench_dnnl_matmul_destroy(void *opaque) {
  delete static_cast<Matrix *>(opaque);
}
