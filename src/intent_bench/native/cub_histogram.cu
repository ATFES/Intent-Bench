#include <cub/device/device_histogram.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/version.cuh>
#include <cuda_runtime.h>
#include <cstdint>

struct BinIndex {
  __host__ __device__ int operator()(float value) const {
    return static_cast<int>(value);
  }
};

// TransformInputIterator preserves the author's f32 -> i32 bin conversion
// without a separate preprocessing kernel. CUB owns the histogram algorithm,
// including output initialization, partial counts and their merge.
extern "C" int bench_cub_histogram(void *temporary, size_t *bytes,
                                   const float *samples, int *counts,
                                   int64_t size, cudaStream_t stream) {
  cub::TransformInputIterator<int, BinIndex, const float *> input(samples, BinIndex{});
  return cub::DeviceHistogram::HistogramEven(temporary, *bytes, input, counts,
                                            257, 0, 256, size, stream);
}

extern "C" const char *bench_cuda_error(int status) {
  return cudaGetErrorString(static_cast<cudaError_t>(status));
}

extern "C" int bench_cub_version() { return CUB_VERSION; }
