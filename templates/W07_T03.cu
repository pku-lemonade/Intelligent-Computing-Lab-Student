#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;

template <typename T>
__device__ float to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ float to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ T from_float(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __half from_float<__half>(float value) {
  return __float2half(value);
}

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

__device__ float warp_reduce_sum(float value) {
  // TODO_BEGIN(W07_T01)
  return value;
  // TODO_END(W07_T01)
}

__device__ float warp_reduce_max(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

template <typename scalar_t>
__global__ void row_sum_kernel(const scalar_t* input, scalar_t* output, int rows, int cols) {
  // TODO_BEGIN(W07_T02)
  const int row = blockIdx.x;
  if (threadIdx.x == 0) output[row] = from_float<scalar_t>(0.0f);
  // TODO_END(W07_T02)
}

template <typename scalar_t>
__global__ void row_max_kernel(const scalar_t* input, scalar_t* output, int rows, int cols) {
  // TODO_BEGIN(W07_T03)
  const int row = blockIdx.x;
  if (threadIdx.x == 0) output[row] = from_float<scalar_t>(0.0f);
  // TODO_END(W07_T03)
}

template <typename scalar_t>
void launch_row_sum(const torch::Tensor& input, torch::Tensor& output, int rows, int cols) {
  row_sum_kernel<<<rows, kThreads>>>(
      reinterpret_cast<const scalar_t*>(input.data_ptr()),
      reinterpret_cast<scalar_t*>(output.data_ptr()), rows, cols);
}

template <typename scalar_t>
void launch_row_max(const torch::Tensor& input, torch::Tensor& output, int rows, int cols) {
  row_max_kernel<<<rows, kThreads>>>(
      reinterpret_cast<const scalar_t*>(input.data_ptr()),
      reinterpret_cast<scalar_t*>(output.data_ptr()), rows, cols);
}

void check_input(const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must have shape [rows, cols]");
  TORCH_CHECK(input.size(1) > 0, "cols must be positive");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(
      input.scalar_type() == torch::kFloat32 ||
          input.scalar_type() == torch::kFloat16 ||
          input.scalar_type() == torch::kBFloat16,
      "input must be float32, float16 or bfloat16");
}

torch::Tensor row_sum(torch::Tensor input) {
  check_input(input);
  const int rows = input.size(0);
  const int cols = input.size(1);
  auto output = torch::empty({rows}, input.options());
  if (rows == 0) return output;
  if (input.scalar_type() == torch::kFloat32) launch_row_sum<float>(input, output, rows, cols);
  if (input.scalar_type() == torch::kFloat16) launch_row_sum<__half>(input, output, rows, cols);
  if (input.scalar_type() == torch::kBFloat16) launch_row_sum<__nv_bfloat16>(input, output, rows, cols);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "row_sum kernel launch failed");
  return output;
}

torch::Tensor row_max(torch::Tensor input) {
  check_input(input);
  const int rows = input.size(0);
  const int cols = input.size(1);
  auto output = torch::empty({rows}, input.options());
  if (rows == 0) return output;
  if (input.scalar_type() == torch::kFloat32) launch_row_max<float>(input, output, rows, cols);
  if (input.scalar_type() == torch::kFloat16) launch_row_max<__half>(input, output, rows, cols);
  if (input.scalar_type() == torch::kBFloat16) launch_row_max<__nv_bfloat16>(input, output, rows, cols);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "row_max kernel launch failed");
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("row_sum", &row_sum, "Row-wise sum reduction");
  m.def("row_max", &row_max, "Row-wise maximum reduction");
}
