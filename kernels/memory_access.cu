#include <torch/extension.h>

__global__ void copy_coalesced_kernel(const float* input, float* output, int n) {
  // TODO_BEGIN(W05_T01)
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) output[idx] = 0.0f;
  // TODO_END(W05_T01)
}

__global__ void copy_strided_kernel(
    const float* input, float* output, int rows, int cols, int n) {
  // TODO_BEGIN(W05_T02)
  const int logical = blockIdx.x * blockDim.x + threadIdx.x;
  if (logical < n) output[logical] = 0.0f;
  // TODO_END(W05_T02)
}

__global__ void transpose_naive_kernel(
    const float* input, float* output, int rows, int cols) {
  // TODO_BEGIN(W05_T03)
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) output[col * rows + row] = 0.0f;
  // TODO_END(W05_T03)
}

void check_matrix(const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(input.dim() == 2, "input must be a matrix");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
}

torch::Tensor copy_coalesced(torch::Tensor input) {
  check_matrix(input);
  auto output = torch::empty_like(input);
  const int n = input.numel();
  if (n == 0) return output;
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  copy_coalesced_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
  return output;
}

torch::Tensor copy_strided(torch::Tensor input) {
  check_matrix(input);
  auto output = torch::empty_like(input);
  const int rows = input.size(0);
  const int cols = input.size(1);
  const int n = input.numel();
  if (n == 0) return output;
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  copy_strided_kernel<<<blocks, threads>>>(
      input.data_ptr<float>(), output.data_ptr<float>(), rows, cols, n);
  return output;
}

torch::Tensor transpose_naive(torch::Tensor input) {
  check_matrix(input);
  auto output = torch::empty({input.size(1), input.size(0)}, input.options());
  const int rows = input.size(0);
  const int cols = input.size(1);
  if (rows == 0 || cols == 0) return output;
  const dim3 threads(16, 16);
  const dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);
  transpose_naive_kernel<<<blocks, threads>>>(
      input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("copy_coalesced", &copy_coalesced, "Coalesced matrix copy");
  m.def("copy_strided", &copy_strided, "Column-major traversal of a row-major matrix");
  m.def("transpose_naive", &transpose_naive, "Naive matrix transpose");
}
