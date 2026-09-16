#include <torch/extension.h>

__global__ void saxpy_kernel(const float* x, const float* y, float* out, float alpha, int n) {
  // TODO_BEGIN(W03_T03)
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = 0.0f;
  // TODO_END(W03_T03)
}

__global__ void saxpy_grid_stride_kernel(const float* x, const float* y, float* out, float alpha, int n) {
  // TODO_BEGIN(W04_T01)
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = 0.0f;
  // TODO_END(W04_T01)
}

void check_inputs(const torch::Tensor& x, const torch::Tensor& y) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.dtype() == torch::kFloat32 && y.dtype() == torch::kFloat32, "inputs must be float32");
  TORCH_CHECK(x.sizes() == y.sizes(), "input shapes must match");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "inputs must be contiguous");
}

torch::Tensor saxpy(torch::Tensor x, torch::Tensor y, double alpha) {
  check_inputs(x, y);
  auto out = torch::empty_like(x);
  const int n = x.numel();
  if (n == 0) return out;
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  saxpy_kernel<<<blocks, threads>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), static_cast<float>(alpha), n);
  return out;
}

torch::Tensor saxpy_grid_stride(torch::Tensor x, torch::Tensor y, double alpha, int blocks) {
  check_inputs(x, y);
  TORCH_CHECK(blocks > 0, "blocks must be positive");
  auto out = torch::empty_like(x);
  const int n = x.numel();
  if (n == 0) return out;
  constexpr int threads = 256;
  saxpy_grid_stride_kernel<<<blocks, threads>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), static_cast<float>(alpha), n);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("saxpy", &saxpy, "SAXPY");
  m.def("saxpy_grid_stride", &saxpy_grid_stride, "SAXPY with a grid-stride loop");
}
