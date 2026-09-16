#include <torch/extension.h>

__global__ void scale_kernel(const float* x, float* out, float scale, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = scale * x[idx];
}

__global__ void add_bias_kernel(const float* x, float* out, float bias, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = x[idx] + bias;
}

__global__ void relu_kernel(const float* x, float* out, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
}

__global__ void fused_scale_bias_relu_kernel(
    const float* x, float* out, float scale, float bias, int n) {
  // TODO_BEGIN(W04_T02)
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = 0.0f;
  // TODO_END(W04_T02)
}

void check_input(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
}

torch::Tensor unfused_scale_bias_relu(torch::Tensor x, double scale, double bias) {
  check_input(x);
  auto first = torch::empty_like(x);
  auto second = torch::empty_like(x);
  auto out = torch::empty_like(x);
  const int n = x.numel();
  if (n == 0) return out;
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  scale_kernel<<<blocks, threads>>>(x.data_ptr<float>(), first.data_ptr<float>(), static_cast<float>(scale), n);
  add_bias_kernel<<<blocks, threads>>>(first.data_ptr<float>(), second.data_ptr<float>(), static_cast<float>(bias), n);
  relu_kernel<<<blocks, threads>>>(second.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}

torch::Tensor fused_scale_bias_relu(torch::Tensor x, double scale, double bias) {
  check_input(x);
  auto out = torch::empty_like(x);
  const int n = x.numel();
  if (n == 0) return out;
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  fused_scale_bias_relu_kernel<<<blocks, threads>>>(
      x.data_ptr<float>(), out.data_ptr<float>(), static_cast<float>(scale), static_cast<float>(bias), n);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unfused_scale_bias_relu", &unfused_scale_bias_relu, "Unfused scale, bias and ReLU");
  m.def("fused_scale_bias_relu", &fused_scale_bias_relu, "Fused scale, bias and ReLU");
}
