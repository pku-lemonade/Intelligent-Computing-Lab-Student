#include <torch/extension.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
  // TODO_BEGIN(W03_T01)
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = 0.0f;
  // TODO_END(W03_T01)
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "inputs must be float32");
  TORCH_CHECK(a.numel() == b.numel(), "input sizes must match");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");
  auto out = torch::empty_like(a);
  int n = a.numel();
  if (n == 0) return out;
  // TODO_BEGIN(W03_T02)
  // Configure and launch vector_add_kernel here.
  // TODO_END(W03_T02)
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", &vector_add, "vector_add");
}
