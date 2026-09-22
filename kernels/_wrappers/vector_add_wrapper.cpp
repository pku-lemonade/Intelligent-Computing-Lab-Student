#include <torch/extension.h>

extern "C" void launch_vector_add(const float* a, const float* b, float* out, int n);

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "inputs must be float32");
  TORCH_CHECK(a.numel() == b.numel(), "input sizes must match");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");
  auto out = torch::empty_like(a);
  int n = a.numel();
  if (n == 0) return out;
  launch_vector_add(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", &vector_add, "vector_add");
}
