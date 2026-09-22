#include <torch/extension.h>

extern "C" void launch_saxpy(const float* x, const float* y, float* out, float alpha, int n);
extern "C" void launch_saxpy_grid_stride(const float* x, const float* y, float* out, float alpha, int n, int blocks);

static void check_inputs(const torch::Tensor& x, const torch::Tensor& y) {
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
  launch_saxpy(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), static_cast<float>(alpha), n);
  return out;
}

torch::Tensor saxpy_grid_stride(torch::Tensor x, torch::Tensor y, double alpha, int blocks) {
  check_inputs(x, y);
  TORCH_CHECK(blocks > 0, "blocks must be positive");
  auto out = torch::empty_like(x);
  const int n = x.numel();
  if (n == 0) return out;
  launch_saxpy_grid_stride(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), static_cast<float>(alpha), n, blocks);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("saxpy", &saxpy, "SAXPY");
  m.def("saxpy_grid_stride", &saxpy_grid_stride, "SAXPY with a grid-stride loop");
}
