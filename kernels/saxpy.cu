#include <cuda_runtime.h>

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

extern "C" void launch_saxpy(const float* x, const float* y, float* out, float alpha, int n) {
  if (n == 0) return;
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  saxpy_kernel<<<blocks, threads>>>(x, y, out, alpha, n);
}

extern "C" void launch_saxpy_grid_stride(const float* x, const float* y, float* out, float alpha, int n, int grid_blocks) {
  if (n == 0) return;
  constexpr int threads = 256;
  saxpy_grid_stride_kernel<<<grid_blocks, threads>>>(x, y, out, alpha, n);
}
