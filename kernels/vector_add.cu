#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
  // TODO_BEGIN(W03_T01)
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = 0.0f;
  // TODO_END(W03_T01)
}

extern "C" void launch_vector_add(const float* a, const float* b, float* out, int n) {
  // TODO_BEGIN(W03_T02)
  // Configure and launch vector_add_kernel here.
  // TODO_END(W03_T02)
}
