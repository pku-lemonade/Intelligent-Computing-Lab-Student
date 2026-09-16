#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>

namespace {

constexpr int kThreads = 256;

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

template <typename scalar_t>
__global__ void softmax_kernel(const scalar_t* x, scalar_t* out, int rows, int cols) {
  // TODO_BEGIN(W08_T01)
  const int row = blockIdx.x;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    out[row * cols + col] = from_float<scalar_t>(0.0f);
  }
  // TODO_END(W08_T01)
}

template <typename scalar_t>
__global__ void rms_norm_kernel(const scalar_t* x, const scalar_t* weight, scalar_t* out, int rows, int cols, float eps) {
  // TODO_BEGIN(W08_T02)
  const int row = blockIdx.x;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    out[row * cols + col] = x[row * cols + col];
  }
  // TODO_END(W08_T02)
}

template <typename scalar_t>
__global__ void rope_kernel(const scalar_t* x, const scalar_t* cos, const scalar_t* sin, scalar_t* out, int rows, int cols) {
  int row = blockIdx.x;
  int half = cols / 2;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    int rotated_col = col < half ? col + half : col - half;
    float sign = col < half ? -1.0f : 1.0f;
    int idx = row * cols + col;
    out[idx] = from_float<scalar_t>(to_float(x[idx]) * to_float(cos[idx]) + sign * to_float(x[row * cols + rotated_col]) * to_float(sin[idx]));
  }
}

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, int m, int n, int k) {
  // TODO_BEGIN(W06_T01)
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) out[row * n + col] = from_float<scalar_t>(0.0f);
  // TODO_END(W06_T01)
}

template <typename scalar_t, int Tile>
__global__ void matmul_tiled_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, int m, int n, int k) {
  // TODO_BEGIN(W06_T02)
  const int row = blockIdx.y * Tile + threadIdx.y;
  const int col = blockIdx.x * Tile + threadIdx.x;
  if (row < m && col < n) out[row * n + col] = from_float<scalar_t>(0.0f);
  // TODO_END(W06_T02)
}

template <typename scalar_t>
__global__ void paged_attention_decode_kernel(
    const scalar_t* query,
    const scalar_t* key_cache,
    const scalar_t* value_cache,
    const int64_t* block_tables,
    const int64_t* context_lens,
    scalar_t* out,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks,
    int block_size,
    float scale) {
  // TODO_BEGIN(W12_T05)
  const int batch = blockIdx.x;
  const int head = blockIdx.y;
  const int dim = threadIdx.x;
  if (batch < batch_size && head < num_heads && dim < head_dim) {
    out[(batch * num_heads + head) * head_dim + dim] = from_float<scalar_t>(0.0f);
  }
  // TODO_END(W12_T05)
}

template <typename scalar_t>
__global__ void dense_attention_prefill_kernel(
    const scalar_t* query,
    const scalar_t* key,
    const scalar_t* value,
    scalar_t* out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale) {
  // TODO_BEGIN(W09_T02)
  const int batch = blockIdx.x;
  const int head = blockIdx.y;
  const int query_pos = blockIdx.z;
  const int dim = threadIdx.x;
  if (batch < batch_size && head < num_heads && query_pos < seq_len && dim < head_dim) {
    out[((batch * num_heads + head) * seq_len + query_pos) * head_dim + dim] = from_float<scalar_t>(0.0f);
  }
  // TODO_END(W09_T02)
}

template <typename scalar_t>
__global__ void dense_attention_decode_kernel(
    const scalar_t* query,
    const scalar_t* key,
    const scalar_t* value,
    scalar_t* out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale) {
  // TODO_BEGIN(W09_T03)
  const int batch = blockIdx.x;
  const int head = blockIdx.y;
  const int dim = threadIdx.x;
  if (batch < batch_size && head < num_heads && dim < head_dim) {
    out[(batch * num_heads + head) * head_dim + dim] = from_float<scalar_t>(0.0f);
  }
  // TODO_END(W09_T03)
}

template <typename scalar_t>
int launch_softmax(const void* x, void* out, int rows, int cols) {
  softmax_kernel<<<rows, kThreads>>>(static_cast<const scalar_t*>(x), static_cast<scalar_t*>(out), rows, cols);
  return static_cast<int>(cudaGetLastError());
}

template <typename scalar_t>
int launch_rms_norm(const void* x, const void* weight, void* out, int rows, int cols, float eps) {
  rms_norm_kernel<<<rows, kThreads>>>(static_cast<const scalar_t*>(x), static_cast<const scalar_t*>(weight), static_cast<scalar_t*>(out), rows, cols, eps);
  return static_cast<int>(cudaGetLastError());
}

template <typename scalar_t>
int launch_rope(const void* x, const void* cos, const void* sin, void* out, int rows, int cols) {
  rope_kernel<<<rows, kThreads>>>(
      static_cast<const scalar_t*>(x), static_cast<const scalar_t*>(cos), static_cast<const scalar_t*>(sin), static_cast<scalar_t*>(out), rows, cols);
  return static_cast<int>(cudaGetLastError());
}

template <typename scalar_t>
int launch_matmul(const void* a, const void* b, void* out, int m, int n, int k) {
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  matmul_kernel<<<grid, block>>>(static_cast<const scalar_t*>(a), static_cast<const scalar_t*>(b), static_cast<scalar_t*>(out), m, n, k);
  return static_cast<int>(cudaGetLastError());
}

template <typename scalar_t>
int launch_matmul_tiled(const void* a, const void* b, void* out, int m, int n, int k) {
  constexpr int kTile = 16;
  dim3 block(kTile, kTile);
  dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
  matmul_tiled_kernel<scalar_t, kTile><<<grid, block>>>(
      static_cast<const scalar_t*>(a), static_cast<const scalar_t*>(b), static_cast<scalar_t*>(out), m, n, k);
  return static_cast<int>(cudaGetLastError());
}

template <typename scalar_t>
int launch_paged_attention_decode(
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const int64_t* block_tables,
    const int64_t* context_lens,
    void* out,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks,
    int block_size,
    float scale) {
  dim3 grid(batch_size, num_heads);
  paged_attention_decode_kernel<<<grid, kThreads>>>(
      static_cast<const scalar_t*>(query),
      static_cast<const scalar_t*>(key_cache),
      static_cast<const scalar_t*>(value_cache),
      block_tables,
      context_lens,
      static_cast<scalar_t*>(out),
      batch_size,
      num_heads,
      num_kv_heads,
      head_dim,
      max_blocks,
      block_size,
      scale);
  return static_cast<int>(cudaGetLastError());
}

template <typename scalar_t>
int launch_dense_attention_prefill(
    const void* query,
    const void* key,
    const void* value,
    void* out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale) {
  dim3 grid(batch_size, num_heads, seq_len);
  dense_attention_prefill_kernel<<<grid, kThreads>>>(
      static_cast<const scalar_t*>(query),
      static_cast<const scalar_t*>(key),
      static_cast<const scalar_t*>(value),
      static_cast<scalar_t*>(out),
      batch_size,
      num_heads,
      seq_len,
      head_dim,
      scale);
  return static_cast<int>(cudaGetLastError());
}

template <typename scalar_t>
int launch_dense_attention_decode(
    const void* query,
    const void* key,
    const void* value,
    void* out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale) {
  dim3 grid(batch_size, num_heads);
  dense_attention_decode_kernel<<<grid, kThreads>>>(
      static_cast<const scalar_t*>(query),
      static_cast<const scalar_t*>(key),
      static_cast<const scalar_t*>(value),
      static_cast<scalar_t*>(out),
      batch_size,
      num_heads,
      seq_len,
      head_dim,
      scale);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" int softmax_cuda_launcher(const void* x, void* out, int rows, int cols, int dtype) {
  if (dtype == 0) return launch_softmax<float>(x, out, rows, cols);
  if (dtype == 1) return launch_softmax<__half>(x, out, rows, cols);
  return launch_softmax<__nv_bfloat16>(x, out, rows, cols);
}

extern "C" int rms_norm_cuda_launcher(const void* x, const void* weight, void* out, int rows, int cols, float eps, int dtype) {
  if (dtype == 0) return launch_rms_norm<float>(x, weight, out, rows, cols, eps);
  if (dtype == 1) return launch_rms_norm<__half>(x, weight, out, rows, cols, eps);
  return launch_rms_norm<__nv_bfloat16>(x, weight, out, rows, cols, eps);
}

extern "C" int rope_cuda_launcher(const void* x, const void* cos, const void* sin, void* out, int rows, int cols, int dtype) {
  if (dtype == 0) return launch_rope<float>(x, cos, sin, out, rows, cols);
  if (dtype == 1) return launch_rope<__half>(x, cos, sin, out, rows, cols);
  return launch_rope<__nv_bfloat16>(x, cos, sin, out, rows, cols);
}

extern "C" int matmul_cuda_launcher(const void* a, const void* b, void* out, int m, int n, int k, int dtype) {
  if (dtype == 0) return launch_matmul<float>(a, b, out, m, n, k);
  if (dtype == 1) return launch_matmul<__half>(a, b, out, m, n, k);
  return launch_matmul<__nv_bfloat16>(a, b, out, m, n, k);
}

extern "C" int matmul_tiled_cuda_launcher(const void* a, const void* b, void* out, int m, int n, int k, int dtype) {
  if (dtype == 0) return launch_matmul_tiled<float>(a, b, out, m, n, k);
  if (dtype == 1) return launch_matmul_tiled<__half>(a, b, out, m, n, k);
  return launch_matmul_tiled<__nv_bfloat16>(a, b, out, m, n, k);
}

extern "C" int paged_attention_decode_cuda_launcher(
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const int64_t* block_tables,
    const int64_t* context_lens,
    void* out,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks,
    int block_size,
    float scale,
    int dtype) {
  if (dtype == 0) {
    return launch_paged_attention_decode<float>(
        query, key_cache, value_cache, block_tables, context_lens, out, batch_size, num_heads, num_kv_heads, head_dim, max_blocks, block_size, scale);
  }
  if (dtype == 1) {
    return launch_paged_attention_decode<__half>(
        query, key_cache, value_cache, block_tables, context_lens, out, batch_size, num_heads, num_kv_heads, head_dim, max_blocks, block_size, scale);
  }
  return launch_paged_attention_decode<__nv_bfloat16>(
      query, key_cache, value_cache, block_tables, context_lens, out, batch_size, num_heads, num_kv_heads, head_dim, max_blocks, block_size, scale);
}

extern "C" int dense_attention_prefill_cuda_launcher(
    const void* query,
    const void* key,
    const void* value,
    void* out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    int dtype) {
  if (dtype == 0) {
    return launch_dense_attention_prefill<float>(query, key, value, out, batch_size, num_heads, seq_len, head_dim, scale);
  }
  if (dtype == 1) {
    return launch_dense_attention_prefill<__half>(query, key, value, out, batch_size, num_heads, seq_len, head_dim, scale);
  }
  return launch_dense_attention_prefill<__nv_bfloat16>(query, key, value, out, batch_size, num_heads, seq_len, head_dim, scale);
}

extern "C" int dense_attention_decode_cuda_launcher(
    const void* query,
    const void* key,
    const void* value,
    void* out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    int dtype) {
  if (dtype == 0) {
    return launch_dense_attention_decode<float>(query, key, value, out, batch_size, num_heads, seq_len, head_dim, scale);
  }
  if (dtype == 1) {
    return launch_dense_attention_decode<__half>(query, key, value, out, batch_size, num_heads, seq_len, head_dim, scale);
  }
  return launch_dense_attention_decode<__nv_bfloat16>(query, key, value, out, batch_size, num_heads, seq_len, head_dim, scale);
}
