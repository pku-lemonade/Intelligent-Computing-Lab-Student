#include <torch/extension.h>
#include <cuda_runtime_api.h>

extern "C" int softmax_cuda_launcher(const void* x, void* out, int rows, int cols, int dtype);
extern "C" int rms_norm_cuda_launcher(const void* x, const void* weight, void* out, int rows, int cols, float eps, int dtype);
extern "C" int rope_cuda_launcher(const void* x, const void* cos, const void* sin, void* out, int rows, int cols, int dtype);
extern "C" int matmul_cuda_launcher(const void* a, const void* b, void* out, int m, int n, int k, int dtype);
extern "C" int matmul_tiled_cuda_launcher(const void* a, const void* b, void* out, int m, int n, int k, int dtype);
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
    int dtype);
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
    int dtype);
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
    int dtype);

namespace {

void check_cuda(torch::Tensor tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.is_floating_point(), name, " must be floating point");
}

int dtype_code(torch::Tensor tensor) {
  if (tensor.scalar_type() == torch::kFloat32) return 0;
  if (tensor.scalar_type() == torch::kFloat16) return 1;
  if (tensor.scalar_type() == torch::kBFloat16) return 2;
  TORCH_CHECK(false, "only float32, float16, and bfloat16 are supported");
}

void check_launch(int error_code) {
  TORCH_CHECK(error_code == 0, "CUDA kernel launch failed: ", cudaGetErrorString(static_cast<cudaError_t>(error_code)));
}

}  // namespace

torch::Tensor softmax(torch::Tensor x) {
  check_cuda(x, "x");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  auto out = torch::empty_like(x);
  check_launch(softmax_cuda_launcher(x.data_ptr(), out.data_ptr(), x.size(0), x.size(1), dtype_code(x)));
  return out;
}

torch::Tensor rms_norm(torch::Tensor x, torch::Tensor weight, double eps) {
  check_cuda(x, "x");
  check_cuda(weight, "weight");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1), "weight must have shape [hidden_size]");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight dtype must match x dtype");
  auto out = torch::empty_like(x);
  check_launch(rms_norm_cuda_launcher(x.data_ptr(), weight.data_ptr(), out.data_ptr(), x.size(0), x.size(1), static_cast<float>(eps), dtype_code(x)));
  return out;
}

torch::Tensor rope(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
  check_cuda(x, "x");
  check_cuda(cos, "cos");
  check_cuda(sin, "sin");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(cos.sizes() == x.sizes() && sin.sizes() == x.sizes(), "cos/sin must match x shape");
  TORCH_CHECK(cos.scalar_type() == x.scalar_type() && sin.scalar_type() == x.scalar_type(), "cos/sin dtype must match x dtype");
  TORCH_CHECK(x.size(1) % 2 == 0, "RoPE head dimension must be even");
  auto out = torch::empty_like(x);
  check_launch(rope_cuda_launcher(x.data_ptr(), cos.data_ptr(), sin.data_ptr(), out.data_ptr(), x.size(0), x.size(1), dtype_code(x)));
  return out;
}

torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
  check_cuda(a, "a");
  check_cuda(b, "b");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "matmul inputs must be 2D");
  TORCH_CHECK(a.size(1) == b.size(0), "matmul shape mismatch");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "matmul input dtypes must match");
  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  check_launch(matmul_cuda_launcher(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.size(0), b.size(1), a.size(1), dtype_code(a)));
  return out;
}

torch::Tensor matmul_tiled(torch::Tensor a, torch::Tensor b) {
  check_cuda(a, "a");
  check_cuda(b, "b");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "matmul inputs must be 2D");
  TORCH_CHECK(a.size(1) == b.size(0), "matmul shape mismatch");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "matmul input dtypes must match");
  auto out = torch::empty({a.size(0), b.size(1)}, a.options());
  check_launch(matmul_tiled_cuda_launcher(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.size(0), b.size(1), a.size(1), dtype_code(a)));
  return out;
}

torch::Tensor paged_attention_decode(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int64_t block_size,
    double scale) {
  check_cuda(query, "query");
  check_cuda(key_cache, "key_cache");
  check_cuda(value_cache, "value_cache");
  TORCH_CHECK(block_tables.is_cuda() && block_tables.dtype() == torch::kInt64 && block_tables.is_contiguous(), "block_tables must be contiguous CUDA int64");
  TORCH_CHECK(context_lens.is_cuda() && context_lens.dtype() == torch::kInt64 && context_lens.is_contiguous(), "context_lens must be contiguous CUDA int64");
  TORCH_CHECK(query.dim() == 3 && key_cache.dim() == 3 && value_cache.sizes() == key_cache.sizes(), "bad paged attention shapes");
  TORCH_CHECK(query.size(2) == key_cache.size(2), "query and KV head_dim must match");
  TORCH_CHECK(query.size(1) % key_cache.size(1) == 0, "query heads must be divisible by KV heads");
  TORCH_CHECK(query.scalar_type() == key_cache.scalar_type() && query.scalar_type() == value_cache.scalar_type(), "query/KV dtypes must match");
  TORCH_CHECK(query.size(2) <= 256, "head_dim must be <= 256 for this teaching kernel");
  auto out = torch::empty_like(query);
  check_launch(paged_attention_decode_cuda_launcher(
      query.data_ptr(),
      key_cache.data_ptr(),
      value_cache.data_ptr(),
      block_tables.data_ptr<int64_t>(),
      context_lens.data_ptr<int64_t>(),
      out.data_ptr(),
      query.size(0),
      query.size(1),
      key_cache.size(1),
      query.size(2),
      block_tables.size(1),
      static_cast<int>(block_size),
      static_cast<float>(scale),
      dtype_code(query)));
  return out;
}

torch::Tensor dense_attention_prefill(torch::Tensor query, torch::Tensor key, torch::Tensor value, double scale) {
  check_cuda(query, "query");
  check_cuda(key, "key");
  check_cuda(value, "value");
  TORCH_CHECK(query.dim() == 4 && key.sizes() == query.sizes() && value.sizes() == query.sizes(), "expected Q/K/V shape [batch, heads, seq, dim]");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() && query.scalar_type() == value.scalar_type(), "Q/K/V dtypes must match");
  TORCH_CHECK(query.size(3) <= 256, "head_dim must be <= 256 for this teaching kernel");
  auto out = torch::empty_like(query);
  check_launch(dense_attention_prefill_cuda_launcher(
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      out.data_ptr(),
      query.size(0),
      query.size(1),
      query.size(2),
      query.size(3),
      static_cast<float>(scale),
      dtype_code(query)));
  return out;
}

torch::Tensor dense_attention_decode(torch::Tensor query, torch::Tensor key, torch::Tensor value, double scale) {
  check_cuda(query, "query");
  check_cuda(key, "key");
  check_cuda(value, "value");
  TORCH_CHECK(query.dim() == 3, "expected query shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4 && value.sizes() == key.sizes(), "expected K/V shape [batch, heads, seq, dim]");
  TORCH_CHECK(query.size(0) == key.size(0) && query.size(1) == key.size(1) && query.size(2) == key.size(3), "query and K/V shapes must align");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() && query.scalar_type() == value.scalar_type(), "Q/K/V dtypes must match");
  TORCH_CHECK(query.size(2) <= 256, "head_dim must be <= 256 for this teaching kernel");
  auto out = torch::empty_like(query);
  check_launch(dense_attention_decode_cuda_launcher(
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      out.data_ptr(),
      query.size(0),
      query.size(1),
      key.size(2),
      query.size(2),
      static_cast<float>(scale),
      dtype_code(query)));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax", &softmax, "row-wise softmax");
  m.def("rms_norm", &rms_norm, "RMSNorm");
  m.def("rope", &rope, "Qwen-style RoPE");
  m.def("matmul", &matmul, "naive matrix multiplication");
  m.def("matmul_tiled", &matmul_tiled, "tiled matrix multiplication");
  m.def("paged_attention_decode", &paged_attention_decode, "paged attention decode");
  m.def("dense_attention_prefill", &dense_attention_prefill, "dense prefill attention with online softmax");
  m.def("dense_attention_decode", &dense_attention_decode, "dense decode attention with online softmax");
}
