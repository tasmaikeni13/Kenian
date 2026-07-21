// Custom CUDA kernels for the Kenian elementwise hot path.
// Mirrors kenian_backends.TorchBackend exactly; verified in tests/test_kernels.py.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONT(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK(x) CHECK_CUDA(x); CHECK_CONT(x)

constexpr int THREADS = 256;

template <typename T> __device__ __forceinline__ T ksqrt(T x);
template <> __device__ __forceinline__ float ksqrt<float>(float x) { return sqrtf(x); }
template <> __device__ __forceinline__ double ksqrt<double>(double x) { return sqrt(x); }

// ---- phase A: AdamW moments (in place) + base step and denominator -----------
// Arithmetic in scalar_t (matches torch's dtype-native ops and keeps fp32 off the
// slow Turing fp64 units); scalar hyperparameters are cast to scalar_t once.
template <typename scalar_t>
__global__ void phase_a_adamw_kernel(
    const scalar_t* __restrict__ g, scalar_t* __restrict__ m, scalar_t* __restrict__ v,
    scalar_t* __restrict__ base, scalar_t* __restrict__ denom, long n,
    double b1_, double b2_, double eps_, double inv_sqrt_bc2_, double neg_lr_over_bc1_) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  scalar_t b1 = (scalar_t)b1_, b2 = (scalar_t)b2_, eps = (scalar_t)eps_;
  scalar_t inv_sqrt_bc2 = (scalar_t)inv_sqrt_bc2_, neg_lr_over_bc1 = (scalar_t)neg_lr_over_bc1_;
  scalar_t one = (scalar_t)1;
  scalar_t gi = g[i], mi = m[i], vi = v[i];
  mi = b1 * mi + (one - b1) * gi;
  vi = b2 * vi + (one - b2) * gi * gi;
  scalar_t d = ksqrt<scalar_t>(vi) * inv_sqrt_bc2 + eps;
  m[i] = mi;
  v[i] = vi;
  denom[i] = d;
  base[i] = neg_lr_over_bc1 * mi / d;
}

// ---- partial norms: base^2 * denom, base^2, slice^2 / (4 * denom) -----------
template <typename scalar_t>
__global__ void partial_norms_kernel(
    const scalar_t* __restrict__ base, const scalar_t* __restrict__ denom,
    const scalar_t* __restrict__ kappa, double* __restrict__ out, long n, bool has_k) {
  __shared__ double sh0[THREADS];
  __shared__ double sh1[THREADS];
  __shared__ double sh2[THREADS];
  int tid = threadIdx.x;
  long i = (long)blockIdx.x * blockDim.x + tid;
  double pP = 0.0, pEu = 0.0, pCorr = 0.0;
  if (i < n) {
    double b = (double)base[i], d = (double)denom[i];
    pEu = b * b;
    pP = pEu * d;
    if (has_k) { double k = (double)kappa[i]; pCorr = 0.25 * k * k / d; }
  }
  sh0[tid] = pP; sh1[tid] = pEu; sh2[tid] = pCorr;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) { sh0[tid] += sh0[tid + s]; sh1[tid] += sh1[tid + s]; sh2[tid] += sh2[tid + s]; }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(&out[0], sh0[0]);
    atomicAdd(&out[1], sh1[0]);
    atomicAdd(&out[2], sh2[0]);
  }
}

// ---- phase B: apply decoupled decay and the corrected update ----------------
template <typename scalar_t>
__global__ void phase_b_kernel(
    scalar_t* __restrict__ p, const scalar_t* __restrict__ base,
    const scalar_t* __restrict__ denom, const scalar_t* __restrict__ kappa,
    scalar_t* __restrict__ prev, long n, double coeff, double one_minus_lrwd, bool has_k) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  scalar_t upd = base[i];
  if (has_k) upd = base[i] - (scalar_t)coeff * kappa[i] / denom[i];
  p[i] = p[i] * (scalar_t)one_minus_lrwd + upd;
  prev[i] = upd;
}

// ---- launchers -------------------------------------------------------------
std::vector<torch::Tensor> phase_a_adamw(
    torch::Tensor g, torch::Tensor m, torch::Tensor v,
    double lr, double b1, double b2, double eps, double bc1, double bc2) {
  CHECK(g); CHECK(m); CHECK(v);
  g = g.reshape(-1); m = m.reshape(-1); v = v.reshape(-1);
  long n = g.numel();
  auto base = torch::empty_like(g);
  auto denom = torch::empty_like(g);
  const at::cuda::OptionalCUDAGuard guard(device_of(g));
  int blocks = (n + THREADS - 1) / THREADS;
  double inv_sqrt_bc2 = 1.0 / std::sqrt(bc2);
  double neg_lr_over_bc1 = -lr / bc1;
  AT_DISPATCH_FLOATING_TYPES(g.scalar_type(), "phase_a_adamw", [&] {
    phase_a_adamw_kernel<scalar_t><<<blocks, THREADS>>>(
        g.data_ptr<scalar_t>(), m.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
        base.data_ptr<scalar_t>(), denom.data_ptr<scalar_t>(), n,
        b1, b2, eps, inv_sqrt_bc2, neg_lr_over_bc1);
  });
  return {base, denom};
}

torch::Tensor partial_norms(
    torch::Tensor base, torch::Tensor denom, c10::optional<torch::Tensor> kappa) {
  CHECK(base); CHECK(denom);
  base = base.reshape(-1); denom = denom.reshape(-1);
  long n = base.numel();
  bool has_k = kappa.has_value();
  torch::Tensor k = has_k ? kappa.value().reshape(-1) : base;
  if (has_k) { CHECK(k); }
  auto out = torch::zeros({3}, base.options().dtype(torch::kFloat64));
  const at::cuda::OptionalCUDAGuard guard(device_of(base));
  int blocks = (n + THREADS - 1) / THREADS;
  AT_DISPATCH_FLOATING_TYPES(base.scalar_type(), "partial_norms", [&] {
    partial_norms_kernel<scalar_t><<<blocks, THREADS>>>(
        base.data_ptr<scalar_t>(), denom.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
        out.data_ptr<double>(), n, has_k);
  });
  return out;
}

torch::Tensor phase_b(
    torch::Tensor p, torch::Tensor base, torch::Tensor denom,
    c10::optional<torch::Tensor> kappa, double coeff, double lr, double wd) {
  CHECK(p); CHECK(base); CHECK(denom);
  auto pf = p.reshape(-1); base = base.reshape(-1); denom = denom.reshape(-1);
  long n = pf.numel();
  bool has_k = kappa.has_value();
  torch::Tensor k = has_k ? kappa.value().reshape(-1) : base;
  if (has_k) { CHECK(k); }
  auto prev = torch::empty_like(base);
  const at::cuda::OptionalCUDAGuard guard(device_of(p));
  int blocks = (n + THREADS - 1) / THREADS;
  double one_minus_lrwd = 1.0 - lr * wd;
  AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "phase_b", [&] {
    phase_b_kernel<scalar_t><<<blocks, THREADS>>>(
        pf.data_ptr<scalar_t>(), base.data_ptr<scalar_t>(), denom.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(), prev.data_ptr<scalar_t>(), n, coeff, one_minus_lrwd, has_k);
  });
  return prev.reshape(p.sizes());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("phase_a_adamw", &phase_a_adamw, "Kenian AdamW phase");
  mod.def("partial_norms", &partial_norms, "Kenian partial norms");
  mod.def("phase_b", &phase_b, "Kenian phase B");
}
