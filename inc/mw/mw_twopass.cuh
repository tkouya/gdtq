/*
 * mw_twopass.cuh -- two-pass evaluation of the generic multi-word functions
 *                   on the GPU (fast pass + compaction + second pass)
 *
 * TWO_PASS: pass 1 evaluates every input that belongs to the function's
 * main class (the branch taken by "regular" inputs) and appends the indices
 * of all other inputs to a list (warp-aggregated atomics); pass 2 evaluates
 * the listed inputs with the complete function.  A rare expensive branch
 * (Payne-Hanek reduction, done in FP64 even for the float classes) then no
 * longer makes whole warps execute it.
 * PARTITION: a classification kernel compacts both classes (main class to
 * the front of the list, the rest to the back) and one evaluation kernel
 * runs over the list, so that functions with two common algorithms
 * (sinh/cosh/tanh/expm1/asinh for |x| <= 1 and > 1, acosh) see
 * single-class warps.
 * The results are bitwise identical to the single-pass functions: both
 * passes call the same code (the main class without the Payne-Hanek branch).
 * The list lengths stay on the device, so no host round trip is needed.
 */
#ifndef MW_TWOPASS_CUH
#define MW_TWOPASS_CUH

#include "mw_elem.h"

namespace mw2p {
using mw::mwn;

enum Op { EXP, EXPM1, LOG, LOG10, LOG1P, SIN, COS, TAN, ASIN, ACOS, ATAN, SINH, COSH, TANH,
          ASINH, ACOSH, ATANH, SQRT, ATAN2, POW, NOPS };

/* complete (single-pass) function; PH = false drops the Payne-Hanek branch */
template <int OP, class S, class T, bool PH = true>
__device__ __forceinline__ mwn<T, S::K> full(const mwn<T, S::K> &a, const mwn<T, S::K> &b) {
  constexpr int K = S::K;
  if constexpr (OP == EXP) return mw::exp<S, T>(a);
  else if constexpr (OP == EXPM1) return mw::expm1<S, T>(a);
  else if constexpr (OP == LOG) return mw::log<S, T>(a);
  else if constexpr (OP == LOG10) return mw::log10<S, T>(a);
  else if constexpr (OP == LOG1P) return mw::log1p<S, T>(a);
  else if constexpr (OP == SIN) return mw::sin<S, T, K, PH>(a);
  else if constexpr (OP == COS) return mw::cos<S, T, K, PH>(a);
  else if constexpr (OP == TAN) return mw::tan<S, T, K, PH>(a);
  else if constexpr (OP == ASIN) return mw::asin<S, T>(a);
  else if constexpr (OP == ACOS) return mw::acos<S, T>(a);
  else if constexpr (OP == ATAN) return mw::atan<S, T>(a);
  else if constexpr (OP == SINH) return mw::sinh<S, T>(a);
  else if constexpr (OP == COSH) return mw::cosh<S, T>(a);
  else if constexpr (OP == TANH) return mw::tanh<S, T>(a);
  else if constexpr (OP == ASINH) return mw::asinh<S, T>(a);
  else if constexpr (OP == ACOSH) return mw::acosh<S, T>(a);
  else if constexpr (OP == ATANH) return mw::atanh<S, T>(a);
  else if constexpr (OP == SQRT) return mw::sqrt<K>(a);
  else if constexpr (OP == ATAN2) return mw::atan2<S, T>(a, b);
  else return mw::pow<S, T>(a, b);
}

/* main class of each function (evaluated by pass 1).  Everything else --
   special values, huge arguments, the other algorithm of a two-branch
   function -- goes to pass 2.  NaN fails every comparison, so it always
   goes to pass 2.                                                         */
template <int OP, class T, int K>
__device__ __forceinline__ bool main_class(const mwn<T, K> &a, const mwn<T, K> &b) {
  typedef mw::Lim<T> L;
  const T x = a.v[0], ax = mw::abs_(x);
  if constexpr (OP == EXP) return ax < L::EXP_FAST;
  else if constexpr (OP == EXPM1) return ax < T(1);
  else if constexpr (OP == LOG || OP == LOG10)
    return x >= mw::pow2i<T>(mw::FP<T>::EMIN + 2) && x < mw::pow2i<T>(mw::FP<T>::EMAX - 2);
  else if constexpr (OP == LOG1P) return x > T(-0.5) && x < L::BIG;
  else if constexpr (OP == SIN || OP == COS || OP == TAN) return ax < L::SC_FAST;
  else if constexpr (OP == ASIN || OP == ACOS) return ax < T(1);
  else if constexpr (OP == ATAN) return ax <= L::BIG;
  else if constexpr (OP == SINH || OP == COSH || OP == TANH) return ax <= T(1);
  else if constexpr (OP == ASINH) return ax <= T(1);
  else if constexpr (OP == ACOSH) return x >= T(1) && x < T(2);
  else if constexpr (OP == ATANH) return ax < T(1);
  else if constexpr (OP == SQRT) return x > T(0) && x <= L::HUGE_;
  else if constexpr (OP == ATAN2) {
    const T y = b.v[0];
    const T m = ax > mw::abs_(y) ? ax : mw::abs_(y);
    return m <= L::BIG && m >= T(1) / L::BIG;
  } else {                                     /* pow: a > 0 normal, b finite, no overflow of b log a */
    const T bb = mw::abs_(b.v[0]);
    int e;
    mw::frexp_(x, &e);
    const T la = T(e < 0 ? 1 - e : e + 1) * T(0.6932);        /* >= |log a| */
    return x >= mw::pow2i<T>(mw::FP<T>::EMIN + 2) && x < mw::pow2i<T>(mw::FP<T>::EMAX - 2) &&
           bb * la < L::EXP_FAST * T(0.9);
  }
}

/* ---------------------------------------------------------------- kernels */
/* RT > 1 repeats the evaluation RT times with a dependency (x += 0 f(x)),
   only for throughput measurements; the library use is RT = 1.            */
template <int OP, class S, class T, int RT>
__global__ void single_pass(const mwn<T, S::K> *a, const mwn<T, S::K> *b, mwn<T, S::K> *y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  mwn<T, S::K> x = a[i], r;
  const mwn<T, S::K> bb = b[i];
  for (int t = 0; t < RT; t++) {
    r = full<OP, S, T>(x, bb);
    x.v[0] += T(0) * r.v[0];
  }
  y[i] = r;
}

template <int OP, class S, class T, int RT>
__global__ void fast_pass(const mwn<T, S::K> *a, const mwn<T, S::K> *b, mwn<T, S::K> *y, int n,
                          int *list, int *count, int *next) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) { next[0] = 0; next[1] = 0; }         /* counters of the next call */
  bool hard = false;
  if (i < n) {
    mwn<T, S::K> x = a[i], r;
    const mwn<T, S::K> bb = b[i];
    hard = !main_class<OP>(x, bb);
    if (!hard) {
      for (int t = 0; t < RT; t++) {
        r = full<OP, S, T, false>(x, bb);
        x.v[0] += T(0) * r.v[0];
      }
      y[i] = r;
    }
  }
  /* warp-aggregated append of the hard indices */
  const unsigned mask = __ballot_sync(0xffffffffu, hard);
  if (mask) {
    const int lane = threadIdx.x & 31, leader = __ffs(mask) - 1;
    int base = 0;
    if (lane == leader) base = atomicAdd(&count[1], __popc(mask));
    base = __shfl_sync(0xffffffffu, base, leader);
    if (hard) list[base + __popc(mask & ((1u << lane) - 1u))] = i;
  }
}

template <int OP, class S, class T, int RT>
__global__ void second_pass(const mwn<T, S::K> *a, const mwn<T, S::K> *b, mwn<T, S::K> *y,
                            const int *list, const int *count) {
  const int m = count[1];
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < m; t += gridDim.x * blockDim.x) {
    const int i = list[t];
    mwn<T, S::K> x = a[i], r;
    const mwn<T, S::K> bb = b[i];
    for (int u = 0; u < RT; u++) {
      r = full<OP, S, T>(x, bb);
      x.v[0] += T(0) * r.v[0];
    }
    y[i] = r;
  }
}

/* partitioned evaluation (for functions whose two classes are both common):
   classify writes the main-class indices to the front of the list and the
   others to the back; evaluate then runs over the list, so every warp
   except the one at the boundary sees a single class                     */
template <int OP, class S, class T>
__global__ void classify(const mwn<T, S::K> *a, const mwn<T, S::K> *b, int n, int *list, int *count, int *next) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) { next[0] = 0; next[1] = 0; }
  const bool in = i < n;
  const bool m = in && main_class<OP>(a[i], b[i]);
  const bool o = in && !m;
  const unsigned mm = __ballot_sync(0xffffffffu, m), mo = __ballot_sync(0xffffffffu, o);
  const int lane = threadIdx.x & 31;
  const unsigned below = (1u << lane) - 1u;
  int bm = 0, bo = 0;
  if (lane == 0) {
    if (mm) bm = atomicAdd(&count[0], __popc(mm));
    if (mo) bo = atomicAdd(&count[1], __popc(mo));
  }
  bm = __shfl_sync(0xffffffffu, bm, 0);
  bo = __shfl_sync(0xffffffffu, bo, 0);
  if (m) list[bm + __popc(mm & below)] = i;
  if (o) list[n - 1 - (bo + __popc(mo & below))] = i;
}

template <int OP, class S, class T, int RT>
__global__ void evaluate_partitioned(const mwn<T, S::K> *a, const mwn<T, S::K> *b, mwn<T, S::K> *y, int n,
                                     const int *list, const int *count) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  const int i = list[t];
  mwn<T, S::K> x = a[i], r;
  const mwn<T, S::K> bb = b[i];
  if (t < count[0]) {
    for (int u = 0; u < RT; u++) { r = full<OP, S, T, false>(x, bb); x.v[0] += T(0) * r.v[0]; }
  } else {
    for (int u = 0; u < RT; u++) { r = full<OP, S, T>(x, bb); x.v[0] += T(0) * r.v[0]; }
  }
  y[i] = r;
}

/* ------------------------------------------------------------ host side */
/* Two counter slots are used alternately: the first kernel of a call
   zeroes the slot of the next call (the previous call on the same stream
   has finished by then), so no memset launch is needed.  One Work object
   per stream.                                                            */
struct Work {
  int *list = nullptr, *count = nullptr;   /* count: 2 slots x [main, other] */
  int cap = 0, sms = 0, slot = 0;
  void reserve(int n) {
    if (n <= cap) return;
    if (list) cudaFree(list);
    if (!count) { cudaMalloc(&count, 4 * sizeof(int)); cudaMemset(count, 0, 4 * sizeof(int)); }
    cudaMalloc(&list, sizeof(int) * n);
    cap = n;
  }
  int *cur() { return count + 2 * slot; }
  int *nxt() { return count + 2 * (1 - slot); }
  ~Work() { if (list) cudaFree(list); if (count) cudaFree(count); }
};

enum Mode { SINGLE, TWO_PASS, PARTITION };

/* suggested mode (GB10 and H100 measurements, see the report):
   - sin/cos/tan: TWO_PASS (a few huge arguments take the Payne-Hanek path,
     3-500x more expensive than the fast path; cost <= 5 % without them);
   - functions with two common algorithm classes: PARTITION when one
     evaluation is expensive compared with its memory traffic -- double
     words on GPUs with weak FP64 (GB10, FP32:FP64 = 64:1: 1.9x) and QD
     everywhere (H100: 1.5x); SINGLE otherwise (the extra pass costs more:
     DD/DS on H100 0.4-0.6x);
   - the rest: SINGLE (their rare branches are cheap special values).
   fp32_per_fp64 = cudaDevAttrSingleToDoublePrecisionPerfRatio.          */
template <int OP, class S, class T>
Mode default_mode(int fp32_per_fp64) {
  if (OP == SIN || OP == COS || OP == TAN) return TWO_PASS;
  const bool two_alg = (OP == SINH || OP == COSH || OP == TANH || OP == EXPM1 || OP == ASINH || OP == ACOSH);
  if (two_alg && sizeof(T) == 8 && (S::K >= 4 || fp32_per_fp64 >= 8)) return PARTITION;
  return SINGLE;
}
template <int OP, class S, class T>
Mode default_mode() {
  int dev = 0, r = 2;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&r, cudaDevAttrSingleToDoublePrecisionPerfRatio, dev);
  return default_mode<OP, S, T>(r);
}

/* y[i] = f(a[i] [, b[i]]), i < n.  TWO_PASS: the main class is evaluated in
   place and only the other inputs are compacted (rare hard cases);
   PARTITION: both classes are compacted (two common classes).            */
template <int OP, class S, class T, int RT = 1>
void eval(Mode mode, const mwn<T, S::K> *a, const mwn<T, S::K> *b, mwn<T, S::K> *y, int n, Work &w,
          cudaStream_t st = 0, int block = 128) {
  const int grid = (n + block - 1) / block;
  if (!b) b = a;
  if (mode == SINGLE) {
    single_pass<OP, S, T, RT><<<grid, block, 0, st>>>(a, b, y, n);
    return;
  }
  w.reserve(n);
  if (mode == TWO_PASS) {
    /* the second pass is grid-stride with at most 16 blocks per SM (enough
       for full occupancy; fewer empty blocks when the list is short)    */
    if (w.sms == 0) {
      int dev;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&w.sms, cudaDevAttrMultiProcessorCount, dev);
    }
    const int grid2 = grid < 16 * w.sms ? grid : 16 * w.sms;
    fast_pass<OP, S, T, RT><<<grid, block, 0, st>>>(a, b, y, n, w.list, w.cur(), w.nxt());
    second_pass<OP, S, T, RT><<<grid2, block, 0, st>>>(a, b, y, w.list, w.cur());
  } else {
    classify<OP, S, T><<<grid, block, 0, st>>>(a, b, n, w.list, w.cur(), w.nxt());
    evaluate_partitioned<OP, S, T, RT><<<grid, block, 0, st>>>(a, b, y, n, w.list, w.cur());
  }
  w.slot = 1 - w.slot;
}

} // namespace mw2p

#endif
