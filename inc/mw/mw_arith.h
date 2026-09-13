/*
 * mw_arith.h -- generic multi-word (floating-point expansion) arithmetic
 *               for T = double / float and K = 1..5 words.
 *
 * Header-only, usable from host C++ (g++ -ffp-contract=off) and from CUDA
 * device code (nvcc -fmad=false).  Every FMA is explicit.
 *
 * Numbers are unevaluated sums x = v[0] + v[1] + ... + v[K-1] with
 * |v[i+1]| <~ ulp(v[i]).  Products and sums are formed with error-free
 * transformations; the partial terms are grouped by "order" (a term of
 * order n is ~u^n times the leading term) and each order is accumulated
 * with a TwoSum chain whose errors are pushed to the next order, the last
 * order being summed plainly.  All loops have compile-time trip counts and
 * are fully unrolled, so the level buffers live in registers.
 */
#ifndef MW_ARITH_H
#define MW_ARITH_H

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define MW_HD __host__ __device__
#define MW_INL __forceinline__
#define MW_PRAGMA_UNROLL _Pragma("unroll")
#else
#define MW_HD
#define MW_INL inline __attribute__((always_inline))
#define MW_PRAGMA_UNROLL _Pragma("GCC unroll 64")
#endif
#define MWF MW_HD MW_INL

namespace mw {

/* ------------------------------------------------------------------ */
/* scalar primitives                                                   */
/* ------------------------------------------------------------------ */
MWF double fma_(double a, double b, double c) {
#ifdef __CUDA_ARCH__
  return __fma_rn(a, b, c);
#else
  return __builtin_fma(a, b, c);
#endif
}
MWF float fma_(float a, float b, float c) {
#ifdef __CUDA_ARCH__
  return __fmaf_rn(a, b, c);
#else
  return __builtin_fmaf(a, b, c);
#endif
}
MWF double sqrt_(double a) {
#ifdef __CUDA_ARCH__
  return ::sqrt(a);
#else
  return __builtin_sqrt(a);
#endif
}
MWF float sqrt_(float a) {
#ifdef __CUDA_ARCH__
  return ::sqrtf(a);
#else
  return __builtin_sqrtf(a);
#endif
}
MWF double rint_(double a) {
#ifdef __CUDA_ARCH__
  return rint(a);
#else
  return __builtin_rint(a);
#endif
}
MWF float rint_(float a) {
#ifdef __CUDA_ARCH__
  return rintf(a);
#else
  return __builtin_rintf(a);
#endif
}

/* libm pieces used on rare paths (host: builtins, device: CUDA math) */
MWF double floor_(double a) {
#ifdef __CUDA_ARCH__
  return ::floor(a);
#else
  return __builtin_floor(a);
#endif
}
MWF float floor_(float a) {
#ifdef __CUDA_ARCH__
  return ::floorf(a);
#else
  return __builtin_floorf(a);
#endif
}
MWF double fmod_(double a, double b) {
#ifdef __CUDA_ARCH__
  return ::fmod(a, b);
#else
  return __builtin_fmod(a, b);
#endif
}
MWF float fmod_(float a, float b) {
#ifdef __CUDA_ARCH__
  return ::fmodf(a, b);
#else
  return __builtin_fmodf(a, b);
#endif
}
MWF double frexp_(double a, int *e) {
#ifdef __CUDA_ARCH__
  return ::frexp(a, e);
#else
  return __builtin_frexp(a, e);
#endif
}
MWF float frexp_(float a, int *e) {
#ifdef __CUDA_ARCH__
  return ::frexpf(a, e);
#else
  return __builtin_frexpf(a, e);
#endif
}
MWF bool signbit_(double a) { uint64_t u; memcpy(&u, &a, 8); return (u >> 63) != 0; }
MWF bool signbit_(float a) { uint32_t u; memcpy(&u, &a, 4); return (u >> 31) != 0; }

template <class T>
MWF T two_sum(T a, T b, T &e) {
  T s = a + b;
  T bb = s - a;
  e = (a - (s - bb)) + (b - bb);
  return s;
}
template <class T>
MWF T fast_two_sum(T a, T b, T &e) {
  T s = a + b;
  e = b - (s - a);
  return s;
}
template <class T>
MWF T two_prod(T a, T b, T &e) {
  T p = a * b;
  e = fma_(a, b, -p);
  return p;
}

/* float properties */
template <class T> struct FP;
template <> struct FP<double> {
  static constexpr int P = 53;                  /* significand bits */
  static constexpr int EMIN = -1022, EMAX = 1023;
  typedef uint64_t U;
  static constexpr int MBITS = 52;
  static constexpr U MMASK = 0x000FFFFFFFFFFFFFull;
  static constexpr int BIAS = 1023;
};
template <> struct FP<float> {
  static constexpr int P = 24;
  static constexpr int EMIN = -126, EMAX = 127;
  typedef uint32_t U;
  static constexpr int MBITS = 23;
  static constexpr U MMASK = 0x007FFFFFu;
  static constexpr int BIAS = 127;
};

template <class T>
MWF typename FP<T>::U bits_of(T x) {
  typename FP<T>::U u;
  memcpy(&u, &x, sizeof u);
  return u;
}
template <class T>
MWF T from_bits(typename FP<T>::U u) {
  T x;
  memcpy(&x, &u, sizeof x);
  return x;
}
/* 2^m for EMIN <= m <= EMAX */
template <class T>
MWF T pow2i(int m) {
  return from_bits<T>(static_cast<typename FP<T>::U>(m + FP<T>::BIAS) << FP<T>::MBITS);
}
/* x * 2^m for any m (two exact steps; handles the subnormal range) */
template <class T>
MWF T ldexp_(T x, int m) {
  if (m > FP<T>::EMAX) { x *= pow2i<T>(FP<T>::EMAX); m -= FP<T>::EMAX; if (m > FP<T>::EMAX) m = FP<T>::EMAX; }
  else if (m < FP<T>::EMIN) { x *= pow2i<T>(FP<T>::EMIN + FP<T>::P); m -= FP<T>::EMIN + FP<T>::P;
                              if (m < FP<T>::EMIN) { x *= pow2i<T>(FP<T>::EMIN + FP<T>::P); m -= FP<T>::EMIN + FP<T>::P; }
                              if (m < FP<T>::EMIN) m = FP<T>::EMIN; }
  return x * pow2i<T>(m);
}

/* ------------------------------------------------------------------ */
/* expansion type                                                      */
/* ------------------------------------------------------------------ */
template <class T, int K>
struct mwn {
  T v[K];
  MWF T &operator[](int i) { return v[i]; }
  MWF const T &operator[](int i) const { return v[i]; }
};

template <int W, class T, int K>
MWF mwn<T, W> trunc(const mwn<T, K> &a) {
  mwn<T, W> r;
  MW_PRAGMA_UNROLL
  for (int i = 0; i < W; i++) r.v[i] = (i < K) ? a.v[i] : T(0);
  return r;
}
template <int W, class T>
MWF mwn<T, W> load(const T *p) {
  mwn<T, W> r;
  MW_PRAGMA_UNROLL
  for (int i = 0; i < W; i++) r.v[i] = p[i];
  return r;
}
template <int W, class T>
MWF mwn<T, W> from_scalar(T a) {
  mwn<T, W> r;
  r.v[0] = a;
  MW_PRAGMA_UNROLL
  for (int i = 1; i < W; i++) r.v[i] = T(0);
  return r;
}
template <class T, int K>
MWF mwn<T, K> neg(const mwn<T, K> &a) {
  mwn<T, K> r;
  MW_PRAGMA_UNROLL
  for (int i = 0; i < K; i++) r.v[i] = -a.v[i];
  return r;
}
template <class T, int K>
MWF mwn<T, K> scale(const mwn<T, K> &a, T s) {   /* s a power of two */
  mwn<T, K> r;
  MW_PRAGMA_UNROLL
  for (int i = 0; i < K; i++) r.v[i] = a.v[i] * s;
  return r;
}

/* ------------------------------------------------------------------ */
/* order-level accumulator                                             */
/* ------------------------------------------------------------------ */
/* Renormalisation of N ordered-by-level partial sums into C words.
   A bottom-up TwoSum pass followed by a top-down TwoSum pass gives a
   non-overlapping expansion in the common case; exact cancellation of the
   leading levels or zero words in the middle can leave it out of order,
   which is detected (|S[l+1]| must not change S[l]) and fixed by repeating
   the two passes (rarely taken branch).  All steps are error-free, so the
   only loss is the final truncation to C words.                          */
template <int N, class T>
MWF bool overlapping(const T *S) {
  int bad = 0;
  MW_PRAGMA_UNROLL
  for (int l = 0; l < N - 1; l++) bad |= static_cast<int>(S[l] + S[l + 1] != S[l]);
  return bad != 0;
}
template <int N, class T>
MWF void renorm_passes(T *S) {
  MW_PRAGMA_UNROLL
  for (int l = N - 1; l >= 1; l--) S[l - 1] = two_sum(S[l - 1], S[l], S[l]);
  MW_PRAGMA_UNROLL
  for (int l = 1; l < N - 1; l++) S[l] = two_sum(S[l], S[l + 1], S[l + 1]);
}
/* rarely executed: repeat the passes until the expansion is ordered.
   Inlined on GPUs: an out-of-line call (even never taken) makes nvcc save
   live registers around the call site, which cost ~40 % in the float
   kernels; on CPUs the cold out-of-line version is faster.              */
template <int C, int N, class T>
#if defined(__CUDACC__)
__host__ __device__ __forceinline__
#else
__attribute__((noinline, cold))     /* on CPUs the out-of-line version is faster */
#endif
void renorm_fix(T *S) {
  /* taken often only in long division, where the remainder cancels */
  for (int pass = 0; pass < N - 1; pass++) {
    renorm_passes<N>(S);
    if (!overlapping<(N < C + 1 ? N : C + 1)>(S)) break;
  }
}

template <int C, int N, bool FAST, class T>
MWF mwn<T, C> renorm(T *S) {
  /* both passes are error-free; products never cancel at the leading
     level, so the ordering check is only needed for sums */
  renorm_passes<N>(S);
  if constexpr (!FAST) {
    if (__builtin_expect(overlapping<(N < C + 1 ? N : C + 1)>(S), 0)) renorm_fix<C, N>(S);
  }
  mwn<T, C> r;
  MW_PRAGMA_UNROLL
  for (int l = 0; l < C; l++) r.v[l] = (l < N) ? S[l] : T(0);
  if constexpr (N > C) r.v[C - 1] += S[C];
  return r;
}

/* L levels.  Each level keeps a running sum; push(l, x) adds x to level l
   with TwoSum and cascades the (exact) rounding error down to level l+1,
   l+2, ...; the last level is summed plainly.  With MW_CASCADE_GUARD the
   rounding errors of the last level are collected in a guard word that
   the renormalisation takes as one more input: the QD product error drops
   from ~25 to ~1 u^4 on random operands, but the FP64-bound GPU kernels
   become ~2x slower, so it is off by default.  No term lists: every index
   is a compile-time constant after unrolling, so the state stays in
   registers on CPUs and GPUs alike.  A level's first push is a plain
   assignment (tracked by a flag that the compiler folds).  CAP is kept for
   interface compatibility and unused.                                    */
template <class T, int L, int CAP>
struct LevelAccCascade {
#if defined(MW_CASCADE_GUARD)
  static constexpr bool GUARD = true;
#else
  static constexpr bool GUARD = false;
#endif
  T s[L];
  T g;          /* guard: rounding errors of the last level */
  bool used[L];
  MWF LevelAccCascade() {
    g = T(0);
    MW_PRAGMA_UNROLL
    for (int i = 0; i < L; i++) { s[i] = T(0); used[i] = false; }
  }
  MWF void push(int lev, T x) {
    if (lev > L - 1) lev = L - 1;
    /* no early return: nvcc does not unroll a loop with several exits and
       then keeps s[] / used[] in local memory */
    bool done = false;
    MW_PRAGMA_UNROLL
    for (int l = 0; l < L; l++) {
      if (done || l < lev) continue;
      if (!used[l]) {
        s[l] = x; used[l] = true; done = true;
      } else if (l == L - 1 && !GUARD) {
        s[l] += x; done = true;
      } else {
        T e;
        s[l] = two_sum(s[l], x, e);
        if (l == L - 1) { g += e; done = true; }
        else x = e;
      }
    }
  }
  /* MODE: 0 = lazy (level sums only, no renormalisation: the words are
     ordered by level but may overlap by a few ulps -- fine as an input of
     further error-free products / dominated sums), 1 = products, 2 = safe */
  template <int C, int MODE>
  MWF mwn<T, C> finish() {
    constexpr int N = GUARD ? L + 1 : L;
    T S[N];
    MW_PRAGMA_UNROLL
    for (int l = 0; l < L; l++) S[l] = s[l];
    if constexpr (GUARD) S[L] = g;
    if constexpr (MODE == 0) {
      mwn<T, C> r;
      MW_PRAGMA_UNROLL
      for (int l = 0; l < C; l++) r.v[l] = (l < L) ? S[l] : T(0);
      if constexpr (L > C) r.v[C - 1] += S[C];
      else if constexpr (GUARD) r.v[C - 1] += g;
      return r;
    } else {
      return renorm<C, N, MODE == 1>(S);
    }
  }
};

/* L levels, CAP terms per level (list form).  push() appends to a level;
   finish() sums each level with a TwoSum chain (errors appended to the
   next level, the last level summed plainly).  GCC resolves the counters
   at compile time and keeps the lists in registers; nvcc does not (they
   end up in local memory), so the CUDA build uses the cascade form.     */
template <class T, int L, int CAP>
struct LevelAccList {
  T t[L][CAP];
  int n[L];
  MWF LevelAccList() {
    MW_PRAGMA_UNROLL
    for (int i = 0; i < L; i++) n[i] = 0;
  }
  MWF void push(int lev, T x) {
    if (lev > L - 1) lev = L - 1;
    t[lev][n[lev]++] = x;
  }
  template <int C, int MODE>
  MWF mwn<T, C> finish() {
    T S[L];
    MW_PRAGMA_UNROLL
    for (int l = 0; l < L; l++) {
      T s = T(0);
      if (n[l] > 0) s = t[l][0];
      MW_PRAGMA_UNROLL
      for (int k = 1; k < CAP; k++) {
        if (k < n[l]) {
          if (l < L - 1) {
            T e;
            s = two_sum(s, t[l][k], e);
            t[l + 1][n[l + 1]++] = e;
          } else {
            s = s + t[l][k];
          }
        }
      }
      S[l] = s;
    }
    if constexpr (MODE == 0) {
      mwn<T, C> r;
      MW_PRAGMA_UNROLL
      for (int l = 0; l < C; l++) r.v[l] = (l < L) ? S[l] : T(0);
      if constexpr (L > C) r.v[C - 1] += S[C];
      return r;
    } else {
      return renorm<C, L, MODE == 1>(S);
    }
  }
};

#if defined(__CUDACC__) || defined(MW_CASCADE_ACC)
template <class T, int L, int CAP> using LevelAcc = LevelAccCascade<T, L, CAP>;
#else
template <class T, int L, int CAP> using LevelAcc = LevelAccList<T, L, CAP>;
#endif

/* ------------------------------------------------------------------ */
/* arithmetic                                                          */
/* ------------------------------------------------------------------ */
/* C-word product of an A-word and a B-word expansion (MODE 0: lazy) */
template <int C, int MODE = 1, class T, int A, int B>
MWF mwn<T, C> mul(const mwn<T, A> &a, const mwn<T, B> &b) {
  if constexpr (C == 1) {
    return from_scalar<1>(a.v[0] * b.v[0]);
  } else {
    LevelAcc<T, C, (C + 1) * (C + 2) / 2 + 2 * C> acc;
    MW_PRAGMA_UNROLL
    for (int n = 0; n < C; n++) {
      MW_PRAGMA_UNROLL
      for (int i = 0; i < A; i++) {
        const int j = n - i;
        if (j >= 0 && j < B) {
          if (n < C - 1) {
            T e;
            T p = two_prod(a.v[i], b.v[j], e);
            acc.push(n, p);
            acc.push(n + 1, e);
          } else {
            acc.push(n, a.v[i] * b.v[j]);
          }
        }
      }
    }
    return acc.template finish<C, MODE>();
  }
}

/* C-word sum of an A-word and a B-word expansion (words aligned by index) */
template <int C, class T, int A, int B>
MWF mwn<T, C> add(const mwn<T, A> &a, const mwn<T, B> &b) {
  if constexpr (C == 1) {
    return from_scalar<1>(a.v[0] + b.v[0]);
  } else {
    LevelAcc<T, C + 1, A + B + C + 3> acc;
    MW_PRAGMA_UNROLL
    for (int i = 0; i < A; i++) acc.push(i, a.v[i]);
    MW_PRAGMA_UNROLL
    for (int i = 0; i < B; i++) acc.push(i, b.v[i]);
    return acc.template finish<C, 2>();
  }
}
/* c + q when |q| <= |c|/2 is guaranteed (no cancellation): C levels and
   the unchecked renormalisation, as for products */
template <int C, int MODE = 1, class T, int A, int B>
MWF mwn<T, C> add_dom(const mwn<T, A> &c, const mwn<T, B> &q) {
  if constexpr (C == 1) {
    return from_scalar<1>(c.v[0] + q.v[0]);
  } else {
    LevelAcc<T, C, A + B + C + 2> acc;
    MW_PRAGMA_UNROLL
    for (int i = 0; i < A; i++) acc.push(i, c.v[i]);
    MW_PRAGMA_UNROLL
    for (int i = 0; i < B; i++) acc.push(i, q.v[i]);
    return acc.template finish<C, MODE>();
  }
}

template <int C, class T, int A, int B>
MWF mwn<T, C> sub(const mwn<T, A> &a, const mwn<T, B> &b) {
  return add<C>(a, neg(b));
}

/* c + a b, with the product formed at C words */
template <int C, class T, int A, int B, int D>
MWF mwn<T, C> fma(const mwn<T, D> &c, const mwn<T, A> &a, const mwn<T, B> &b) {
  if constexpr (C == 1) {
    return from_scalar<1>(fma_(a.v[0], b.v[0], c.v[0]));
  } else {
    LevelAcc<T, C + 1, (C + 1) * (C + 2) / 2 + 3 * C + D + 4> acc;
    MW_PRAGMA_UNROLL
    for (int i = 0; i < D; i++) acc.push(i, c.v[i]);
    MW_PRAGMA_UNROLL
    for (int n = 0; n < C; n++) {
      MW_PRAGMA_UNROLL
      for (int i = 0; i < A; i++) {
        const int j = n - i;
        if (j >= 0 && j < B) {
          if (n < C - 1) {
            T e;
            T p = two_prod(a.v[i], b.v[j], e);
            acc.push(n, p);
            acc.push(n + 1, e);
          } else {
            acc.push(n, a.v[i] * b.v[j]);
          }
        }
      }
    }
    return acc.template finish<C, 2>();
  }
}

/* C-word quotient a / b (long division, C+1 digits, one hardware division) */
template <int C, class T, int A, int B>
MWF mwn<T, C> div(const mwn<T, A> &a, const mwn<T, B> &b) {
  const T inv = T(1) / b.v[0];
  if constexpr (C == 1) {
    return from_scalar<1>(a.v[0] * inv);
  } else if constexpr (C == 2) {
    /* two words: first remainder kept exactly, third quotient digit
       (the ddcore dd_div; ~0.5 u^2, no renormalisation checks)        */
    const T ah = a.v[0], al = (A > 1) ? a.v[1] : T(0);
    const T bh = b.v[0], bl = (B > 1) ? b.v[1] : T(0);
    const T q1 = ah * inv;
    T pe, e1, e2, e3, ple;
    const T p = two_prod(q1, bh, pe);
    const T pl = two_prod(q1, bl, ple);
    T r = two_sum(ah - p, -pe, e1);
    r = two_sum(r, al, e2);
    r = two_sum(r, -pl, e3);
    T rl = ((e1 + e2) + e3) - ple;
    if constexpr (A > 2) rl += a.v[2];
    const T q2 = r * inv;
    T p2e;
    const T p2 = two_prod(q2, bh, p2e);
    const T r2 = ((r - p2) - p2e) + fma_(-q2, bl, rl);
    const T q3 = r2 * inv;
    T l;
    const T h = fast_two_sum(q1, q2, l);
    l += q3;
    mwn<T, 2> o;
    o.v[0] = fast_two_sum(h, l, o.v[1]);
    return o;
  } else {
    T q[C + 1];
    mwn<T, C> r = trunc<C>(a);
    q[0] = a.v[0] * inv;
    MW_PRAGMA_UNROLL
    for (int k = 1; k <= C; k++) {
      /* r <- r - q[k-1] b */
      r = fma<C>(r, trunc<C>(b), from_scalar<1>(-q[k - 1]));
      q[k] = r.v[0] * inv;
    }
    LevelAcc<T, C + 1, C + 3> acc;
    MW_PRAGMA_UNROLL
    for (int k = 0; k <= C; k++) acc.push(k, q[k]);
    return acc.template finish<C, 2>();
  }
}

/* C-word square root (Newton, doubling the number of words) */
template <int C, class T, int A>
MWF mwn<T, C> sqrt(const mwn<T, A> &a) {
  const T s0 = sqrt_(a.v[0]);
  if (!(a.v[0] > T(0))) return from_scalar<C>(a.v[0] == T(0) ? a.v[0] : s0);
  if constexpr (C == 1) {
    return from_scalar<1>(s0);
  } else if constexpr (C == 2) {
    /* s + (a - s^2) / (2 s) */
    T pe;
    const T p = two_prod(s0, s0, pe);
    T r = (a.v[0] - p) - pe;
    if constexpr (A > 1) r += a.v[1];
    if constexpr (A > 2) r += a.v[2];
    mwn<T, 2> o;
    o.v[0] = fast_two_sum(s0, r / (T(2) * s0), o.v[1]);
    return o;
  } else {
    /* one step: s + (a - s^2) / (2 s) at C words, correction at C - h words */
    constexpr int H = (C + 1) / 2;
    const mwn<T, H> sh = sqrt<H>(a);
    const mwn<T, C> r = fma<C>(trunc<C>(a), neg(sh), sh);       /* a - s^2 */
    const T inv2s = T(0.5) / sh.v[0];
    mwn<T, C - H> d;
    if constexpr (C - H == 1) {
      d.v[0] = r.v[0] * inv2s;
    } else {
      d = div<C - H>(r, scale(sh, T(2)));
    }
    return add<C>(sh, d);
  }
}

} // namespace mw

#endif
