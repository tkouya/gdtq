/*
 * mw_elem.h -- generic multi-word elementary functions (CORE-dtq design)
 *
 *   template parameter S : schedule struct MWS_xx from mw_tables.h
 *                          (words K, target bits, Horner degrees / widths)
 *   template parameter T : base type (double / float)
 *
 * Same algorithms as the hand-tuned double-double versions (ddcore):
 *   - strong Cody-Waite reduction with exact k * C_i (TwoProd) summed by an
 *     error-free order-level accumulator; Payne-Hanek for huge sin/cos args
 *   - (K+1)-word tables
 *   - Horner with a per-stage word count (mixed precision); every stage is
 *     "dominated" (|z p| << |c|) so no cancellation checks are needed there
 *   - reconstruction in the form big + small correction, final sums with the
 *     cancellation-safe accumulator
 */
#ifndef MW_ELEM_H
#define MW_ELEM_H

#include "mw_arith.h"
#include "mw_tables.h"

namespace mw {

/* ------------------------------------------------------------------ */
/* table access                                                        */
/* ------------------------------------------------------------------ */
template <class T> struct Tab;

#define MW_TAB_ACCESSORS(TT, P)                                                         \
  template <> struct Tab<TT> {                                                          \
    static MWF const TT *exp_t1(int j) { return MWT_##P##_EXP_T1[j]; }                  \
    static MWF const TT *exp_d0(int i) { return MWT_##P##_EXP_D0[i]; }                  \
    static MWF const TT *em_t1(int j) { return MWT_##P##_EM_T1[j + 96]; }               \
    static MWF const TT *em_d0(int i) { return MWT_##P##_EM_D0[i + 32]; }               \
    static MWF const TT *log_t1(int i) { return MWT_##P##_LOG_T1[i]; }                  \
    static MWF const TT *log_t2(int i) { return MWT_##P##_LOG_T2[i + MW_LOG_N2]; }      \
    static MWF const TT *sc_tab(int j) { return MWT_##P##_SC_TAB[j]; }                  \
    static MWF const TT *at_tab(int j) { return MWT_##P##_AT_TAB[j]; }                  \
    static MWF const TT *hy_tab(int k) { return MWT_##P##_HY_TAB[k]; }                  \
    static MWF const TT *ln2() { return MWT_##P##_LN2; }                                \
    static MWF const TT *inv_ln10() { return MWT_##P##_INV_LN10; }                      \
    static MWF const TT *pi() { return MWT_##P##_PI; }                                  \
    static MWF const TT *pi2() { return MWT_##P##_PI2; }                                \
    static MWF const TT *pi256() { return MWT_##P##_PI256; }                            \
    static MWF TT exp_part(int i) { return MWT_##P##_EXP_PARTS[i]; }                    \
    static MWF TT exp_inv() { return MWT_##P##_EXP_INV[0]; }                            \
    static MWF TT sc_part(int i) { return MWT_##P##_SC_PARTS[i]; }                      \
    static MWF TT sc_inv() { return MWT_##P##_SC_INV[0]; }                              \
    static constexpr int SC_SHIFT = MWT_##P##_SC_SHIFT;                                 \
    static constexpr int SC_NPARTS_AVAIL = MWT_##P##_SC_NPARTS;                         \
    static MWF const TT *expc(int i) { return MWT_##P##_EXPC[i]; }                      \
    static MWF const TT *logc(int i) { return MWT_##P##_LOGC[i]; }                      \
    static MWF const TT *sinc(int i) { return MWT_##P##_SINC[i]; }                      \
    static MWF const TT *cosc(int i) { return MWT_##P##_COSC[i]; }                      \
    static MWF const TT *atanc(int i) { return MWT_##P##_ATANC[i]; }                    \
    static MWF const TT *sinhc(int i) { return MWT_##P##_SINHC[i]; }                    \
    static MWF const TT *coshc(int i) { return MWT_##P##_COSHC[i]; }                    \
  };
MW_TAB_ACCESSORS(double, D)
MW_TAB_ACCESSORS(float, F)
#undef MW_TAB_ACCESSORS

template <class T> struct Lim;
template <> struct Lim<double> {
  static constexpr double EXP_FAST = 708.0, EXP_OVF = 709.782712893384, EXP_UNF = -745.2;
  static constexpr double SC_FAST = 1048576.0;
  static constexpr double BIG = 0x1p1000, HUGE_ = 1.7976931348623157e308;
  static constexpr double SQRT_BIG = 0x1p500;    /* x^2 safe, 1/x^2 below any target */
};
template <> struct Lim<float> {
  static constexpr float EXP_FAST = 87.0f, EXP_OVF = 88.72283f, EXP_UNF = -103.98f;
  static constexpr float SC_FAST = 4096.0f;
  static constexpr float BIG = 0x1p100f, HUGE_ = 3.4028234e38f;
  static constexpr float SQRT_BIG = 0x1p60f;
};

/* ------------------------------------------------------------------ */
/* small helpers                                                       */
/* ------------------------------------------------------------------ */
template <class T> MWF T abs_(T x) { return x < T(0) ? -x : x; }
template <class T> MWF bool isnan_(T x) { return x != x; }
template <class T> MWF bool isinf_(T x) { return abs_(x) > Lim<T>::HUGE_; }
template <class T> MWF T inf_() { return Lim<T>::HUGE_ * T(2); }
template <class T> MWF T nan_() { return inf_<T>() - inf_<T>(); }

template <int K, class T> MWF mwn<T, K> mwnan() { return from_scalar<K>(nan_<T>()); }
template <int K, class T> MWF mwn<T, K> mwinf(bool neg = false) { return from_scalar<K>(neg ? -inf_<T>() : inf_<T>()); }
template <class T, int K> MWF mwn<T, K> neg_if(const mwn<T, K> &a, bool n) { return n ? neg(a) : a; }
template <class T, int K> MWF mwn<T, K> mwabs(const mwn<T, K> &a) { return a.v[0] < T(0) ? neg(a) : a; }
template <int W, class T> MWF mwn<T, W> loadn(const T *p, T s) {   /* s = +-1 */
  mwn<T, W> r;
  MW_PRAGMA_UNROLL
  for (int i = 0; i < W; i++) r.v[i] = s * p[i];
  return r;
}

/* ------------------------------------------------------------------ */
/* mixed-width Horner                                                  */
/* ------------------------------------------------------------------ */
/* p(z) = sum_{i<DEG} c_i z^i;  stage i keeps W(i) words, the product
   z p_{i+1} is formed with W(i+1) words (its own precision need) and is
   dominated by c_i, so the dominated (unchecked) sum is used.           */
template <class Pl, class T, int K, int I>
MWF mwn<T, Pl::W(I)> horner_rec(const mwn<T, K> &z) {
  constexpr int W = Pl::W(I);
  if constexpr (I == Pl::DEG - 1) {
    return load<W>(Pl::C(I));
  } else {
    constexpr int W1 = Pl::W(I + 1);
    const mwn<T, W1> p = horner_rec<Pl, T, K, I + 1>(z);
    if constexpr (W == 1) {
      return from_scalar<1>(fma_(z.v[0], p.v[0], Pl::C(I)[0]));
    } else {
      const mwn<T, W1> q = mul<W1, 0>(trunc<W1>(z), p);
      return add_dom<W, (I == 0) ? 1 : 0>(load<W>(Pl::C(I)), q);
    }
  }
}
template <class Pl, class T, int K>
MWF mwn<T, Pl::W(0)> horner(const mwn<T, K> &z) {
  return horner_rec<Pl, T, K, 0>(z);
}

#define MW_POLY(NAME, CN, ACC)                                             \
  template <class S, class T> struct NAME {                                \
    static constexpr int DEG = S::CN##_DEG;                                \
    static constexpr int W(int i) { return S::CN##_W(i); }                \
    static MWF const T *C(int i) { return Tab<T>::ACC(i); }                \
  };
MW_POLY(PolyExp, EXPC, expc)
MW_POLY(PolyLog, LOGC, logc)
MW_POLY(PolySin, SINC, sinc)
MW_POLY(PolyCos, COSC, cosc)
MW_POLY(PolyAtan, ATANC, atanc)
MW_POLY(PolySinh, SINHC, sinhc)
MW_POLY(PolyCosh, COSHC, coshc)
MW_POLY(PolyLog1, LOGC1, logc)       /* one-level log: |z| < 2^-7 */
#undef MW_POLY

/* ================================================================== */
/* exp family                                                          */
/* ================================================================== */
/* x = k L + r,  L = log2/4096,  k = 4096 m + 64 j + i.  The input may
   carry up to K+1 words (pow).  r is returned with K words.            */
template <class S, class T, int KX>
MWF mwn<T, S::K> exp_reduce(const mwn<T, KX> &x, int64_t &k) {
  constexpr int K = S::K, NP = S::EXP_NPARTS;
  const T kd = rint_(x.v[0] * Tab<T>::exp_inv());
  k = static_cast<int64_t>(kd);
  LevelAcc<T, K + 1, 2 * NP + KX + K + 4> acc;
  acc.push(0, fma_(-kd, Tab<T>::exp_part(0), x.v[0]));        /* exact */
  MW_PRAGMA_UNROLL
  for (int i = 1; i < NP; i++) {
    T e;
    const T p = two_prod(kd, Tab<T>::exp_part(i), e);
    acc.push(i - 1, -p);
    acc.push(i, -e);
  }
  MW_PRAGMA_UNROLL
  for (int j = 1; j < KX; j++) acc.push(j - 1, x.v[j]);
  return acc.template finish<K, 2>();
}

/* q = expm1(r) = r Q(r) */
template <class S, class T>
MWF mwn<T, S::K> expm1_poly(const mwn<T, S::K> &r) {
  const auto Q = horner<PolyExp<S, T>>(r);
  return mul<S::K, 0>(r, Q);
}

/* exp(x) = 2^m T1 (1 + w),  w = d0 + q (1 + d0) */
template <class S, class T, int KX>
MWF mwn<T, S::K> exp_kernel(const mwn<T, KX> &x, int &m) {
  constexpr int K = S::K;
  int64_t k;
  const mwn<T, K> r = exp_reduce<S>(x, k);
  const mwn<T, K> q = expm1_poly<S>(r);
  const int i = static_cast<int>(k & 63), j = static_cast<int>((k >> 6) & 63);
  m = static_cast<int>(k >> 12);
  const mwn<T, K> d0 = load<K>(Tab<T>::exp_d0(i));
  const mwn<T, K> w = add_dom<K, 0>(add_dom<K, 0>(d0, q), mul<K, 0>(q, d0));
  const mwn<T, K + 1> T1 = load<K + 1>(Tab<T>::exp_t1(j));
  return add_dom<K>(T1, mul<K, 0>(trunc<K>(T1), w));
}

template <class T, int K>
MWF mwn<T, K> scale_pow2(const mwn<T, K> &y, int m) {
  if (m >= FP<T>::EMIN && m <= FP<T>::EMAX) return scale(y, pow2i<T>(m));
  mwn<T, K> r;
  MW_PRAGMA_UNROLL
  for (int i = 0; i < K; i++) r.v[i] = ldexp_(y.v[i], m);
  return r;
}

template <class S, class T, int KX>
MWF mwn<T, S::K> exp_any(const mwn<T, KX> &x) {
  constexpr int K = S::K;
  const T x0 = x.v[0];
  if (!(abs_(x0) < Lim<T>::EXP_FAST)) {
    if (isnan_(x0)) return mwnan<K, T>();
    if (x0 >= Lim<T>::EXP_OVF) return mwinf<K, T>();
    if (x0 <= Lim<T>::EXP_UNF) return from_scalar<K>(T(0));
  }
  int m;
  const mwn<T, K> y = exp_kernel<S>(x, m);
  return scale_pow2(y, m);
}

template <class S, class T, int KX = S::K>
MWF mwn<T, S::K> exp(const mwn<T, KX> &x) { return exp_any<S>(x); }

template <class S, class T>
MWF mwn<T, S::K> expm1(const mwn<T, S::K> &x) {
  constexpr int K = S::K;
  const T x0 = x.v[0];
  if (!(abs_(x0) < T(1))) {
    if (isnan_(x0)) return x;
    const mwn<T, K> e = exp<S>(x);
    if (isinf_(e.v[0])) return e;
    return add<K>(e, from_scalar<1>(T(-1)));
  }
  /* |x| < 1: k = 64 j + i, i in [-32, 31], j in [-93, 93], no 2^m;
     expm1(x) = D_j + T_j w, w = d_i + q (1 + d_i) (no cancellation)    */
  int64_t k;
  const mwn<T, K> r = exp_reduce<S>(x, k);
  const mwn<T, K> q = expm1_poly<S>(r);
  const int i = static_cast<int>(((k + 32) & 63) - 32);
  const int j = static_cast<int>((k - i) >> 6);
  const T *Tj = Tab<T>::em_t1(j);
  const mwn<T, K> d0 = load<K>(Tab<T>::em_d0(i));
  const mwn<T, K> w = add_dom<K, 0>(add_dom<K, 0>(d0, q), mul<K, 0>(q, d0));
  const mwn<T, K + 1> D = load<K + 1>(Tj + MW_NW);
  return add<K>(D, mul<K, 0>(load<K>(Tj), w));
}

/* ================================================================== */
/* log family                                                          */
/* ================================================================== */
/* log(2^eadd a), a = a0 + a1 + ... (KA words, a0 > 0 normal, pre-scaled
   so that e + up stays in the exponent range); result with KR words.   */
template <class S, class T, int KR, int KA>
MWF mwn<T, KR> log_kernel(const mwn<T, KA> &a, int eadd) {
  constexpr int K = S::K;
  typedef typename FP<T>::U U;
  const U bits = bits_of(a.v[0]);
  const int e = static_cast<int>(bits >> FP<T>::MBITS) - FP<T>::BIAS;
  const int i1 = static_cast<int>((bits >> (FP<T>::MBITS - 7)) & 127);
  const int up = (i1 >= 53) ? 1 : 0;
  const T E = static_cast<T>(e + up + eadd);
  const T m0 = from_bits<T>((bits & FP<T>::MMASK) | (static_cast<U>(FP<T>::BIAS - up) << FP<T>::MBITS));
  const T sc = pow2i<T>(-(e + up));

  /* level 1: z1 = m c1 - 1 (the FMA is exact) */
  const T *t1 = Tab<T>::log_t1(i1);
  const T c1 = t1[0];
  mwn<T, K> z1;
  {
    LevelAcc<T, K + 1, 2 * KA + K + 4> acc;
    acc.push(0, fma_(m0, c1, T(-1)));
    MW_PRAGMA_UNROLL
    for (int j = 1; j < KA; j++) {
      T er;
      const T p = two_prod(a.v[j] * sc, c1, er);
      acc.push(j, p);
      acc.push(j + 1, er);
    }
    z1 = acc.template finish<K, 2>();
  }
  /* level 2: z2 = z1 c2 + (c2 - 1)  (skipped for low-precision classes,
     where a longer single-word polynomial tail is cheaper) */
  constexpr bool TWO = (S::LOG_LEVELS == 2);
  const int i2 = TWO ? static_cast<int>(rint_(z1.v[0] * T(16384))) : 0;
  const T *t2 = Tab<T>::log_t2(i2);
  const T c2 = t2[0];
  mwn<T, K> z = z1;
  if constexpr (TWO) {
    LevelAcc<T, K + 1, 3 * K + 6> acc;
    acc.push(0, c2 - T(1));
    MW_PRAGMA_UNROLL
    for (int j = 0; j < K; j++) {
      T er;
      const T p = two_prod(z1.v[j], c2, er);
      acc.push(j, p);
      acc.push(j + 1, er);
    }
    z = acc.template finish<K, 2>();
  }
  /* log1p(z) = z Q(z) */
  mwn<T, K> P;
  if constexpr (TWO) P = mul<K, 0>(z, horner<PolyLog<S, T>>(z));
  else P = mul<K, 0>(z, horner<PolyLog1<S, T>>(z));
  /* E log2 - log c1 - log c2 + P */
  LevelAcc<T, KR + 1, 6 * (KR + 1) + K + 6> acc;
  const T *ln2 = Tab<T>::ln2();
  constexpr int NT = (KR + 1 < MW_NW) ? KR + 1 : MW_NW;     /* table words used */
  MW_PRAGMA_UNROLL
  for (int j = 0; j < NT; j++) {
    T er;
    const T p = two_prod(E, ln2[j], er);
    acc.push(j, p);
    acc.push(j + 1, er);
  }
  MW_PRAGMA_UNROLL
  for (int j = 0; j < NT; j++) { acc.push(j, t1[1 + j]); if (TWO) acc.push(j, t2[1 + j]); }
  MW_PRAGMA_UNROLL
  for (int j = 0; j < K; j++) acc.push(j, P.v[j]);
  return acc.template finish<KR, 2>();
}

/* checks and pre-scaling; returns false (and sets y) for special inputs */
template <class T, int KA, int KR>
MWF bool log_prepare(mwn<T, KA> &a, int &eadd, mwn<T, KR> &y) {
  eadd = 0;
  const T a0 = a.v[0];
  if (!(a0 > T(0))) {
    y = (a0 == T(0)) ? mwinf<KR, T>(true) : mwnan<KR, T>();
    return false;
  }
  if (isinf_(a0)) { y = mwinf<KR, T>(); return false; }
  const int eb = (sizeof(T) == 8) ? 64 : 32;
  if (a0 >= pow2i<T>(FP<T>::EMAX - 2)) {
    a = scale(a, pow2i<T>(-eb)); eadd = eb;
  } else if (a0 < pow2i<T>(FP<T>::EMIN + 2)) {
    MW_PRAGMA_UNROLL
    for (int i = 0; i < KA; i++) a.v[i] = a.v[i] * pow2i<T>(FP<T>::P + 4);   /* exact */
    eadd = -(FP<T>::P + 4);
  }
  return true;
}

template <class S, class T, int KX = S::K>
MWF mwn<T, S::K> log(const mwn<T, KX> &x) {
  constexpr int K = S::K;
  mwn<T, KX> a = x;
  mwn<T, K> y;
  int eadd;
  if (!log_prepare(a, eadd, y)) return y;
  return log_kernel<S, T, K>(a, eadd);
}

template <class S, class T, int KX = S::K>
MWF mwn<T, S::K> log10(const mwn<T, KX> &x) {
  constexpr int K = S::K;
  mwn<T, KX> a = x;
  mwn<T, K> y;
  int eadd;
  if (!log_prepare(a, eadd, y)) return y;
  const mwn<T, K + 1> L = log_kernel<S, T, K + 1>(a, eadd);
  /* one extra word, so that the plainly summed last level is below u^K */
  return trunc<K>(mul<K + 1>(L, load<K + 1>(Tab<T>::inv_ln10())));
}

/* log1p of y given as an unevaluated KY-word sum (KY <= K + 1) */
template <class S, class T, int KY>
MWF mwn<T, S::K> log1p_n(const mwn<T, KY> &y) {
  constexpr int K = S::K;
  const T y0 = y.v[0];
  if (isnan_(y0)) return mwnan<K, T>();
  if (y0 <= T(-1)) {
    if (y0 < T(-1)) return mwnan<K, T>();
    T l = T(0);
    MW_PRAGMA_UNROLL
    for (int i = 1; i < KY; i++) l += y.v[i];
    if (l <= T(0)) return (l == T(0)) ? mwinf<K, T>(true) : mwnan<K, T>();
  }
  if (isinf_(y0)) return mwinf<K, T>();
  /* 1 + y exactly (K+1 words) */
  mwn<T, K + 1> a;
  if (abs_(y0) < Lim<T>::BIG) {
    a = add<K + 1>(from_scalar<1>(T(1)), y);
  } else {
    a = trunc<K + 1>(y);
  }
  mwn<T, K> r;
  int eadd;
  if (!log_prepare(a, eadd, r)) return r;
  return log_kernel<S, T, K>(a, eadd);
}
template <class S, class T>
MWF mwn<T, S::K> log1p(const mwn<T, S::K> &y) { return log1p_n<S>(y); }

/* ================================================================== */
/* sin / cos                                                           */
/* ================================================================== */
/* Payne-Hanek (computed in double for either base type): x * 256/pi mod 512 */
struct PHAcc {
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0;
  MWF void add(double v) {
    double e, e2, e3, e4, e5;
    s0 = two_sum(s0, v, e);
    s1 = two_sum(s1, e, e2);
    s2 = two_sum(s2, e2, e3);
    s3 = two_sum(s3, e3, e4);
    s4 = two_sum(s4, e4, e5);
    s5 += e5;
  }
};
MWF void ph_mod(double v, double &K, PHAcc &F) {
  int ex;
  const double M = ldexp_(frexp_(v, &ex), 53);
  const int E = ex - 53;
  int i0 = (E - 1 + 23) / 24;
  if (E - 1 <= 0 || i0 < 1) i0 = 1;
  for (int i = i0; i <= MW_NCHUNK; i++) {
    const int sh = E + 7 - 24 * i;
    if (sh < -330) break;
    const double sc = ldexp_(1.0, sh);
    double e;
    const double p = two_prod(M, MWT_TWO_OVER_PI[i - 1], e);
    const double d0 = p * sc, d1 = e * sc;
    double fl = rint_(d0);
    K += fmod_(fl, 512.0);
    F.add(d0 - fl);
    fl = rint_(d1);
    K += fmod_(fl, 512.0);
    F.add(d1 - fl);
  }
}
template <class S, class T, int K, int KX>
#if defined(__CUDACC__)
__host__ __device__ __noinline__
#else
__attribute__((noinline))
#endif
void sc_reduce_ph(const mwn<T, KX> &x, mwn<T, K> &r, int64_t &k) {
  double Kt = 0.0;
  PHAcc F;
  for (int c = 0; c < KX; c++) {
    const double v = static_cast<double>(x.v[c]);
    if (v == 0.0) continue;
    double Kc = 0.0;
    PHAcc Fc;
    ph_mod(v < 0 ? -v : v, Kc, Fc);
    const double sg = v < 0 ? -1.0 : 1.0;
    Kt += sg * fmod_(Kc, 512.0);
    F.add(sg * Fc.s0); F.add(sg * Fc.s1); F.add(sg * Fc.s2); F.add(sg * Fc.s3); F.add(sg * Fc.s4); F.add(sg * Fc.s5);
  }
  const double n = rint_(F.s0 + F.s1);
  Kt += n;
  F.s0 -= n;
  k = static_cast<int64_t>(fmod_(Kt, 512.0)) & 511;
  /* f (5 doubles) * pi/256 (5 doubles) -> K words of T */
  mwn<double, 5> f;
  {
    double S6[6] = {F.s0, F.s1, F.s2, F.s3, F.s4, F.s5};
    f = renorm<5, 6, false>(S6);
  }
  const mwn<double, 4> rd = mul<4>(f, load<5>(MWT_D_PI256));
  if constexpr (sizeof(T) == 8) {
    r = trunc<K>(rd);
  } else {
    /* split the doubles into floats */
    LevelAcc<float, K, 16> acc;
    for (int i = 0; i < 4; i++) {
      double v = rd.v[i];
      MW_PRAGMA_UNROLL
      for (int s = 0; s < 3; s++) {
        const float h = static_cast<float>(v);
        acc.push(2 * i + s < K - 1 ? 2 * i + s : K - 1, h);
        v -= static_cast<double>(h);
      }
    }
    r = acc.template finish<K, 2>();
  }
}

/* PH = false: Cody-Waite range only (|x| < SC_FAST); larger arguments
   return false (used by the fast pass of the two-pass evaluation)      */
template <class S, class T, int KX, bool PH = true>
MWF bool sc_reduce(const mwn<T, KX> &x, mwn<T, S::K> &r, int64_t &k) {
  constexpr int K = S::K, NP = S::SC_NPARTS;
  const T x0 = x.v[0];
  if (abs_(x0) < Lim<T>::SC_FAST) {
    /* float: work on x 2^SHIFT (exact) against P 2^SHIFT, scale back at the end */
    constexpr int SH = Tab<T>::SC_SHIFT;
    const T up = pow2i<T>(SH), dn = pow2i<T>(-SH);
    const T kd = rint_(x0 * Tab<T>::sc_inv());
    k = static_cast<int64_t>(kd);
    LevelAcc<T, K + 1, 2 * NP + KX + K + 4> acc;
    acc.push(0, fma_(-kd, Tab<T>::sc_part(0), x0 * up));
    MW_PRAGMA_UNROLL
    for (int i = 1; i < NP; i++) {
      T e;
      const T p = two_prod(kd, Tab<T>::sc_part(i), e);
      acc.push(i - 1, -p);
      acc.push(i, -e);
    }
    MW_PRAGMA_UNROLL
    for (int j = 1; j < KX; j++) acc.push(j - 1, x.v[j] * up);
    r = acc.template finish<K, 2>();
    if constexpr (SH != 0) r = scale(r, dn);
    return true;
  }
  if constexpr (!PH) {
    return false;
  } else {
    if (isnan_(x0) || isinf_(x0)) return false;
    sc_reduce_ph<S, T, K, KX>(x, r, k);
    return true;
  }
}

/* rotation reconstruction  A g(r) + B h(r) = A + B h + A (g - 1) */
template <class T, int K>
MWF mwn<T, K> rot_recon(const T *A, T sa, const T *B, T sb, const mwn<T, K> &h, const mwn<T, K> &gm1) {
  const mwn<T, K + 1> Aw = loadn<K + 1>(A, sa);
  const mwn<T, K> Bh = mul<K, 0>(loadn<K>(B, sb), h);
  const mwn<T, K> Ag = mul<K, 0>(trunc<K>(Aw), gm1);
  return add<K>(Aw, add<K>(Bh, Ag));
}

template <class S, class T>
struct SCParts { mwn<T, S::K> s, cm1; int64_t k; };

template <class S, class T, int KX, bool PH = true>
MWF bool sc_prepare(const mwn<T, KX> &x, SCParts<S, T> &o) {
  constexpr int K = S::K;
  mwn<T, K> r;
  if (!sc_reduce<S, T, KX, PH>(x, r, o.k)) return false;
  const mwn<T, K> z = mul<K>(r, r);
  o.s = mul<K, 0>(r, horner<PolySin<S, T>>(z));
  o.cm1 = mul<K, 0>(z, horner<PolyCos<S, T>>(z));
  return true;
}
template <class S, class T>
MWF mwn<T, S::K> sc_value(const SCParts<S, T> &o, int64_t k) {
  const int q = static_cast<int>((k >> 7) & 3), j = static_cast<int>(k & 127);
  const T *Tj = Tab<T>::sc_tab(j);
  const T *A = (q & 1) ? Tj + MW_NW : Tj;
  const T *B = (q & 1) ? Tj : Tj + MW_NW;
  const T sa = (q & 2) ? T(-1) : T(1);
  const T sb = ((q + 1) & 2) ? T(-1) : T(1);
  return rot_recon<T, S::K>(A, sa, B, sb, o.s, o.cm1);
}

template <class S, class T, int KX = S::K, bool PH = true>
MWF mwn<T, S::K> sin(const mwn<T, KX> &x) {
  SCParts<S, T> o;
  if (!sc_prepare<S, T, KX, PH>(x, o)) return mwnan<S::K, T>();
  return sc_value<S>(o, o.k);
}
template <class S, class T, int KX = S::K, bool PH = true>
MWF mwn<T, S::K> cos(const mwn<T, KX> &x) {
  SCParts<S, T> o;
  if (!sc_prepare<S, T, KX, PH>(x, o)) return mwnan<S::K, T>();
  return sc_value<S>(o, o.k + 128);
}
template <class S, class T, int KX = S::K, bool PH = true>
MWF void sincos(const mwn<T, KX> &x, mwn<T, S::K> &s, mwn<T, S::K> &c) {
  SCParts<S, T> o;
  if (!sc_prepare<S, T, KX, PH>(x, o)) { s = c = mwnan<S::K, T>(); return; }
  s = sc_value<S>(o, o.k);
  c = sc_value<S>(o, o.k + 128);
}
template <class S, class T, int KX = S::K, bool PH = true>
MWF mwn<T, S::K> tan(const mwn<T, KX> &x) {
  mwn<T, S::K> s, c;
  sincos<S, T, KX, PH>(x, s, c);
  return div<S::K>(s, c);
}

/* ================================================================== */
/* sinh / cosh / tanh                                                  */
/* ================================================================== */
/* 0 <= x <= 1:  x = k/128 + r,  sinh/cosh by the rotation kernel */
template <class S, class T>
MWF void hy_small(const mwn<T, S::K> &x, mwn<T, S::K> &sh, mwn<T, S::K> &ch, bool want_s, bool want_c) {
  constexpr int K = S::K;
  const T kd = rint_(x.v[0] * T(128));
  mwn<T, K> r = x;
  r.v[0] = x.v[0] - kd * T(1.0 / 128.0);                /* exact */
  r = add<K>(r, from_scalar<1>(T(0)));
  const mwn<T, K> z = mul<K>(r, r);
  const mwn<T, K> s = mul<K, 0>(r, horner<PolySinh<S, T>>(z));
  const mwn<T, K> cm1 = mul<K, 0>(z, horner<PolyCosh<S, T>>(z));
  const T *Hk = Tab<T>::hy_tab(static_cast<int>(kd));
  if (want_s) sh = rot_recon<T, K>(Hk, T(1), Hk + MW_NW, T(1), s, cm1);
  if (want_c) ch = rot_recon<T, K>(Hk + MW_NW, T(1), Hk, T(1), s, cm1);
}

template <class S, class T>
MWF mwn<T, S::K> sinh(const mwn<T, S::K> &a) {
  constexpr int K = S::K;
  const bool ng = a.v[0] < T(0);
  const mwn<T, K> x = mwabs(a);
  if (isnan_(x.v[0])) return a;
  mwn<T, K> s, c;
  if (x.v[0] <= T(1)) {
    hy_small<S>(x, s, c, true, false);
    return neg_if(s, ng);
  }
  const T big = (sizeof(T) == 8) ? T(40) : T(20);
  if (x.v[0] > big) {
    if (x.v[0] > Lim<T>::EXP_OVF + T(0.7)) return mwinf<K, T>(ng);
    int m;
    const mwn<T, K> y = exp_kernel<S>(x, m);
    return neg_if(scale_pow2(y, m - 1), ng);
  }
  const mwn<T, K> E = exp<S>(x);
  const mwn<T, K> iE = (K == 2) ? div<K>(from_scalar<1>(T(1)), E) : exp<S>(neg(x));
  return neg_if(scale(add<K>(E, neg(iE)), T(0.5)), ng);
}
template <class S, class T>
MWF mwn<T, S::K> cosh(const mwn<T, S::K> &a) {
  constexpr int K = S::K;
  const mwn<T, K> x = mwabs(a);
  if (isnan_(x.v[0])) return a;
  mwn<T, K> s, c;
  if (x.v[0] <= T(1)) {
    hy_small<S>(x, s, c, false, true);
    return c;
  }
  const T big = (sizeof(T) == 8) ? T(40) : T(20);
  if (x.v[0] > big) {
    if (x.v[0] > Lim<T>::EXP_OVF + T(0.7)) return mwinf<K, T>();
    int m;
    const mwn<T, K> y = exp_kernel<S>(x, m);
    return scale_pow2(y, m - 1);
  }
  const mwn<T, K> E = exp<S>(x);
  const mwn<T, K> iE = (K == 2) ? div<K>(from_scalar<1>(T(1)), E) : exp<S>(neg(x));
  return scale(add_dom<K>(E, iE), T(0.5));
}
template <class S, class T>
MWF mwn<T, S::K> tanh(const mwn<T, S::K> &a) {
  constexpr int K = S::K;
  const bool ng = a.v[0] < T(0);
  const mwn<T, K> x = mwabs(a);
  if (isnan_(x.v[0])) return a;
  if (x.v[0] <= T(1)) {
    mwn<T, K> s, c;
    hy_small<S>(x, s, c, true, true);
    return neg_if(div<K>(s, c), ng);
  }
  const T big = (sizeof(T) == 8) ? T(40) * K : T(10) * K;
  if (x.v[0] > big) return neg_if(from_scalar<K>(T(1)), ng);
  /* 1 - 2 / (e^{2x} + 1) */
  const mwn<T, K> E2 = exp<S>(scale(x, T(2)));
  const mwn<T, K> t = div<K>(from_scalar<1>(T(2)), add_dom<K>(E2, from_scalar<1>(T(1))));
  return neg_if(add<K>(from_scalar<1>(T(1)), neg(t)), ng);
}

/* ================================================================== */
/* atan / atan2 / asin / acos                                          */
/* ================================================================== */
template <class S, class T>
struct AtanParts { int j; mwn<T, S::K> at; };

/* atan(b/a) = atan(c_j) + atan(t), 0 <= b <= a, a > 0 */
template <class S, class T>
MWF AtanParts<S, T> atan_core(const mwn<T, S::K> &b, const mwn<T, S::K> &a) {
  constexpr int K = S::K;
  AtanParts<S, T> o;
  o.j = static_cast<int>(rint_(T(256) * (b.v[0] / a.v[0])));
  const T c = static_cast<T>(o.j) * T(1.0 / 256.0);
  mwn<T, K> num, den;
  {
    LevelAcc<T, K + 1, 3 * K + 6> acc;
    MW_PRAGMA_UNROLL
    for (int i = 0; i < K; i++) {
      T e;
      const T p = two_prod(c, a.v[i], e);
      acc.push(i, b.v[i]);
      acc.push(i, -p);
      acc.push(i + 1, -e);
    }
    num = acc.template finish<K, 2>();
  }
  {
    LevelAcc<T, K + 1, 3 * K + 6> acc;
    MW_PRAGMA_UNROLL
    for (int i = 0; i < K; i++) {
      T e;
      const T p = two_prod(c, b.v[i], e);
      acc.push(i, a.v[i]);
      acc.push(i, p);
      acc.push(i + 1, e);
    }
    den = acc.template finish<K, 1>();
  }
  const mwn<T, K> t = div<K>(num, den);
  const mwn<T, K> z = mul<K>(t, t);
  o.at = mul<K, 0>(t, horner<PolyAtan<S, T>>(z));
  return o;
}

/* H + s (atan(c_j) + atan t),  H = hsel pi/2 */
template <class S, class T>
MWF mwn<T, S::K> atan_assemble(const AtanParts<S, T> &o, int hsel, T s) {
  constexpr int K = S::K;
  LevelAcc<T, K + 1, 3 * K + 8> acc;
  const T *A = Tab<T>::at_tab(o.j);
  const T *H = (hsel == 1) ? Tab<T>::pi2() : Tab<T>::pi();
  MW_PRAGMA_UNROLL
  for (int i = 0; i <= K; i++) {
    acc.push(i, (hsel != 0) ? H[i] : T(0));      /* unconditional push: no divergent code */
    acc.push(i, s * A[i]);
  }
  MW_PRAGMA_UNROLL
  for (int i = 0; i < K; i++) acc.push(i, s * o.at.v[i]);
  return acc.template finish<K, 2>();
}

template <class S, class T>
MWF mwn<T, S::K> atan2_finite(const mwn<T, S::K> &y, const mwn<T, S::K> &x) {
  const bool yneg = y.v[0] < T(0) || (y.v[0] == T(0) && signbit_(y.v[0]));
  const bool xneg = x.v[0] < T(0) || (x.v[0] == T(0) && signbit_(x.v[0]));
  const mwn<T, S::K> ay = yneg ? neg(y) : y, ax = xneg ? neg(x) : x;
  bool swap = ay.v[0] > ax.v[0];
  if (ay.v[0] == ax.v[0]) {
    const mwn<T, S::K> d = add<S::K>(ay, neg(ax));
    swap = d.v[0] > T(0);
  }
  /* select the operands instead of branching between two inlined copies of
     atan_core: on a GPU a warp with both cases would execute both copies
     (asin/acos on [-1,1] were 1.6-1.7x slower than on either half)      */
  mwn<T, S::K> bn, bd;
  MW_PRAGMA_UNROLL
  for (int i = 0; i < S::K; i++) { bn.v[i] = swap ? ax.v[i] : ay.v[i]; bd.v[i] = swap ? ay.v[i] : ax.v[i]; }
  const AtanParts<S, T> o = atan_core<S>(bn, bd);
  const int hsel = swap ? 1 : (xneg ? 2 : 0);
  const T s = (swap != xneg) ? T(-1) : T(1);
  return neg_if(atan_assemble<S>(o, hsel, s), yneg);
}

template <class S, class T>
MWF mwn<T, S::K> atan2(const mwn<T, S::K> &y, const mwn<T, S::K> &x) {
  constexpr int K = S::K;
  if (isnan_(x.v[0]) || isnan_(y.v[0])) return mwnan<K, T>();
  if (x.v[0] == T(0) && y.v[0] == T(0)) {
    const mwn<T, K> r = signbit_(x.v[0]) ? trunc<K>(load<K + 1>(Tab<T>::pi())) : from_scalar<K>(T(0));
    return signbit_(y.v[0]) ? neg(r) : r;
  }
  if (isinf_(x.v[0]) || isinf_(y.v[0])) {
    const T xs = isinf_(x.v[0]) ? (x.v[0] < 0 ? T(-1) : T(1)) : (signbit_(x.v[0]) ? T(-0.0) : T(0));
    const T ys = isinf_(y.v[0]) ? (y.v[0] < 0 ? T(-1) : T(1)) : (signbit_(y.v[0]) ? T(-0.0) : T(0));
    if (ys == T(0)) {
      const mwn<T, K> r = signbit_(xs) ? trunc<K>(load<K + 1>(Tab<T>::pi())) : from_scalar<K>(T(0));
      return signbit_(ys) ? neg(r) : r;
    }
    return atan2_finite<S>(from_scalar<K>(ys), from_scalar<K>(xs));
  }
  const T mx = abs_(x.v[0]) > abs_(y.v[0]) ? abs_(x.v[0]) : abs_(y.v[0]);
  if (mx > Lim<T>::BIG || mx < T(1) / Lim<T>::BIG) {
    int e;
    frexp_(mx, &e);
    return atan2_finite<S>(scale_pow2(y, -e), scale_pow2(x, -e));
  }
  return atan2_finite<S>(y, x);
}

template <class S, class T>
MWF mwn<T, S::K> atan(const mwn<T, S::K> &a) {
  constexpr int K = S::K;
  if (isnan_(a.v[0])) return a;
  const bool ng = a.v[0] < T(0);
  const mwn<T, K> x = mwabs(a);
  if (x.v[0] == T(0)) return a;
  if (isinf_(x.v[0])) return neg_if(trunc<K>(load<K + 1>(Tab<T>::pi2())), ng);
  if (x.v[0] > Lim<T>::BIG) {
    const mwn<T, K> ix = div<K>(from_scalar<1>(T(1)), x);
    return neg_if(add<K>(load<K + 1>(Tab<T>::pi2()), neg(ix)), ng);
  }
  /* |x| <= 1: atan(x / 1);  |x| > 1: pi/2 - atan(1 / x)  (operands selected,
     one code path) */
  const bool le = x.v[0] <= T(1);
  mwn<T, K> bn, bd;
  MW_PRAGMA_UNROLL
  for (int i = 0; i < K; i++) {
    const T one = (i == 0) ? T(1) : T(0);
    bn.v[i] = le ? x.v[i] : one;
    bd.v[i] = le ? one : x.v[i];
  }
  return neg_if(atan_assemble<S>(atan_core<S>(bn, bd), le ? 0 : 1, le ? T(1) : T(-1)), ng);
}

/* sqrt((1 - |x|)(1 + |x|)), accurate near |x| = 1 */
template <class S, class T>
MWF mwn<T, S::K> sqrt_one_minus_sq(const mwn<T, S::K> &ax) {
  constexpr int K = S::K;
  const mwn<T, K> d1 = add<K>(from_scalar<1>(T(1)), neg(ax));
  const mwn<T, K> d2 = add<K>(from_scalar<1>(T(1)), ax);
  return sqrt<K>(mul<K>(d1, d2));
}

template <class S, class T>
MWF mwn<T, S::K> asin(const mwn<T, S::K> &a) {
  constexpr int K = S::K;
  const bool ng = a.v[0] < T(0);
  const mwn<T, K> x = mwabs(a);
  if (!(x.v[0] <= T(1))) return mwnan<K, T>();
  if (x.v[0] == T(1)) {
    T l = T(0);
    for (int i = 1; i < K; i++) l += x.v[i];
    if (l > T(0)) return mwnan<K, T>();
    if (l == T(0)) return neg_if(trunc<K>(load<K + 1>(Tab<T>::pi2())), ng);
  }
  if (x.v[0] == T(0)) return a;
  return neg_if(atan2_finite<S>(x, sqrt_one_minus_sq<S>(x)), ng);
}
template <class S, class T>
MWF mwn<T, S::K> acos(const mwn<T, S::K> &a) {
  constexpr int K = S::K;
  const bool ng = a.v[0] < T(0);
  const mwn<T, K> x = mwabs(a);
  if (!(x.v[0] <= T(1))) return mwnan<K, T>();
  if (x.v[0] == T(1)) {
    T l = T(0);
    for (int i = 1; i < K; i++) l += x.v[i];
    if (l > T(0)) return mwnan<K, T>();
    if (l == T(0)) return ng ? trunc<K>(load<K + 1>(Tab<T>::pi())) : from_scalar<K>(T(0));
  }
  return atan2_finite<S>(sqrt_one_minus_sq<S>(x), a);
}

/* ================================================================== */
/* asinh / acosh / atanh / pow                                         */
/* ================================================================== */
template <class S, class T>
MWF mwn<T, S::K> asinh(const mwn<T, S::K> &a) {
  constexpr int K = S::K;
  const bool ng = a.v[0] < T(0);
  const mwn<T, K> x = mwabs(a);
  if (x.v[0] == T(0) || isnan_(x.v[0])) return a;
  if (!(x.v[0] < Lim<T>::SQRT_BIG)) {
    if (isinf_(x.v[0])) return a;
    mwn<T, K> xx = x, y;
    int eadd;
    log_prepare(xx, eadd, y);
    return neg_if(log_kernel<S, T, K>(xx, eadd + 1), ng);
  }
  const mwn<T, K> x2 = mul<K>(x, x);
  const mwn<T, K> q = sqrt<K>(add_dom<K>(x2.v[0] > T(1) ? x2 : from_scalar<K>(T(1)),
                                         x2.v[0] > T(1) ? from_scalar<K>(T(1)) : x2));
  if (x.v[0] <= T(1)) {
    /* log1p(|x| + x^2 / (1 + sqrt(1 + x^2))) */
    const mwn<T, K> t = div<K>(x2, add_dom<K>(q, from_scalar<1>(T(1))));
    return neg_if(log1p_n<S>(add<K + 1>(x, t)), ng);
  }
  return neg_if(log<S>(add<K>(x, q)), ng);
}

template <class S, class T>
MWF mwn<T, S::K> acosh(const mwn<T, S::K> &a) {
  constexpr int K = S::K;
  if (!(a.v[0] >= T(1))) return mwnan<K, T>();
  if (!(a.v[0] < Lim<T>::SQRT_BIG)) {
    if (isinf_(a.v[0])) return a;
    mwn<T, K> xx = a, y;
    int eadd;
    log_prepare(xx, eadd, y);
    return log_kernel<S, T, K>(xx, eadd + 1);
  }
  const mwn<T, K> t = add<K>(a, from_scalar<1>(T(-1)));      /* exact-ish */
  if (t.v[0] < T(0)) return mwnan<K, T>();
  if (t.v[0] == T(0)) return from_scalar<K>(T(0));
  if (a.v[0] < T(2)) {
    /* log1p(t + sqrt(2t + t^2)) */
    const mwn<T, K> p = add_dom<K>(scale(t, T(2)), mul<K>(t, t));
    return log1p_n<S>(add<K + 1>(t, sqrt<K>(p)));
  }
  const mwn<T, K> q = sqrt<K>(add<K>(mul<K>(a, a), from_scalar<1>(T(-1))));
  return log<S>(add<K>(a, q));
}

template <class S, class T>
MWF mwn<T, S::K> atanh(const mwn<T, S::K> &a) {
  constexpr int K = S::K;
  const bool ng = a.v[0] < T(0);
  const mwn<T, K> x = mwabs(a);
  if (x.v[0] == T(0) || isnan_(x.v[0])) return a;
  if (!(x.v[0] <= T(1))) return mwnan<K, T>();
  const mwn<T, K> d = add<K>(from_scalar<1>(T(1)), neg(x));   /* 1 - |x| */
  if (d.v[0] == T(0)) return mwinf<K, T>(ng);
  if (d.v[0] < T(0)) return mwnan<K, T>();
  const mwn<T, K> y = div<K>(scale(x, T(2)), d);
  return neg_if(scale(log1p_n<S>(y), T(0.5)), ng);
}

template <class S, class T, int KX = S::K>
MWF mwn<T, S::K> pow(const mwn<T, KX> &a, const mwn<T, KX> &b) {
  constexpr int K = S::K;
  const T ah = a.v[0], bh = b.v[0];
  if (bh == T(0)) return from_scalar<K>(T(1));
  bool a_one = (ah == T(1));
  for (int i = 1; i < KX; i++) a_one = a_one && (a.v[i] == T(0));
  if (a_one) return from_scalar<K>(T(1));
  if (isnan_(ah) || isnan_(bh)) return mwnan<K, T>();
  bool b_int = !isinf_(bh);
  for (int i = 0; i < KX; i++) b_int = b_int && (floor_(b.v[i]) == b.v[i]);
  bool b_odd = false;
  if (b_int) {
    const T lim = pow2i<T>(FP<T>::P);
    for (int i = KX - 1; i >= 0; i--) {
      if (b.v[i] != T(0) && abs_(b.v[i]) < lim) { b_odd = fmod_(b.v[i], T(2)) != T(0); break; }
    }
  }
  if (isinf_(bh)) {
    const mwn<T, K> m = add<K>(mwabs(a), from_scalar<1>(T(-1)));
    if (m.v[0] == T(0)) return from_scalar<K>(T(1));
    return ((m.v[0] > T(0)) == (bh > T(0))) ? mwinf<K, T>() : from_scalar<K>(T(0));
  }
  if (ah == T(0) || isinf_(ah)) {
    const bool big = isinf_(ah);
    const bool pos = (bh > T(0)) == big;
    const T r = pos ? inf_<T>() : T(0);
    return from_scalar<K>((signbit_(ah) && b_odd) ? -r : r);
  }
  bool ng = false;
  mwn<T, KX> aa = a;
  if (ah < T(0)) {
    if (!b_int) return mwnan<K, T>();
    ng = b_odd;
    aa = neg(a);
  }
  mwn<T, K> y;
  int eadd;
  log_prepare(aa, eadd, y);
  const mwn<T, K + 1> L = log_kernel<S, T, K + 1>(aa, eadd);
  const mwn<T, K + 1> Y = mul<K + 1>(L, b);
  if (isinf_(Y.v[0]) || isnan_(Y.v[0])) return from_scalar<K>(Y.v[0] > 0 ? inf_<T>() : T(0));
  return neg_if(exp_any<S>(Y), ng);
}

} // namespace mw

#endif
