/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. */
/*
 * gdtq_mw.cuh -- adapters between the gdtq vector types and the generic
 *                multi-word expansions of the elementary-function kernels
 *                in mw/ (gdtq 0.0.4).
 *
 * mw/mw_arith.h   expansion arithmetic with order-level error-free
 *                 accumulation (register cascade on the GPU)
 * mw/mw_elem.h    exp, expm1, log, log10, log1p, sin, cos, tan, sincos,
 *                 asin, acos, atan, atan2, sinh, cosh, tanh, asinh, acosh,
 *                 atanh, pow: table-based reduction ((K+1)-word tables,
 *                 Cody-Waite and Payne-Hanek), mixed-width Horner schemes
 *                 generated from an error budget, accurate reconstruction
 * mw/mw_tables.h  tables and per-type schedules (generated)
 * mw/mw_twopass.cuh  two-pass / partitioned array evaluation (see
 *                 gdtq_twopass.cuh)
 *
 * The same kernels serve all six classes: double words for gdd/gtd/gqd
 * (schedules MWS_DD/TD/QD), float words for gds/gts/gqs (MWS_DS/TS/QS;
 * only the rare Payne-Hanek reduction uses double arithmetic).
 * The headers are identical to the benchmark repository's versions except
 * that the table linkage can be preset (below).  They need C++17 (the nvcc
 * default since CUDA 11) and, like the rest of gdtq, -fmad=false.
 */
#ifndef __GDTQ_MW_CUH__
#define __GDTQ_MW_CUH__

#include "gqd_type.h"

/* Every translation unit keeps its own copy of the tables, so that the
   two objects of libgqd.a (gqd.cu and gqs.cu) link together. */
#ifndef MW_TAB
#define MW_TAB static __device__ const
#define MW_CTAB static __device__ const
#endif
/* The kernels are __host__ __device__ templates; their host instantiations
   are never called here, but nvcc warns about the device tables and the
   constexpr schedule functions they refer to (#20091, #20013). */
#pragma nv_diagnostic push
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20091
#include "mw/mw_elem.h"
#pragma nv_diagnostic pop

namespace gdtq_mw {

using mw::mwn;

template <class G> struct Tr;
template <> struct Tr<gdd_real> { typedef double T; typedef MWS_DD S; static constexpr int K = 2; };
template <> struct Tr<gtd_real> { typedef double T; typedef MWS_TD S; static constexpr int K = 3; };
template <> struct Tr<gqd_real> { typedef double T; typedef MWS_QD S; static constexpr int K = 4; };
template <> struct Tr<gds_real> { typedef float T; typedef MWS_DS S; static constexpr int K = 2; };
template <> struct Tr<gts_real> { typedef float T; typedef MWS_TS S; static constexpr int K = 3; };
template <> struct Tr<gqs_real> { typedef float T; typedef MWS_QS S; static constexpr int K = 4; };

#define GDTQ_MW_CONV2(G, T)                                                      \
  __host__ __device__ __forceinline__ mwn<T, 2> to_mw(const G &a) {              \
    mwn<T, 2> r; r.v[0] = a.x; r.v[1] = a.y; return r;                           \
  }                                                                              \
  __host__ __device__ __forceinline__ G from_mw(const mwn<T, 2> &r) {            \
    G a; a.x = r.v[0]; a.y = r.v[1]; return a;                                   \
  }
#define GDTQ_MW_CONV3(G, T)                                                      \
  __host__ __device__ __forceinline__ mwn<T, 3> to_mw(const G &a) {              \
    mwn<T, 3> r; r.v[0] = a.x; r.v[1] = a.y; r.v[2] = a.z; return r;             \
  }                                                                              \
  __host__ __device__ __forceinline__ G from_mw(const mwn<T, 3> &r) {            \
    G a; a.x = r.v[0]; a.y = r.v[1]; a.z = r.v[2]; return a;                     \
  }
#define GDTQ_MW_CONV4(G, T)                                                      \
  __host__ __device__ __forceinline__ mwn<T, 4> to_mw(const G &a) {              \
    mwn<T, 4> r; r.v[0] = a.x; r.v[1] = a.y; r.v[2] = a.z; r.v[3] = a.w;         \
    return r;                                                                    \
  }                                                                              \
  __host__ __device__ __forceinline__ G from_mw(const mwn<T, 4> &r) {            \
    G a; a.x = r.v[0]; a.y = r.v[1]; a.z = r.v[2]; a.w = r.v[3]; return a;       \
  }
GDTQ_MW_CONV2(gdd_real, double)
GDTQ_MW_CONV3(gtd_real, double)
GDTQ_MW_CONV4(gqd_real, double)
GDTQ_MW_CONV2(gds_real, float)
GDTQ_MW_CONV3(gts_real, float)
GDTQ_MW_CONV4(gqs_real, float)
#undef GDTQ_MW_CONV2
#undef GDTQ_MW_CONV3
#undef GDTQ_MW_CONV4

} // namespace gdtq_mw

/* definitions of the public functions for one class G (used by
   gqd_elem.cu and gqs_elem.cu) */
#define GDTQ_MW_UNARY(G, F)                                                      \
  __device__ G F(const G &a) {                                                   \
    typedef gdtq_mw::Tr<G> M;                                                    \
    return gdtq_mw::from_mw(mw::F<M::S, M::T>(gdtq_mw::to_mw(a)));               \
  }
#define GDTQ_MW_BINARY(G, F)                                                     \
  __device__ G F(const G &a, const G &b) {                                       \
    typedef gdtq_mw::Tr<G> M;                                                    \
    return gdtq_mw::from_mw(mw::F<M::S, M::T>(gdtq_mw::to_mw(a), gdtq_mw::to_mw(b))); \
  }

/* the functions of gdtq 0.0.3: exp expm1 log log10 sin cos tan sincos */
#define GDTQ_MW_BASIC(G)                                                         \
  GDTQ_MW_UNARY(G, exp) GDTQ_MW_UNARY(G, expm1)                                  \
  GDTQ_MW_UNARY(G, log) GDTQ_MW_UNARY(G, log10)                                  \
  GDTQ_MW_UNARY(G, sin) GDTQ_MW_UNARY(G, cos) GDTQ_MW_UNARY(G, tan)             \
  __device__ void sincos(const G &a, G &s, G &c) {                               \
    typedef gdtq_mw::Tr<G> M;                                                    \
    mw::mwn<M::T, M::K> ms, mc;                                                  \
    mw::sincos<M::S, M::T>(gdtq_mw::to_mw(a), ms, mc);                           \
    s = gdtq_mw::from_mw(ms); c = gdtq_mw::from_mw(mc);                          \
  }
/* the "advanced" functions (0.0.3: DD/DS by default, QD/QS with ALL_MATH) */
#define GDTQ_MW_ADVANCED(G)                                                      \
  GDTQ_MW_BINARY(G, atan2) GDTQ_MW_UNARY(G, atan)                                \
  GDTQ_MW_UNARY(G, asin) GDTQ_MW_UNARY(G, acos)                                  \
  GDTQ_MW_UNARY(G, sinh) GDTQ_MW_UNARY(G, cosh) GDTQ_MW_UNARY(G, tanh)          \
  GDTQ_MW_UNARY(G, asinh) GDTQ_MW_UNARY(G, acosh) GDTQ_MW_UNARY(G, atanh)       \
  __device__ void sincosh(const G &a, G &s, G &c) {                              \
    typedef gdtq_mw::Tr<G> M;                                                    \
    const mw::mwn<M::T, M::K> x = gdtq_mw::to_mw(a);                             \
    s = gdtq_mw::from_mw(mw::sinh<M::S, M::T>(x));                               \
    c = gdtq_mw::from_mw(mw::cosh<M::S, M::T>(x));                               \
  }
/* new in gdtq 0.0.4 for every class */
#define GDTQ_MW_NEW(G)                                                           \
  GDTQ_MW_UNARY(G, log1p) GDTQ_MW_BINARY(G, pow)

#endif /* __GDTQ_MW_CUH__ */
