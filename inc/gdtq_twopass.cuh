/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. */
/*
 * gdtq_twopass.cuh -- array evaluation of the elementary functions with
 *                     the two-pass / partitioned schemes (gdtq 0.0.4,
 *                     optional, header-only).
 *
 *   #include "gdtq_twopass.cuh"
 *   gdtq::Work w;                                   // one per stream
 *   gdtq::eval<gdtq::SIN>(d_x, nullptr, d_y, n, w); // y[i] = sin(x[i])
 *   gdtq::eval<gdtq::POW>(d_a, d_b, d_y, n, w);     // y[i] = a[i]^b[i]
 *
 * The arrays live in device memory and hold any of the six classes.  The
 * results are bitwise identical to calling the scalar functions of
 * gqd_elem.cu / gqs_elem.cu element by element; only the scheduling
 * differs (see mw/mw_twopass.cuh):
 *
 *   SINGLE     one thread per element (what a plain kernel does);
 *   TWO_PASS   pass 1 evaluates the common inputs without the rare
 *              expensive branch and compacts the others, pass 2 evaluates
 *              the compacted list: sin/cos/tan with a few huge (Payne-Hanek)
 *              arguments, 1.5-19x faster on the GB10;
 *   PARTITION  both input classes are compacted before one evaluation, for
 *              functions with two common algorithms (sinh, cosh, tanh,
 *              expm1, asinh, acosh): ~1.9x for double words on GPUs with
 *              weak FP64.
 *
 * default_mode() picks the scheme from the FP32:FP64 throughput ratio of
 * the current device, as measured on the GB10 and the H100.  eval() without
 * a mode argument uses it.  All kernels are templates, instantiated in the
 * including translation unit (compile with -fmad=false, C++17).
 */
#ifndef __GDTQ_TWOPASS_CUH__
#define __GDTQ_TWOPASS_CUH__

#include "gdtq_mw.cuh"
#pragma nv_diagnostic push
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20091
#include "mw/mw_twopass.cuh"
#pragma nv_diagnostic pop

namespace gdtq {

using mw2p::Work;
using mw2p::Mode;
using mw2p::SINGLE;
using mw2p::TWO_PASS;
using mw2p::PARTITION;

enum Op {
  EXP = mw2p::EXP, EXPM1 = mw2p::EXPM1, LOG = mw2p::LOG, LOG10 = mw2p::LOG10,
  LOG1P = mw2p::LOG1P, SIN = mw2p::SIN, COS = mw2p::COS, TAN = mw2p::TAN,
  ASIN = mw2p::ASIN, ACOS = mw2p::ACOS, ATAN = mw2p::ATAN, SINH = mw2p::SINH,
  COSH = mw2p::COSH, TANH = mw2p::TANH, ASINH = mw2p::ASINH, ACOSH = mw2p::ACOSH,
  ATANH = mw2p::ATANH, ATAN2 = mw2p::ATAN2, POW = mw2p::POW
};

/* suggested scheme for op OP on class G on the current device */
template <int OP, class G>
Mode default_mode() {
  typedef gdtq_mw::Tr<G> M;
  return mw2p::default_mode<OP, typename M::S, typename M::T>();
}

/* y[i] = f(a[i]) or f(a[i], b[i]) (ATAN2: f(y = a[i], x = b[i]), POW:
   a[i]^b[i]), 0 <= i < n.  b may be nullptr for the unary functions. */
template <int OP, class G>
void eval(Mode mode, const G *a, const G *b, G *y, int n, Work &w, cudaStream_t st = 0,
          int block = 128) {
  typedef gdtq_mw::Tr<G> M;
  typedef mw::mwn<typename M::T, M::K> V;
  static_assert(sizeof(V) == sizeof(G), "layout of the gdtq vector type");
  mw2p::eval<OP, typename M::S, typename M::T>(mode, reinterpret_cast<const V *>(a),
                                                reinterpret_cast<const V *>(b),
                                                reinterpret_cast<V *>(y), n, w, st, block);
}
template <int OP, class G>
void eval(const G *a, const G *b, G *y, int n, Work &w, cudaStream_t st = 0, int block = 128) {
  eval<OP, G>(default_mode<OP, G>(), a, b, y, n, w, st, block);
}

} // namespace gdtq

#endif /* __GDTQ_TWOPASS_CUH__ */
