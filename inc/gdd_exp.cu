/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GDD_EXP_CU__
#define __GDD_EXP_CU__

#include <math_constants.h>

#include "gqd.cuh"

#if defined(GDTQ_LEGACY_ELEMENTARY)  /* 0.0.3 implementation; default: mw kernels in gqd_elem.cu */

#define INV_K (1.0/512.0)

/* Strategy:  We first reduce the size of x by noting that

        exp(kr + m * log(2)) = 2^m * exp(r)^k

   where m and k are integers.  By choosing m appropriately
   we can make |kr| <= log(2) / 2 = 0.347.  Then exp(r) is
   evaluated using the familiar Taylor series.  Reducing the
   argument substantially speeds up the convergence.       */
__device__
gdd_real exp(const gdd_real &a) {

        if (a.x <= -709.0)
                return make_dd(0.0);

        if (a.x >=  709.0)
                return make_dd(CUDART_INF);

        if (is_zero(a))
                return make_dd(1.0);

        if (is_one(a))
                return _dd_e;

        double m = floor(a.x / _dd_log2.x + 0.5);
        gdd_real r = mul_pwr2(a - _dd_log2 * m, INV_K);
        gdd_real s, p;

        p = sqr(r);
        s = r + mul_pwr2(p, 0.5);
        p = p * r;
        /* Each term  s += p / i!  is one fused multiply-add; the convergence
           test uses the leading-word product, so the term itself need not be
           materialized. */
        const double thresh = INV_K * _dd_eps;
        double tmag;
        int i = 0;
        do {
                tmag = fabs(to_double(p) * dd_inv_fact[i].x);
                s = dw_fma(p, dd_inv_fact[i], s);
                p = p * r;
                ++i;
        } while (tmag > thresh && i < 6);

        /* Use a different loop variable name to avoid shadowing the outer 'i'
           (triggers -Wshadow with recent nvcc/host compilers). */
        for( int j = 0; j < 9; j++ ) {
                s = mul_pwr2(s, 2.0) + sqr(s);
        }
        s = s + 1.0;

        return ldexp(s, int(m));
}


/* exp(a) - 1, accurate for small |a|.
   For |a| <= log(2)/2 the ln2 reduction in exp() is a no-op (m = 0), and
   the repeated-squaring ladder  s <- 2s + s^2  is exactly the doubling
   rule for u = exp(t) - 1.  So expm1 is exp() without the final +1.
   For larger |a|, exp(a) - 1 suffers no cancellation.                  */
__device__
gdd_real expm1(const gdd_real &a) {

        if (a.x <= -709.0)
                return make_dd(-1.0);

        if (a.x >= 709.0)
                return make_dd(CUDART_INF);

        if (is_zero(a))
                return make_dd(0.0);

        if (fabs(a.x) > 0.346573590279972655)   /* log(2)/2 */
                return exp(a) - 1.0;

        gdd_real r = mul_pwr2(a, INV_K);
        gdd_real s, p;

        p = sqr(r);
        s = r + mul_pwr2(p, 0.5);
        p = p * r;
        /* Fused term accumulation, as in exp() above. */
        const double thresh = INV_K * _dd_eps;
        double tmag;
        int i = 0;
        do {
                tmag = fabs(to_double(p) * dd_inv_fact[i].x);
                s = dw_fma(p, dd_inv_fact[i], s);
                p = p * r;
                ++i;
        } while (tmag > thresh && i < 6);

        for( int j = 0; j < 9; j++ ) {
                s = mul_pwr2(s, 2.0) + sqr(s);
        }

        return s;      /* = exp(a) - 1, no +1 */
}


#endif /* GDTQ_LEGACY_ELEMENTARY */

#endif /* __GDD_EXP_CU__ */
