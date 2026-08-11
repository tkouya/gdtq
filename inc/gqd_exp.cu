/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GQD_EXP_CU__
#define __GQD_EXP_CU__

#include <math_constants.h>

#include "gqd.cuh"

__device__
gqd_real exp( const gqd_real &a ) {
        /* Two-level table-driven reduction (same idea as the sin/cos tables):

               a = m log 2 + j1/64 + j2/8192 + r,   |r| <= 2^-14 + eps,

           so  exp(a) = 2^m * exp(j1/64) * exp(j2/8192) * exp(r), where the two
           table factors are precomputed and exp(r) needs only a short Taylor
           series (~11 terms).  j1/64 and j2/8192 are exact binary values, so
           removing them from r is exact.  This replaces the previous
           |kr| <= 0.347/2^16 reduction whose ladder of 16 repeated squarings
           s <- 2s + s^2 dominated the cost of exp().                          */

        if (a.x <= -709.0)
                return make_qd(0.0);

        if (a.x >=  709.0)
                return make_qd(CUDART_INF);

        if (is_zero(a))
                return make_qd(1.0);

        if (is_one(a)) {
                gqd_real r;
                r.x = 2.718281828459045091e+00;
                r.y = 1.445646891729250158e-16;
                r.z = -2.127717108038176765e-33;
                r.w = 1.515630159841218954e-49;
                return r;
        }

        double m = floor(a.x / _qd_log2.x + 0.5);
        gqd_real r = a - _qd_log2 * m;

        int j1 = (int)(floor(r.x * 64.0 + 0.5));
        r = r - j1 * (1.0 / 64.0);                  /* exact */
        int j2 = (int)(floor(r.x * 8192.0 + 0.5));
        r = r - j2 * (1.0 / 8192.0);                /* exact */

        gqd_real s, p;
        double thresh = _qd_eps;
        double tmag;

        p = sqr(r);
        s = 1.0 + r + mul_pwr2(p, 0.5);
        int i = 0;
        do {
                p = p * r;
                /* s += p / i!, fused; the term magnitude for the convergence
                   test is estimated from the leading words. */
                tmag = fabs(to_double(p) * inv_fact[i].x);
                s = qw_fma(p, inv_fact[i], s);
                ++i;
        } while (tmag > thresh && i < 13);

        s = s * d_exp_table_64[j1 + 23];
        s = s * d_exp_table_8192[j2 + 65];

        return ldexp(s, int(m));
}


/* exp(a) - 1, accurate for small |a|.
   For |a| <= log(2)/2 the ln2 reduction in exp() is a no-op (m = 0), and
   the repeated-squaring ladder  s <- 2s + s^2  is exactly the doubling
   rule for u = exp(t) - 1.  So expm1 is exp() without the final +1.
   For larger |a|, exp(a) - 1 suffers no cancellation.                  */
__device__
gqd_real expm1( const gqd_real &a ) {
        const double k = ldexp(1.0, 16);
        const double inv_k = 1.0 / k;

        if (a.x <= -709.0)
                return make_qd(-1.0);

        if (a.x >= 709.0)
                return make_qd(CUDART_INF);

        if (is_zero(a))
                return make_qd(0.0);

        if (fabs(a.x) > 0.346573590279972655)   /* log(2)/2 */
                return exp(a) - 1.0;

        gqd_real r = mul_pwr2(a, inv_k);
        gqd_real s, p;
        double thresh = inv_k * _qd_eps;
        double tmag;

        p = sqr(r);
        s = r + mul_pwr2(p, 0.5);
        int i = 0;
        do {
                p = p * r;
                tmag = fabs(to_double(p) * inv_fact[i].x);
                s = qw_fma(p, inv_fact[i], s);
                ++i;
        } while (tmag > thresh && i < 9);

        for (int j = 0; j < 16; j++)
                s = mul_pwr2(s, 2.0) + sqr(s);

        return s;      /* = exp(a) - 1, no +1 */
}


#endif /* __GQD_EXP_CU__ */
