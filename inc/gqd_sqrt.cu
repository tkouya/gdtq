/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GQD_SQRT_CU__
#define __GQD_SQRT_CU__


//#include "common.cuh"
//#include "gqd_sqrt.cuh"
#include "gqd.cuh"

__device__
gqd_real sqrt(const gqd_real &a) {
        if (is_zero(a))
                return make_qd(0.0);

        //!!!!!!!!!!
        if (is_negative(a)) {
                //TO DO: should return an error
                //return _nan;
                return make_qd(0.0);
        }

        /* Newton doubles the number of correct digits at every step, so
           each step runs at the cheapest precision that can hold its
           result: ~16 -> ~32 digits in gdd_real, ~32 -> ~47 in
           gtd_real, and the final step in gqd_real.  Every step is a
           pair of branch-free FMAs, so neither h*r^2 nor the following
           multiply-and-add is renormalized on its own.  The 0.0.2
           version ran three full quad-double iterations built from
           separate multiplies and adds. */
        gdd_real ad = make_dd(a.x, a.y);
        gdd_real hd = mul_pwr2(ad, 0.5);
        gdd_real rd = make_dd(1.0 / sqrt(a.x));
        rd = dw_fma(dw_fma(negative(hd), sqr(rd), make_dd(0.5)), rd, rd);

        gtd_real at = make_td(a.x, a.y, a.z);
        gtd_real ht = mul_pwr2(at, 0.5);
        gtd_real rt = make_td(rd.x, rd.y, 0.0);
        rt = tw_fma(tw_fma(negative(ht), sqr(rt), make_td(0.5)), rt, rt);

        gqd_real h = mul_pwr2(a, 0.5);
        gqd_real r = make_qd(rt.x, rt.y, rt.z, 0.0);
        r = qw_fma(qw_fma(negative(h), sqr(r), make_qd(0.5)), r, r);

        r = r * a;

        return r;
}

/* Reference (0.0.2) square root, kept for benchmarking. */
__device__
gqd_real sqrt_legacy(const gqd_real &a) {
        if (is_zero(a))
                return make_qd(0.0);

        if (is_negative(a))
                return make_qd(0.0);

        gqd_real r = make_qd((1.0 / sqrt(a.x)));
        gqd_real h = mul_pwr2(a, 0.5);

        r = r + ((0.5 - h * sqr(r)) * r);
        r = r + ((0.5 - h * sqr(r)) * r);
        r = r + ((0.5 - h * sqr(r)) * r);

        r = r * a;

        return r;
}

#endif /* __GQD_SQRT_CU__ */


