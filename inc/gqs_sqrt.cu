/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GQS_SQRT_CU__
#define __GQS_SQRT_CU__


//#include "common.cuh"
//#include "gqd_sqrt.cuh"
#include "gqs.cuh"

__device__
gqs_real sqrt(const gqs_real &a) {
        if (is_zero(a))
                return make_qs(0.0);

        //!!!!!!!!!!
        if (is_negative(a)) {
                //TO DO: should return an error
                //return _nan;
                return make_qs(0.0);
        }

        /* Each Newton step runs at the cheapest precision that can hold
           its result: ~7 -> ~14 digits in gds_real, ~14 -> ~21 in
           gts_real, and the final step in gqs_real.  Every step is a
           pair of branch-free FMAs.  The 0.0.2 version ran three full
           quad-single iterations built from separate multiplies and
           adds. */
        gds_real ad = make_ds(a.x, a.y);
        gds_real hd = mul_pwr2(ad, 0.5f);
        gds_real rd = make_ds(1.0f / sqrt(a.x));
        rd = dw_fma(dw_fma(negative(hd), sqr(rd), make_ds(0.5f)), rd, rd);

        gts_real at = make_ts(a.x, a.y, a.z);
        gts_real ht = mul_pwr2(at, 0.5f);
        gts_real rt = make_ts(rd.x, rd.y, 0.0f);
        rt = tw_fma(tw_fma(negative(ht), sqr(rt), make_ts(0.5f)), rt, rt);

        gqs_real h = mul_pwr2(a, 0.5f);
        gqs_real r = make_qs(rt.x, rt.y, rt.z, 0.0f);
        r = qw_fma(qw_fma(negative(h), sqr(r), make_qs(0.5f)), r, r);

        r = r * a;

        return r;
}

/* Reference (0.0.2) square root, kept for benchmarking. */
__device__
gqs_real sqrt_legacy(const gqs_real &a) {
        if (is_zero(a))
                return make_qs(0.0);

        if (is_negative(a))
                return make_qs(0.0);

        gqs_real r = make_qs((1.0 / sqrt(a.x)));
        gqs_real h = mul_pwr2(a, 0.5);

        r = r + ((0.5 - h * sqr(r)) * r);
        r = r + ((0.5 - h * sqr(r)) * r);
        r = r + ((0.5 - h * sqr(r)) * r);

        r = r * a;

        return r;
}

#endif /* __GQS_SQRT_CU__ */


