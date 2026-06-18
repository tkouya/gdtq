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

        gqs_real r = make_qs((1.0 / sqrt(a.x)));
        gqs_real h = mul_pwr2(a, 0.5);

        r = r + ((0.5 - h * sqr(r)) * r);
        r = r + ((0.5 - h * sqr(r)) * r);
        r = r + ((0.5 - h * sqr(r)) * r);

        r = r * a;

        return r;
}

#endif /* __GQS_SQRT_CU__ */


