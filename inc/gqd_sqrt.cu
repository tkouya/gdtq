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

        gqd_real r = make_qd((1.0 / sqrt(a.x)));
        gqd_real h = mul_pwr2(a, 0.5);

        r = r + ((0.5 - h * sqr(r)) * r);
        r = r + ((0.5 - h * sqr(r)) * r);
        r = r + ((0.5 - h * sqr(r)) * r);

        r = r * a;

        return r;
}

#endif /* __GQD_SQRT_CU__ */


