#ifndef __GQS_LOG_CU__
#define __GQS_LOG_CU__

#include <math_constants.h>

#include "gqs.cuh"

/* log(a) by Newton's method on f(x) = exp(x) - a:
 *   x_{n+1} = x_n + a * exp(-x_n) - 1
 * Newton doubles the number of correct digits per iteration, so start
 * from the ~14-digit float-float logarithm (one gds_real exp) instead
 * of the 7-digit float one: a single quad-float iteration -- one
 * gqs_real exp instead of three -- then reaches full precision. */
__device__
gqs_real log(const gqs_real &a) {
        if (is_one(a)) {
                return make_qs(0.0);
        }

        /* MPFR/IEEE semantics, silently: log(0) = -inf, log(negative) = nan. */
        if (is_zero(a)) {
                return make_qs(-CUDART_INF_F);
        }

        if (a.x < 0.0f) {
                return make_qs(CUDART_NAN_F);
        }

        gds_real x0 = log(make_ds(a.x, a.y));
        gqs_real x = make_qs(x0.x, x0.y, 0.0f, 0.0f);

        x = x + a * exp(negative(x)) - 1.0;

        return x;
}


/* Base-10 logarithm. */
__device__
gqs_real log10(const gqs_real &a)
{
	return log(a) / _qs_log10;
}

#endif /* __GQS_LOG_CU__ */
