#ifndef __GQD_LOG_CU__
#define __GQD_LOG_CU__

#include <math_constants.h>

#include "gqd.cuh"

/* log(a) by Newton's method on f(x) = exp(x) - a:
 *   x_{n+1} = x_n + a * exp(-x_n) - 1
 * Newton doubles the number of correct digits per iteration, so start
 * from the ~32-digit double-double logarithm (one gdd_real exp) instead
 * of the 16-digit double one: a single quad-double iteration -- one
 * gqd_real exp instead of three -- then reaches full precision. */
__device__
gqd_real log(const gqd_real &a) {
        if (is_one(a)) {
                return make_qd(0.0);
        }

        /* MPFR/IEEE semantics, silently: log(0) = -inf, log(negative) = nan.
           Adaptive-quadrature singularity detectors legitimately evaluate
           log of an exactly-zero error estimate, so neither case may return
           nan for zero. */
        if (is_zero(a)) {
                return make_qd(-CUDART_INF);
        }

        if (a.x < 0.0) {
                return make_qd(CUDART_NAN);
        }

        gdd_real x0 = log(make_dd(a.x, a.y));
        gqd_real x = make_qd(x0.x, x0.y, 0.0, 0.0);

        x = x + a * exp(negative(x)) - 1.0;

        return x;
}


/* Base-10 logarithm. */
__device__
gqd_real log10(const gqd_real &a)
{
	return log(a) / _qd_log10;
}

#endif /* __GQD_LOG_CU__ */
