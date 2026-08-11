#ifndef __GDS_LOG_CU__
#define __GDS_LOG_CU__

#include <math_constants.h>

#include "gqs.cuh"

/* Logarithm.  Computes log(x) in float-float precision.
   This is a natural logarithm (i.e., base e).

   Newton iteration on f(x) = exp(x) - a:
       x' = x + a * exp(-x) - 1. */
__device__
gds_real log(const gds_real &a) {

	if (is_one(a)) {
		return make_ds(0.0);
	}

	/* MPFR/IEEE semantics, silently: log(0) = -inf, log(negative) = nan. */
	if (is_zero(a)) {
		return make_ds(-CUDART_INF_F);
	}

	if (a.x < 0.0f) {
		return make_ds(CUDART_NAN_F);
	}

	gds_real x = make_ds(log(a.x));   // Initial approximation

	x = x + a * exp(negative(x)) - 1.0;

	return x;
}

/* Base-10 logarithm. */
__device__
gds_real log10(const gds_real &a)
{
	return log(a) / _ds_log10;
}

#endif /* __GDS_LOG_CU__ */
