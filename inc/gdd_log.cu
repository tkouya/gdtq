#ifndef __GDD_LOG_CU__
#define __GDD_LOG_CU__

#include <math_constants.h>

#include "gqd.cuh"

#if defined(GDTQ_LEGACY_ELEMENTARY)  /* 0.0.3 implementation; default: mw kernels in gqd_elem.cu */

/* Logarithm.  Computes log(x) in double-double precision.
   This is a natural logarithm (i.e., base e).

   Strategy: the Taylor series for log converges much more slowly
   than that of exp, so this routine determines the root of the
   function  f(x) = exp(x) - a  by Newton iteration

       x' = x + a * exp(-x) - 1.

   Only one iteration is needed, since Newton's iteration
   approximately doubles the number of digits per iteration. */
__device__
gdd_real log(const gdd_real &a) {

	if (is_one(a)) {
		return make_dd(0.0);
	}

	/* MPFR/IEEE semantics, silently: log(0) = -inf, log(negative) = nan.
	   Adaptive-quadrature singularity detectors legitimately evaluate
	   log of an exactly-zero error estimate, so neither case may return
	   nan for zero. */
	if (is_zero(a)) {
		return make_dd(-CUDART_INF);
	}

	if (a.x < 0.0) {
		return make_dd(CUDART_NAN);
	}

	gdd_real x = make_dd(log(a.x));   // Initial approximation

	x = x + a * exp(negative(x)) - 1.0;

	return x;
}

/* Base-10 logarithm. */
__device__
gdd_real log10(const gdd_real &a)
{
	return log(a) / _dd_log10;
}

#endif /* GDTQ_LEGACY_ELEMENTARY */

#endif /* __GDD_LOG_CU__ */
