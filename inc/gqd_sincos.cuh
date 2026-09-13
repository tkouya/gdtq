#ifndef __GQD_SIN_COS_CUH__
#define __GQD_SIN_COS_CUH__

//#include "common.cuh"
//#include "gqd_sincos.cuh"

__device__
void sincos_taylor(const gqd_real &a, 
				   gqd_real &sin_a, gqd_real &cos_a);

__device__
gqd_real sin_taylor(const gqd_real &a);

__device__
gqd_real cos_taylor(const gqd_real &a);

__device__
gqd_real sin(const gqd_real &a);

__device__
gqd_real cos(const gqd_real &a);

__device__
void sincos(const gqd_real &a, gqd_real &sin_a, gqd_real &cos_a);

__device__
gqd_real tan(const gqd_real &a);

/* always available (gdtq 0.0.4: mw kernels, see gqd_elem.cu).  A
   GDTQ_LEGACY_ELEMENTARY build uses the 0.0.3 versions of these when
   0.0.3 compiled them, i.e. with ALL_MATH (gqd_type.h). */

__device__
gqd_real atan2(const gqd_real &y, const gqd_real &x);

__device__
gqd_real atan(const gqd_real &a);

__device__
gqd_real asin(const gqd_real &a);

__device__
gqd_real acos(const gqd_real &a);

__device__
gqd_real sinh(const gqd_real &a);

__device__
gqd_real cosh(const gqd_real &a);

__device__
gqd_real tanh(const gqd_real &a);

__device__
void sincosh(const gqd_real &a, gqd_real &s, gqd_real &c);

__device__
gqd_real asinh(const gqd_real &a);

__device__
gqd_real acosh(const gqd_real &a);

__device__
gqd_real atanh(const gqd_real &a);



#endif /* __GQD_SIN_COS_CUH__ */


