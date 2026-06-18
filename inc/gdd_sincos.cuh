#ifndef __GDD_SIN_COS_CUH__
#define __GDD_SIN_COS_CUH__

//#include "common.cuh"

/* Computes sin(a) using Taylor series.
   Assumes |a| <= pi/32.                           */
__device__
gdd_real sin_taylor(const gdd_real &a);

__device__
gdd_real cos_taylor(const gdd_real &a);

__device__
void sincos_taylor(const gdd_real &a, gdd_real &sin_a, gdd_real &cos_a);

__device__
gdd_real sin(const gdd_real &a);

__device__
gdd_real cos(const gdd_real &a);

__device__
void sincos(const gdd_real &a, gdd_real &sin_a, gdd_real &cos_a);

__device__
gdd_real tan(const gdd_real &a);


#ifndef ALL_MATH

__device__
gdd_real atan2(const gdd_real &y, const gdd_real &x);

__device__
gdd_real atan(const gdd_real &a);

__device__
gdd_real asin(const gdd_real &a);

__device__
gdd_real acos(const gdd_real &a);

__device__
gdd_real sinh(const gdd_real &a);

__device__
gdd_real cosh(const gdd_real &a);

__device__
gdd_real tanh(const gdd_real &a);

__device__
void sincosh(const gdd_real &a, gdd_real &sinh_a, gdd_real &cosh_a);

__device__
gdd_real asinh(const gdd_real &a);

__device__
gdd_real acosh(const gdd_real &a);

__device__
gdd_real atanh(const gdd_real &a);

#endif /* ALL_MATH */


#endif /* __GDD_SIN_COS_CUH__ */


