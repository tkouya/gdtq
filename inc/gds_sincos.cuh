#ifndef __GDS_SIN_COS_CUH__
#define __GDS_SIN_COS_CUH__

//#include "common.cuh"

/* Computes sin(a) using Taylor series.
   Assumes |a| <= pi/32.                           */
__device__
gds_real sin_taylor(const gds_real &a);

__device__
gds_real cos_taylor(const gds_real &a);

__device__
void sincos_taylor(const gds_real &a, gds_real &sin_a, gds_real &cos_a);

__device__
gds_real sin(const gds_real &a);

__device__
gds_real cos(const gds_real &a);

__device__
void sincos(const gds_real &a, gds_real &sin_a, gds_real &cos_a);

__device__
gds_real tan(const gds_real &a);


#ifndef ALL_MATH

__device__
gds_real atan2(const gds_real &y, const gds_real &x);

__device__
gds_real atan(const gds_real &a);

__device__
gds_real asin(const gds_real &a);

__device__
gds_real acos(const gds_real &a);

__device__
gds_real sinh(const gds_real &a);

__device__
gds_real cosh(const gds_real &a);

__device__
gds_real tanh(const gds_real &a);

__device__
void sincosh(const gds_real &a, gds_real &sinh_a, gds_real &cosh_a);

__device__
gds_real asinh(const gds_real &a);

__device__
gds_real acosh(const gds_real &a);

__device__
gds_real atanh(const gds_real &a);

#endif /* ALL_MATH */


#endif /* __GDD_SIN_COS_CUH__ */


