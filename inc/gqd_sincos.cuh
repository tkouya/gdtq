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

#ifdef ALL_MATH	

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

#endif /* ALL_MATH */


#endif /* __GQD_SIN_COS_CUH__ */


