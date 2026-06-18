#ifndef __GQS_SIN_COS_CUH__
#define __GQS_SIN_COS_CUH__

//#include "common.cuh"
//#include "gqd_sincos.cuh"

__device__
void sincos_taylor(const gqs_real &a, 
				   gqs_real &sin_a, gqs_real &cos_a);

__device__
gqs_real sin_taylor(const gqs_real &a);

__device__
gqs_real cos_taylor(const gqs_real &a);

__device__
gqs_real sin(const gqs_real &a);

__device__
gqs_real cos(const gqs_real &a);

__device__
void sincos(const gqs_real &a, gqs_real &sin_a, gqs_real &cos_a);

__device__
gqs_real tan(const gqs_real &a);

#ifdef ALL_MATH	

__device__
gqs_real atan2(const gqs_real &y, const gqs_real &x);

__device__
gqs_real atan(const gqs_real &a);

__device__
gqs_real asin(const gqs_real &a);

__device__
gqs_real acos(const gqs_real &a);

__device__
gqs_real sinh(const gqs_real &a);

__device__
gqs_real cosh(const gqs_real &a);

__device__
gqs_real tanh(const gqs_real &a);

__device__
void sincosh(const gqs_real &a, gqs_real &s, gqs_real &c);

__device__
gqs_real asinh(const gqs_real &a);

__device__
gqs_real acosh(const gqs_real &a);

__device__
gqs_real atanh(const gqs_real &a);

#endif /* ALL_MATH */


#endif /* __GQS_SIN_COS_CUH__ */


