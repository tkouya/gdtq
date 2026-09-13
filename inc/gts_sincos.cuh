/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef __GTS_SIN_COS_CUH__
#define __GTS_SIN_COS_CUH__

__device__
void sincos_taylor(const gts_real &a, gts_real &sin_a, gts_real &cos_a);

__device__
gts_real sin_taylor(const gts_real &a);

__device__
gts_real cos_taylor(const gts_real &a);

__device__
gts_real sin(const gts_real &a);

__device__
gts_real cos(const gts_real &a);

__device__
void sincos(const gts_real &a, gts_real &sin_a, gts_real &cos_a);

__device__
gts_real tan(const gts_real &a);

/* inverse trigonometric and hyperbolic functions (new for this type in
   gdtq 0.0.4; mw kernels, see gqs_elem.cu) */
__device__
gts_real atan2(const gts_real &y, const gts_real &x);

__device__
gts_real atan(const gts_real &a);

__device__
gts_real asin(const gts_real &a);

__device__
gts_real acos(const gts_real &a);

__device__
gts_real sinh(const gts_real &a);

__device__
gts_real cosh(const gts_real &a);

__device__
gts_real tanh(const gts_real &a);

__device__
void sincosh(const gts_real &a, gts_real &sinh_a, gts_real &cosh_a);

__device__
gts_real asinh(const gts_real &a);

__device__
gts_real acosh(const gts_real &a);

__device__
gts_real atanh(const gts_real &a);

#endif /* __GTS_SIN_COS_CUH__ */
