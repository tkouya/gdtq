/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef __GTD_SIN_COS_CUH__
#define __GTD_SIN_COS_CUH__

__device__
void sincos_taylor(const gtd_real &a, gtd_real &sin_a, gtd_real &cos_a);

__device__
gtd_real sin_taylor(const gtd_real &a);

__device__
gtd_real cos_taylor(const gtd_real &a);

__device__
gtd_real sin(const gtd_real &a);

__device__
gtd_real cos(const gtd_real &a);

__device__
void sincos(const gtd_real &a, gtd_real &sin_a, gtd_real &cos_a);

__device__
gtd_real tan(const gtd_real &a);

/* inverse trigonometric and hyperbolic functions (new for this type in
   gdtq 0.0.4; mw kernels, see gqd_elem.cu) */
__device__
gtd_real atan2(const gtd_real &y, const gtd_real &x);

__device__
gtd_real atan(const gtd_real &a);

__device__
gtd_real asin(const gtd_real &a);

__device__
gtd_real acos(const gtd_real &a);

__device__
gtd_real sinh(const gtd_real &a);

__device__
gtd_real cosh(const gtd_real &a);

__device__
gtd_real tanh(const gtd_real &a);

__device__
void sincosh(const gtd_real &a, gtd_real &sinh_a, gtd_real &cosh_a);

__device__
gtd_real asinh(const gtd_real &a);

__device__
gtd_real acosh(const gtd_real &a);

__device__
gtd_real atanh(const gtd_real &a);

#endif /* __GTD_SIN_COS_CUH__ */
