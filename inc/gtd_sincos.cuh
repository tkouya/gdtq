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

#endif /* __GTD_SIN_COS_CUH__ */
