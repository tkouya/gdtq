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

#endif /* __GTD_SIN_COS_CUH__ */
