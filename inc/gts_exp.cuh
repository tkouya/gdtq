/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef __GTS_EXP_CUH__
#define __GTS_EXP_CUH__

__device__
gts_real exp(const gts_real &a);

/* exp(a) - 1, accurate for small |a|. */
__device__
gts_real expm1(const gts_real &a);

/* a^b = exp(b log a) for a > 0, with a (K+1)-word log and product
   (new in gdtq 0.0.4). */
__device__
gts_real pow(const gts_real &a, const gts_real &b);

#endif /* __GTS_EXP_CUH__ */
