/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef __GTD_EXP_CUH__
#define __GTD_EXP_CUH__

__device__
gtd_real exp(const gtd_real &a);

/* exp(a) - 1, accurate for small |a|. */
__device__
gtd_real expm1(const gtd_real &a);

/* a^b = exp(b log a) for a > 0, with a (K+1)-word log and product
   (new in gdtq 0.0.4). */
__device__
gtd_real pow(const gtd_real &a, const gtd_real &b);

#endif /* __GTD_EXP_CUH__ */
