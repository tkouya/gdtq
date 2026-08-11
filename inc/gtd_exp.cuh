/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef __GTD_EXP_CUH__
#define __GTD_EXP_CUH__

__device__
gtd_real exp(const gtd_real &a);

/* exp(a) - 1, accurate for small |a|. */
__device__
gtd_real expm1(const gtd_real &a);

#endif /* __GTD_EXP_CUH__ */
