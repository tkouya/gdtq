/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef __GTS_EXP_CUH__
#define __GTS_EXP_CUH__

__device__
gts_real exp(const gts_real &a);

/* exp(a) - 1, accurate for small |a|. */
__device__
gts_real expm1(const gts_real &a);

#endif /* __GTS_EXP_CUH__ */
