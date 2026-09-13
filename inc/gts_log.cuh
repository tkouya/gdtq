/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef __GTS_LOG_CUH__
#define __GTS_LOG_CUH__

__device__
gts_real log(const gts_real &a);

/* Base-10 logarithm. */
__device__
gts_real log10(const gts_real &a);

/* log(1 + a), accurate for small |a| (new in gdtq 0.0.4). */
__device__
gts_real log1p(const gts_real &a);

#endif /* __GTS_LOG_CUH__ */
