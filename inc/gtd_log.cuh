/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef __GTD_LOG_CUH__
#define __GTD_LOG_CUH__

__device__
gtd_real log(const gtd_real &a);

/* Base-10 logarithm. */
__device__
gtd_real log10(const gtd_real &a);

/* log(1 + a), accurate for small |a| (new in gdtq 0.0.4). */
__device__
gtd_real log1p(const gtd_real &a);

#endif /* __GTD_LOG_CUH__ */
