#ifndef __GQD_LOG_CUH__
#define __GQD_LOG_CUH__

#include "common.cuh"

__device__
gqd_real log(const gqd_real &a);

/* Base-10 logarithm. */
__device__
gqd_real log10(const gqd_real &a);

/* log(1 + a), accurate for small |a| (new in gdtq 0.0.4). */
__device__
gqd_real log1p(const gqd_real &a);

#endif /* __GQD_LOG_CUH__ */
