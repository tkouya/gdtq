#ifndef __GQS_LOG_CUH__
#define __GQS_LOG_CUH__

#include "common.cuh"

__device__
gqs_real log(const gqs_real &a);

/* Base-10 logarithm. */
__device__
gqs_real log10(const gqs_real &a);

/* log(1 + a), accurate for small |a| (new in gdtq 0.0.4). */
__device__
gqs_real log1p(const gqs_real &a);

#endif /* __GQS_LOG_CUH__ */
