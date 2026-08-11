#ifndef __GQS_LOG_CUH__
#define __GQS_LOG_CUH__

#include "common.cuh"

__device__
gqs_real log(const gqs_real &a);

/* Base-10 logarithm. */
__device__
gqs_real log10(const gqs_real &a);

#endif /* __GQS_LOG_CUH__ */
