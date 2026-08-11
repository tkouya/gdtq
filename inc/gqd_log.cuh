#ifndef __GQD_LOG_CUH__
#define __GQD_LOG_CUH__

#include "common.cuh"

__device__
gqd_real log(const gqd_real &a);

/* Base-10 logarithm. */
__device__
gqd_real log10(const gqd_real &a);

#endif /* __GQD_LOG_CUH__ */
