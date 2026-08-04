#ifndef __GQD_SQRT_CUH__
#define __GQD_SQRT_CUH__

//#include "common.cuh"

__device__
gqd_real sqrt(const gqd_real &a);

/* Reference (0.0.2) implementation, kept for benchmarking. */
__device__
gqd_real sqrt_legacy(const gqd_real &a);

#endif /* __GQD_SQRT_CUH__ */


