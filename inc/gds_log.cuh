#ifndef __GDS_LOG_CUH__
#define __GDS_LOG_CUH__

//#include "common.cuh"

/* Logarithm.  Computes log(x) in float-float precision.
   This is a natural logarithm (i.e., base e).            */
__device__
gds_real log(const gds_real &a);

/* Base-10 logarithm. */
__device__
gds_real log10(const gds_real &a);

/* log(1 + a), accurate for small |a| (new in gdtq 0.0.4). */
__device__
gds_real log1p(const gds_real &a);

#endif /* __GDS_LOG_CUH__ */


