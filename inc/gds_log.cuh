#ifndef __GDS_LOG_CUH__
#define __GDS_LOG_CUH__

//#include "common.cuh"

/* Logarithm.  Computes log(x) in float-float precision.
   This is a natural logarithm (i.e., base e).            */
__device__
gds_real log(const gds_real &a);

#endif /* __GDS_LOG_CUH__ */


