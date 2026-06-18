#ifndef __GDD_LOG_CUH__
#define __GDD_LOG_CUH__

//#include "common.cuh"

/* Logarithm.  Computes log(x) in double-double precision.
   This is a natural logarithm (i.e., base e).            */
__device__
gdd_real log(const gdd_real &a);

#endif /* __GDD_LOG_CUH__ */


