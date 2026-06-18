#ifndef __GDD_SQRT_CUH__
#define __GDD_SQRT_CUH__

//#include "common.cuh"

/* Computes the square root of the double-double number dd.
   NOTE: dd must be a non-negative number.                   */
__device__
gdd_real sqrt(const gdd_real &a);

#endif /* __GDD_SQRT_CUH__ */


