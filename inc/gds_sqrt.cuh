#ifndef __GDS_SQRT_CUH__
#define __GDS_SQRT_CUH__

//#include "common.cuh"

/* Computes the square root of the float-float number dd.
   NOTE: dd must be a non-negative number.                   */
__device__
gds_real sqrt(const gds_real &a);

#endif /* __GDS_SQRT_CUH__ */


