#ifndef __GDS_EXP_CUH__
#define __GDS_EXP_CUH__

//#include "common.cuh"

//#define INV_K (1.0/512.0) 

__device__
gds_real exp(const gds_real &a);

/* exp(a) - 1, accurate for small |a|. */
__device__
gds_real expm1(const gds_real &a);

/* a^b = exp(b log a) for a > 0, with a (K+1)-word log and product
   (new in gdtq 0.0.4). */
__device__
gds_real pow(const gds_real &a, const gds_real &b);

#endif /* __GDS_EXP_CUH__ */


