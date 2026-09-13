#ifndef __GDD_EXP_CUH__
#define __GDD_EXP_CUH__

//#include "common.cuh"

//#define INV_K (1.0/512.0) 

__device__
gdd_real exp(const gdd_real &a);

/* exp(a) - 1, accurate for small |a|. */
__device__
gdd_real expm1(const gdd_real &a);

/* a^b = exp(b log a) for a > 0, with a (K+1)-word log and product
   (new in gdtq 0.0.4). */
__device__
gdd_real pow(const gdd_real &a, const gdd_real &b);

#endif /* __GDD_EXP_CUH__ */


