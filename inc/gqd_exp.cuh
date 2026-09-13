#ifndef __GQD_EXP_CUH__
#define __GQD_EXP_CUH__

//#include "common.cuh"

__device__
gqd_real exp( const gqd_real &a );

/* exp(a) - 1, accurate for small |a|. */
__device__
gqd_real expm1(const gqd_real &a);

/* a^b = exp(b log a) for a > 0, with a (K+1)-word log and product
   (new in gdtq 0.0.4). */
__device__
gqd_real pow(const gqd_real &a, const gqd_real &b);

#endif /* __GQD_EXP_CUH__ */


