#ifndef __GQS_EXP_CUH__
#define __GQS_EXP_CUH__

//#include "common.cuh"

__device__
gqs_real exp( const gqs_real &a );

/* exp(a) - 1, accurate for small |a|. */
__device__
gqs_real expm1(const gqs_real &a);

/* a^b = exp(b log a) for a > 0, with a (K+1)-word log and product
   (new in gdtq 0.0.4). */
__device__
gqs_real pow(const gqs_real &a, const gqs_real &b);

#endif /* __GQS_EXP_CUH__ */


