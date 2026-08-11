#ifndef __GQS_EXP_CUH__
#define __GQS_EXP_CUH__

//#include "common.cuh"

__device__
gqs_real exp( const gqs_real &a );

/* exp(a) - 1, accurate for small |a|. */
__device__
gqs_real expm1(const gqs_real &a);

#endif /* __GQS_EXP_CUH__ */


