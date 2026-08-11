#ifndef __GQD_EXP_CUH__
#define __GQD_EXP_CUH__

//#include "common.cuh"

__device__
gqd_real exp( const gqd_real &a );

/* exp(a) - 1, accurate for small |a|. */
__device__
gqd_real expm1(const gqd_real &a);

#endif /* __GQD_EXP_CUH__ */


