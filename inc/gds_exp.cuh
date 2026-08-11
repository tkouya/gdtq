#ifndef __GDS_EXP_CUH__
#define __GDS_EXP_CUH__

//#include "common.cuh"

//#define INV_K (1.0/512.0) 

__device__
gds_real exp(const gds_real &a);

/* exp(a) - 1, accurate for small |a|. */
__device__
gds_real expm1(const gds_real &a);

#endif /* __GDS_EXP_CUH__ */


