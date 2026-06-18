#ifndef __GDS_LOG_CU__
#define __GDS_LOG_CU__

//#include "common.cuh"
//#include "gdd_log.cuh"
#include "gqs.cuh"

/* Logarithm.  Computes log(x) in float-float precision.
   This is a natural logarithm (i.e., base e).            */
__device__
gds_real log(const gds_real &a) {
  
	if (is_one(a)) {	
		return make_ds(0.0);
	}

//!!!!!!!!!
//TO DO: return an errro
	if (a.x <= 0.0) {
		//return _nan;
		return make_ds( 0.0 );
	}

	gds_real x = make_ds(log(a.x));   // Initial approximation 

	x = x + a * exp(negative(x)) - 1.0;

	return x;
}

#endif /* __GDS_LOG_CU__ */


