#ifndef __GQD_CU__
#define __GQD_CU__

/**
* the API file
* includes every thing for this library
*/
#include "inline.cu"
#include "common.cu"

/* gdd_library */
#include "gdd_basic.cu"
#include "gdd_sqrt.cu"
#include "gdd_exp.cu"
#include "gdd_log.cu"
#include "gdd_sincos.cu"


/* gtd_library (triple-double; uses helpers shared with gqd) */
#include "gtd_basic.cu"
#include "gtd_sqrt.cu"
#include "gtd_exp.cu"
#include "gtd_log.cu"
#include "gtd_sincos.cu"


/* gqd_libraray */
#include "gqd_basic.cu"
#include "gqd_sqrt.cu"
#include "gqd_exp.cu"
#include "gqd_log.cu"
#include "gqd_sincos.cu"

#endif // __GQD_CU__


