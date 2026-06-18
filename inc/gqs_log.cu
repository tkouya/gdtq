#ifndef __GQS_LOG_CU__
#define __GQS_LOG_CU__

//#include "common.cuh"
//#include "gqd_log.cuh"

#include "gqs.cuh"


__device__
gqs_real log(const gqs_real &a) {
        if (is_one(a)) {
                return make_qs(0.0);
        }

        //!!!!!!!!!!!!!
        if (a.x <= 0.0) {
                //qd_real::error("(qd_real::log): Non-positive argument.");
                //return qd_real::_nan;
                return make_qs( 0.0 );
        }

        //!!!!!!!!!!!!!!
        if (a.x == 0.0)      {
                //return _inf;
                //TO DO: return an error
                return make_qs( 0.0 );
        }

        gqs_real x = make_qs(log(a.x));  
        
        x = x + a * exp(negative(x)) - 1.0;
        x = x + a * exp(negative(x)) - 1.0;
        x = x + a * exp(negative(x)) - 1.0;

        return x;
}


#endif /* __GQS_LOG_CU__ */
