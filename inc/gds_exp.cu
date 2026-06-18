#ifndef __GDS_EXP_CU__
#define __GDS_EXP_CU__

//#include "common.cuh"
//#include "gdd_exp.cuh"
#include "gqs.cuh"

#define INV_K (1.0/512.0) 



//the completed version with additional branches for parameter checking
/*
__device__
gds_real exp(const gds_real &a) {

        const float k = 512.0;
        const float inv_k = 1.0 / k;

        if (a.x <= -709.0)
                return make_ds(0.0);

        if (a.x >=  709.0)
                return make_ds(0.0);
	        //TODO: return dd_real::_inf;

        if (is_zero(a))
                return make_ds(1.0);

        if (is_one(a))
                return _ds_e;

        float m = floor(a.x / _ds_log2.x + 0.5);
        gds_real r = mul_pwr2(a - _ds_log2 * m, inv_k);
        gds_real s, t, p;

        p = sqr(r);
        s = r + mul_pwr2(p, 0.5);
        p = p * r;
        t = p * ds_inv_fact[0];
        int i = 0;
        do {
                s = s + t;
                p = p * r;
                t = p * ds_inv_fact[++i];
        } while ((fabs(to_float(t)) > inv_k * _ds_eps) && (i < 5));

        s = s + t;

        for( int i = 0; i < 9; i++ )
        {
                s = mul_pwr2(s, 2.0) + sqr(s);
        }

        s = s + 1.0;

        return ldexp(s, int(m));
}
*/

__device__
gds_real exp(const gds_real &a) {

        float m = floor(a.x / _ds_log2.x + 0.5);
        gds_real r = mul_pwr2(a - _ds_log2 * m, INV_K);
        gds_real s, t, p;

        p = sqr(r);
        s = r + mul_pwr2(p, 0.5);
        p = p * r;
        t = p * ds_inv_fact[0];
        int i = 0;
        do {
                s = s + t;
                p = p * r;
                t = p * ds_inv_fact[++i];
        } while ((fabs(to_float(t)) > INV_K * _ds_eps) && (i < 5));
        s = s + t;
	
        /* Use a different loop variable name to avoid shadowing the outer 'i'
           (triggers -Wshadow with recent nvcc/host compilers). */
        for( int j = 0; j < 9; j++ ) {
                s = mul_pwr2(s, 2.0) + sqr(s);
        }
        s = s + 1.0;

        return ldexp(s, int(m));
}


#endif /* __GDS_EXP_CU__ */


