#ifndef __GQS_BASIC_CUH__
#define __GQS_BASIC_CUH__

//#include "gqd_basic.cuh" // New!
//#include "common.cu"
#include "gqs.cuh"


/** normalization functions */
__host__ __device__
void quick_renorm(float &c0, float &c1, 
				  float &c2, float &c3, float &c4);

__host__ __device__
void renorm(float &c0, float &c1, 
			float &c2, float &c3);

__host__ __device__
void renorm(float &c0, float &c1, 
			float &c2, float &c3, float &c4);

__host__ __device__
void renorm( gqs_real &x );

__host__ __device__
void renorm( gqs_real &x, float &e);

/** additions */
__host__ __device__
void three_sum(float &a, float &b, float &c);

__host__ __device__
void three_sum2(float &a, float &b, float &c);

///qd = qd + float
__host__ __device__
gqs_real operator+(const gqs_real &a, float b);

///qd = float + qd
__host__ __device__
gqs_real operator+( float a, const gqs_real &b );

///qd = qd + qd
__host__ __device__
gqs_real sloppy_add(const gqs_real &a, const gqs_real &b);

__host__ __device__
gqs_real bf_add(const gqs_real &a, const gqs_real &b);

__host__ __device__
gqs_real operator+(const gqs_real &a, const gqs_real &b);


/** subtractions */
__host__ __device__
gqs_real negative( const gqs_real &a );

__host__ __device__
gqs_real operator-(const gqs_real &a, float b);

__host__ __device__
gqs_real operator-(float a, const gqs_real &b);

__host__ __device__
gqs_real operator-(const gqs_real &a, const gqs_real &b);

/** multiplications */
__host__ __device__
gqs_real mul_pwr2(const gqs_real &a, float b);

//quad_double * float
 __device__
gqs_real operator*(const gqs_real &a, float b);

//quad_double = float*quad_double
__device__
gqs_real operator*( float a, const gqs_real &b );

__device__
gqs_real sloppy_mul(const gqs_real &a, const gqs_real &b);

__device__
gqs_real bf_mul(const gqs_real &a, const gqs_real &b);

 __device__
gqs_real operator*(const gqs_real &a, const gqs_real &b);

 __device__
gqs_real sqr(const gqs_real &a) ;

/** divisions */
__device__
gqs_real sloppy_div(const gqs_real &a, const gqs_real &b);

__device__
gqs_real operator/(const gqs_real &a, const gqs_real &b);

/* float / quad-float */
__device__
gqs_real operator/(float a, const gqs_real &b);

/* quad-float / float */
__device__
gqs_real operator/( const gqs_real &a, float b );

/********** Miscellaneous **********/
__host__ __device__
gqs_real abs(const gqs_real &a);

/********************** Simple Conversion ********************/
__host__ __device__
float to_float(const gqs_real &a);

__host__ __device__
gqs_real ldexp(const gqs_real &a, int n);

__device__
gqs_real inv(const gqs_real &qd);


/********** Greater-Than Comparison ***********/

__host__ __device__
bool operator>=(const gqs_real &a, const gqs_real &b);

/********** Greater-Than-Or-Equal-To Comparison **********/
/*

__device__
bool operator>=(const gqs_real &a, float b);

__device__
bool operator>=(float a, const gqs_real &b);

__device__
bool operator>=(const gqs_real &a, const gqs_real &b);

*/

/********** Less-Than Comparison ***********/
__host__ __device__
bool operator<(const gqs_real &a, float b);

__host__ __device__
bool operator<(const gqs_real &a, const gqs_real &b);

__host__ __device__
bool operator<=(const gqs_real &a, const gqs_real &b);

__host__ __device__
bool operator==(const gqs_real &a, const gqs_real &b);


/********** Less-Than-Or-Equal-To Comparison **********/
__device__
bool operator<=(const gqs_real &a, float b);

/*

__device__
bool operator<=(float a, const gqs_real &b);

*/

/*

__device__
bool operator<=(const gqs_real &a, const gqs_real &b);

*/

/********** Greater-Than-Or-Equal-To Comparison **********/
__device__
bool operator>=(const gqs_real &a, float b);

__device__
bool operator<=(float a, const gqs_real &b);

__device__
bool operator>=(float a, const gqs_real &b);

/*

__device__
bool operator>=(const gqs_real &a, const gqs_real &b) {
  return (a.x > b.x ||
          (a.x == b.x && (a.y > b.y ||
                            (a.y == b.y && (a.z > b.z ||
                                              (a.z == b.z && a.w >= b.w))))));
}

*/

/********** Greater-Than Comparison ***********/
__host__ __device__
bool operator>(const gqs_real &a, float b);

__host__ __device__
bool operator<(float a, const gqs_real &b);

__host__ __device__
bool operator>(float a, const gqs_real &b);

__host__ __device__ 
bool operator>(const gqs_real &a, const gqs_real &b);

__host__ __device__
bool is_zero( const gqs_real &x );

__host__ __device__
bool is_one( const gqs_real &x );

__host__ __device__
bool is_positive( const gqs_real &x );

__host__ __device__
bool is_negative( const gqs_real &x );

__device__
gqs_real nint(const gqs_real &a);

__device__
gqs_real fabs(const gqs_real &a);


#endif // __GQS_BASIC_CUH__


