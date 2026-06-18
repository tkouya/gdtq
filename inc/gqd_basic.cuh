#ifndef __GQD_BASIC_CUH__
#define __GQD_BASIC_CUH__

//#include "gqd_basic.cuh" // New!
//#include "common.cu"
#include "gqd.cuh"


/** normalization functions */
__host__ __device__
void quick_renorm(double &c0, double &c1, 
				  double &c2, double &c3, double &c4);

__host__ __device__
void renorm(double &c0, double &c1, 
			double &c2, double &c3);

__host__ __device__
void renorm(double &c0, double &c1, 
			double &c2, double &c3, double &c4);

__host__ __device__
void renorm( gqd_real &x );

__host__ __device__
void renorm( gqd_real &x, double &e);

/** additions */
__host__ __device__
void three_sum(double &a, double &b, double &c);

__host__ __device__
void three_sum2(double &a, double &b, double &c);

///qd = qd + double
__host__ __device__
gqd_real operator+(const gqd_real &a, double b);

///qd = double + qd
__host__ __device__
gqd_real operator+( double a, const gqd_real &b );

///qd = qd + qd
__host__ __device__
gqd_real sloppy_add(const gqd_real &a, const gqd_real &b);

__host__ __device__
gqd_real bf_add(const gqd_real &a, const gqd_real &b);

__host__ __device__
gqd_real operator+(const gqd_real &a, const gqd_real &b);


/** subtractions */
__host__ __device__
gqd_real negative( const gqd_real &a );

__host__ __device__
gqd_real operator-(const gqd_real &a, double b);

__host__ __device__
gqd_real operator-(double a, const gqd_real &b);

__host__ __device__
gqd_real operator-(const gqd_real &a, const gqd_real &b);

/** multiplications */
__host__ __device__
gqd_real mul_pwr2(const gqd_real &a, double b);

//quad_double * double
 __device__
gqd_real operator*(const gqd_real &a, double b);

//quad_double = double*quad_double
__device__
gqd_real operator*( double a, const gqd_real &b );

__device__
gqd_real sloppy_mul(const gqd_real &a, const gqd_real &b);

__device__
gqd_real bf_mul(const gqd_real &a, const gqd_real &b);

 __device__
gqd_real operator*(const gqd_real &a, const gqd_real &b);

 __device__
gqd_real sqr(const gqd_real &a) ;

/** divisions */
__device__
gqd_real sloppy_div(const gqd_real &a, const gqd_real &b);

__device__
gqd_real operator/(const gqd_real &a, const gqd_real &b);

/* double / quad-double */
__device__
gqd_real operator/(double a, const gqd_real &b);

/* quad-double / double */
__device__
gqd_real operator/( const gqd_real &a, double b );

/********** Miscellaneous **********/
__host__ __device__
gqd_real abs(const gqd_real &a);

/********************** Simple Conversion ********************/
__host__ __device__
double to_double(const gqd_real &a);

__host__ __device__
gqd_real ldexp(const gqd_real &a, int n);

__device__
gqd_real inv(const gqd_real &qd);


/********** Greater-Than Comparison ***********/

__host__ __device__
bool operator>=(const gqd_real &a, const gqd_real &b);

/********** Greater-Than-Or-Equal-To Comparison **********/
/*

__device__
bool operator>=(const gqd_real &a, double b);

__device__
bool operator>=(double a, const gqd_real &b);

__device__
bool operator>=(const gqd_real &a, const gqd_real &b);

*/

/********** Less-Than Comparison ***********/
__host__ __device__
bool operator<(const gqd_real &a, double b);

__host__ __device__
bool operator<(const gqd_real &a, const gqd_real &b);

__host__ __device__
bool operator<=(const gqd_real &a, const gqd_real &b);

__host__ __device__
bool operator==(const gqd_real &a, const gqd_real &b);


/********** Less-Than-Or-Equal-To Comparison **********/
__device__
bool operator<=(const gqd_real &a, double b);

/*

__device__
bool operator<=(double a, const gqd_real &b);

*/

/*

__device__
bool operator<=(const gqd_real &a, const gqd_real &b);

*/

/********** Greater-Than-Or-Equal-To Comparison **********/
__device__
bool operator>=(const gqd_real &a, double b);

__device__
bool operator<=(double a, const gqd_real &b);

__device__
bool operator>=(double a, const gqd_real &b);

/*

__device__
bool operator>=(const gqd_real &a, const gqd_real &b) {
  return (a.x > b.x ||
          (a.x == b.x && (a.y > b.y ||
                            (a.y == b.y && (a.z > b.z ||
                                              (a.z == b.z && a.w >= b.w))))));
}

*/

/********** Greater-Than Comparison ***********/
__host__ __device__
bool operator>(const gqd_real &a, double b);

__host__ __device__
bool operator<(double a, const gqd_real &b);

__host__ __device__
bool operator>(double a, const gqd_real &b);

__host__ __device__ 
bool operator>(const gqd_real &a, const gqd_real &b);

__host__ __device__
bool is_zero( const gqd_real &x );

__host__ __device__
bool is_one( const gqd_real &x );

__host__ __device__
bool is_positive( const gqd_real &x );

__host__ __device__
bool is_negative( const gqd_real &x );

__device__
gqd_real nint(const gqd_real &a);

__device__
gqd_real fabs(const gqd_real &a);


#endif // __GQD_BASIC_CUH__


