#ifndef __GDD_BASIC_CUH__
#define __GDD_BASIC_CUH__

/**
 * arithmetic operators
 * comparison
 */

//#include "common.cuh"

///////////////////// Addition /////////////////////

__device__
gdd_real negative( const gdd_real &a );

/* double-double = double + double */
__device__
gdd_real dd_add(double a, double b);

/* double-double + double */
__host__ __device__
gdd_real operator+(const gdd_real &a, double b);

/* double + double-double */
__host__ __device__
gdd_real operator+(const double &a, gdd_real b);

__inline__ __host__ __device__
gdd_real sloppy_add(const gdd_real &a, const gdd_real &b) 
{
	double s, e;

	s = two_sum(a.x, b.x, e);
	e += (a.y + b.y);
	s = quick_two_sum(s, e, e);
	return make_dd(s, e);
}

__inline__ __host__ __device__
gdd_real operator+(const gdd_real &a, const gdd_real &b) 
{
	return sloppy_add(a, b);
}

/*********** Subtractions *********/

__device__
gdd_real operator-(const gdd_real &a, const gdd_real &b);

/* double-double - double */
__device__
gdd_real operator-(const gdd_real &a, double b);

/* double - double-double */
__device__
gdd_real operator-(double a, const gdd_real &b);

/*********** Squaring **********/
__device__
gdd_real sqr(const gdd_real &a);

/* `sqr_d` is the canonical name for "square a double, return gdd_real".
 * Always available regardless of which other libraries are in the TU.
 * Used internally by gdd_sqrt.cu so the build doesn't depend on the
 * conditional `sqr(double)` overload below. */
__device__
gdd_real sqr_d(double a);

/* Convenience alias `gdd_real sqr(double)`.  QD's <qd/inline.h>
 * declares `inline double sqr(double)`; if both headers are visible in
 * the same translation unit we get
 *   "cannot overload functions distinguished by return type alone".
 * Skip the alias when _QD_INLINE_H is defined; users in that case
 * should call `sqr_d(x)` (or `sqr(make_dd(x))`). */
#ifndef _QD_INLINE_H
__device__
gdd_real sqr(double a);
#endif

/****************** Multiplication ********************/


/* double-double * (2.0 ^ exp) */
__device__
gdd_real ldexp(const gdd_real &a, int exp);

/* double-double * double,  where double is a power of 2. */
__device__
gdd_real mul_pwr2(const gdd_real &a, double b);

/* double-double * double-double */
__device__
gdd_real operator*(const gdd_real &a, const gdd_real &b);

/* double-double * double */
__device__
gdd_real operator*(const gdd_real &a, double b);

/* double * double-double */
__device__
gdd_real operator*(double a, const gdd_real &b);


/******************* Division *********************/

__device__
gdd_real sloppy_div(const gdd_real &a, const gdd_real &b);

/* double-double / double-double */
__device__
gdd_real operator/(const gdd_real &a, const gdd_real &b);

/* double-double / double */
__device__
gdd_real operator/(const gdd_real &a, double b);

__host__ __device__
bool is_zero( const gdd_real &a );

__host__ __device__
bool is_one( const gdd_real &a );

/*  this > 0 */
__device__ 
bool is_positive(const gdd_real &a);

/* this < 0 */
__device__ 
bool is_negative(const gdd_real &a);

/* Cast to double. */
__device__
double to_double(const gdd_real &a);

/************* Comparison ***************/


/* double-double <= double-double */
__host__ __device__
bool operator<=(const gdd_real &a, const gdd_real &b);



/*********** Equality Comparisons ************/
/* double-double == double */
__host__ __device__ bool operator==(const gdd_real &a, double b);

/* double-double == double-double */
__host__ __device__ bool operator==(const gdd_real &a, const gdd_real &b);

/* double == double-double */
__host__ __device__ bool operator==(double a, const gdd_real &b);

/*********** Greater-Than Comparisons ************/
/* double-double > double */
__host__ __device__ bool operator>(const gdd_real &a, double b);

/* double-double > double-double */
__host__ __device__ bool operator>(const gdd_real &a, const gdd_real &b);

/* double > double-double */
__host__ __device__ bool operator>(double a, const gdd_real &b);

/*********** Less-Than Comparisons ************/
/* double-double < double */
__host__ __device__ bool operator<(const gdd_real &a, double b);

/* double-double < double-double */
__host__ __device__ bool operator<(const gdd_real &a, const gdd_real &b);

/* double < double-double */
__host__ __device__ bool operator<(double a, const gdd_real &b);

/*********** Greater-Than-Or-Equal-To Comparisons ************/
/* double-double >= double */
__host__ __device__ bool operator>=(const gdd_real &a, double b);

/* double-double >= double-double */
__host__ __device__ bool operator>=(const gdd_real &a, const gdd_real &b);

/* double >= double-double */
//__host__ __device__ bool operator>=(double a, const gdd_real &b);

/*********** Less-Than-Or-Equal-To Comparisons ************/
/* double-double <= double */
__host__ __device__ bool operator<=(const gdd_real &a, double b);

/* double >= double-double */
__host__ __device__ bool operator>=(double a, const gdd_real &b);


/* double-double <= double-double */
//__host__ __device__ bool operator<=(const gdd_real &a, const gdd_real &b);

/* double <= double-double */
__host__ __device__ bool operator<=(double a, const gdd_real &b);

/*********** Not-Equal-To Comparisons ************/
/* double-double != double */
__host__ __device__ bool operator!=(const gdd_real &a, double b);

/* double-double != double-double */
__host__ __device__ bool operator!=(const gdd_real &a, const gdd_real &b);

/* double != double-double */
__host__ __device__ bool operator!=(double a, const gdd_real &b);

__device__
gdd_real nint(const gdd_real &a);

__device__
gdd_real abs(const gdd_real &a);

__device__
gdd_real fabs(const gdd_real &a);

/* double / double-double */
__device__
gdd_real operator/(double a, const gdd_real &b);

__device__
gdd_real inv(const gdd_real &a);

#endif /* __GDD_BASIC_CUH__ */

