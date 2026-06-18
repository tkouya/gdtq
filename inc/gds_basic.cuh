#ifndef __GDS_BASIC_CUH__
#define __GDS_BASIC_CUH__

/**
 * arithmetic operators
 * comparison
 */

//#include "common.cuh"

///////////////////// Addition /////////////////////

__device__
gds_real negative( const gds_real &a );

/* float-float = float + float */
__device__
gds_real ds_add(float a, float b);

/* float-float + float */
__host__ __device__
gds_real operator+(const gds_real &a, float b);

/* float + float-float */
__host__ __device__
gds_real operator+(const float &a, gds_real b);

__inline__ __host__ __device__
gds_real sloppy_add(const gds_real &a, const gds_real &b) 
{
	float s, e;

	s = two_sum(a.x, b.x, e);
	e += (a.y + b.y);
	s = quick_two_sum(s, e, e);
	return make_ds(s, e);
}

__inline__ __host__ __device__
gds_real operator+(const gds_real &a, const gds_real &b) 
{
	return sloppy_add(a, b);
}

/*********** Subtractions *********/

__device__
gds_real operator-(const gds_real &a, const gds_real &b);

/* float-float - float */
__device__
gds_real operator-(const gds_real &a, float b);

/* float - float-float */
__device__
gds_real operator-(float a, const gds_real &b);

/*********** Squaring **********/
__device__
gds_real sqr(const gds_real &a);

__device__
gds_real sqr(float a);

/****************** Multiplication ********************/


/* float-float * (2.0 ^ exp) */
__device__
gds_real ldexp(const gds_real &a, int exp);

/* float-float * float,  where float is a power of 2. */
__device__
gds_real mul_pwr2(const gds_real &a, float b);

/* float-float * float-float */
__device__
gds_real operator*(const gds_real &a, const gds_real &b);

/* float-float * float */
__device__
gds_real operator*(const gds_real &a, float b);

/* float * float-float */
__device__
gds_real operator*(float a, const gds_real &b);


/******************* Division *********************/

__device__
gds_real sloppy_div(const gds_real &a, const gds_real &b);

/* float-float / float-float */
__device__
gds_real operator/(const gds_real &a, const gds_real &b);

/* float-float / float */
__device__
gds_real operator/(const gds_real &a, float b);

__host__ __device__
bool is_zero( const gds_real &a );

__host__ __device__
bool is_one( const gds_real &a );

/*  this > 0 */
__device__ 
bool is_positive(const gds_real &a);

/* this < 0 */
__device__ 
bool is_negative(const gds_real &a);

/* Cast to float. */
__device__
float to_float(const gds_real &a);

/************* Comparison ***************/


/* float-float <= float-float */
__host__ __device__
bool operator<=(const gds_real &a, const gds_real &b);



/*********** Equality Comparisons ************/
/* float-float == float */
__host__ __device__ bool operator==(const gds_real &a, float b);

/* float-float == float-float */
__host__ __device__ bool operator==(const gds_real &a, const gds_real &b);

/* float == float-float */
__host__ __device__ bool operator==(float a, const gds_real &b);

/*********** Greater-Than Comparisons ************/
/* float-float > float */
__host__ __device__ bool operator>(const gds_real &a, float b);

/* float-float > float-float */
__host__ __device__ bool operator>(const gds_real &a, const gds_real &b);

/* float > float-float */
__host__ __device__ bool operator>(float a, const gds_real &b);

/*********** Less-Than Comparisons ************/
/* float-float < float */
__host__ __device__ bool operator<(const gds_real &a, float b);

/* float-float < float-float */
__host__ __device__ bool operator<(const gds_real &a, const gds_real &b);

/* float < float-float */
__host__ __device__ bool operator<(float a, const gds_real &b);

/*********** Greater-Than-Or-Equal-To Comparisons ************/
/* float-float >= float */
__host__ __device__ bool operator>=(const gds_real &a, float b);

/* float-float >= float-float */
__host__ __device__ bool operator>=(const gds_real &a, const gds_real &b);

/* float >= float-float */
//__host__ __device__ bool operator>=(float a, const gds_real &b);

/*********** Less-Than-Or-Equal-To Comparisons ************/
/* float-float <= float */
__host__ __device__ bool operator<=(const gds_real &a, float b);

/* float >= float-float */
__host__ __device__ bool operator>=(float a, const gds_real &b);


/* float-float <= float-float */
//__host__ __device__ bool operator<=(const gds_real &a, const gds_real &b);

/* float <= float-float */
__host__ __device__ bool operator<=(float a, const gds_real &b);

/*********** Not-Equal-To Comparisons ************/
/* float-float != float */
__host__ __device__ bool operator!=(const gds_real &a, float b);

/* float-float != float-float */
__host__ __device__ bool operator!=(const gds_real &a, const gds_real &b);

/* float != float-float */
__host__ __device__ bool operator!=(float a, const gds_real &b);

__device__
gds_real nint(const gds_real &a);

__device__
gds_real abs(const gds_real &a);

__device__
gds_real fabs(const gds_real &a);

/* float / float-float */
__device__
gds_real operator/(float a, const gds_real &b);

__device__
gds_real inv(const gds_real &a);

#endif /* __GDS_BASIC_CUH__ */

