/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GQD_BASIC_CU__
#define __GQD_BASIC_CU__


//#include "common.cuh"
//#include "gqd_basic.cuh"
#include "gqd.cuh"


/** normalization functions */
__host__ __device__
void quick_renorm(double &c0, double &c1, 
				  double &c2, double &c3, double &c4) 
{
	double t0, t1, t2, t3;
	double s;
	s  = quick_two_sum(c3, c4, t3);
	s  = quick_two_sum(c2, s , t2);
	s  = quick_two_sum(c1, s , t1);
	c0 = quick_two_sum(c0, s , t0);

	s  = quick_two_sum(t2, t3, t2);
	s  = quick_two_sum(t1, s , t1);
	c1 = quick_two_sum(t0, s , t0);

	s  = quick_two_sum(t1, t2, t1);
	c2 = quick_two_sum(t0, s , t0);

	c3 = t0 + t1;
}

__host__ __device__
void renorm(double &c0, double &c1, 
			double &c2, double &c3) 
{
	double s0, s1, s2 = 0.0, s3 = 0.0;

	//if (isinf(c0)) return;

	s0 = quick_two_sum(c2, c3, c3);
	s0 = quick_two_sum(c1, s0, c2);
	c0 = quick_two_sum(c0, s0, c1);

	s0 = c0;
	s1 = c1;
	if (s1 != 0.0) {
		s1 = quick_two_sum(s1, c2, s2);
		if (s2 != 0.0)
			s2 = quick_two_sum(s2, c3, s3);
		else
			s1 = quick_two_sum(s1, c3, s2);
	} else {
		s0 = quick_two_sum(s0, c2, s1);
		if (s1 != 0.0)
			s1 = quick_two_sum(s1, c3, s2);
		else
			s0 = quick_two_sum(s0, c3, s1);
	}

	c0 = s0;
	c1 = s1;
	c2 = s2;
	c3 = s3;
}

__host__ __device__
void renorm(double &c0, double &c1, 
			double &c2, double &c3, double &c4) 
{
	double s0, s1, s2 = 0.0, s3 = 0.0;

	//if (isinf(c0)) return;

	s0 = quick_two_sum(c3, c4, c4);
	s0 = quick_two_sum(c2, s0, c3);
	s0 = quick_two_sum(c1, s0, c2);
	c0 = quick_two_sum(c0, s0, c1);

	/* The original code had `s0 = c0; s1 = c1;` here, but both stores
	   are dead: s0 and s1 are overwritten by the following call.  They
	   are removed to silence dead-store / unused-but-set warnings. */
	s0 = quick_two_sum(c0, c1, s1);
	if (s1 != 0.0) 
	{
		s1 = quick_two_sum(s1, c2, s2);
		if (s2 != 0.0) {
			s2 = quick_two_sum(s2, c3, s3);
			if (s3 != 0.0)
				s3 += c4;
			else
				s2 += c4;
		} else {
			s1 = quick_two_sum(s1, c3, s2);
			if (s2 != 0.0)
				s2 = quick_two_sum(s2, c4, s3);
			else
				s1 = quick_two_sum(s1, c4, s2);
		}
	} else {
		s0 = quick_two_sum(s0, c2, s1);
		if (s1 != 0.0) {
			s1 = quick_two_sum(s1, c3, s2);
			if (s2 != 0.0)
				s2 = quick_two_sum(s2, c4, s3);
			else
				s1 = quick_two_sum(s1, c4, s2);
		} else {
			s0 = quick_two_sum(s0, c3, s1);
			if (s1 != 0.0)
				s1 = quick_two_sum(s1, c4, s2);
			else
				s0 = quick_two_sum(s0, c4, s1);
		}
	}

	c0 = s0;
	c1 = s1;
	c2 = s2;
	c3 = s3;
}

__host__ __device__
void renorm( gqd_real &x ) {
	renorm(x.x, x.y, x.z, x.w);
}

__host__ __device__
void renorm( gqd_real &x, double &e) {
	renorm(x.x, x.y, x.z, x.w, e);
}

/** additions */
/* three_sum / three_sum2 are now defined in inline.cu so that
 * gtd_basic.cu can share them. */

///qd = qd + double
__host__ __device__
gqd_real operator+(const gqd_real &a, double b) {
        double c0, c1, c2, c3;
        double e;

        c0 = two_sum(a.x, b, e);
        c1 = two_sum(a.y, e, e);
        c2 = two_sum(a.z, e, e);
        c3 = two_sum(a.w, e, e);

        renorm(c0, c1, c2, c3, e);

        return make_qd(c0, c1, c2, c3);
}

///qd = double + qd
__host__ __device__
gqd_real operator+( double a, const gqd_real &b )
{
        return ( b + a );
}

///qd = qd + qd
__host__ __device__
gqd_real sloppy_add(const gqd_real &a, const gqd_real &b)
{
        double s0, s1, s2, s3;
        double t0, t1, t2, t3;

        double v0, v1, v2, v3;
        double u0, u1, u2, u3;
        double w0, w1, w2, w3;

        s0 = a.x + b.x;
        s1 = a.y + b.y;
        s2 = a.z + b.z;
        s3 = a.w + b.w;

        v0 = s0 - a.x;
        v1 = s1 - a.y;
        v2 = s2 - a.z;
        v3 = s3 - a.w;

        u0 = s0 - v0;
        u1 = s1 - v1;
        u2 = s2 - v2;
        u3 = s3 - v3;

        w0 = a.x - u0;
        w1 = a.y - u1;
        w2 = a.z - u2;
        w3 = a.w - u3;

        u0 = b.x - v0;
        u1 = b.y - v1;
        u2 = b.z - v2;
        u3 = b.w - v3;

        t0 = w0 + u0;
        t1 = w1 + u1;
        t2 = w2 + u2;
        t3 = w3 + u3;

        s1 = two_sum(s1, t0, t0);
        three_sum(s2, t0, t1);
        three_sum2(s3, t0, t2);
        t0 = t0 + t1 + t3;
  
	renorm(s0, s1, s2, s3, t0);

        return make_qd(s0, s1, s2, s3);
}


// 2025-12-25(Wed) T.Kouya
// Branch free algorithm
// void Add4(const double x[4], const double y[4], double z[4]) {
//static inline void c_qd_add_bf(const double *a, const double *b, double *c)
__host__ __device__ 
gqd_real bf_add(const gqd_real &a, const gqd_real &b)
{
	double a0 , b0 , c0 , d0 , e0, f0, g0, h0;
	double a1 , b1 , c1 , d1 , e1, f1, g1, h1;
	double a2 , b2 , c2 , d2 , e2, f2, g2;
	double      b3 , c3 , d3 , e3, f3, g3;
	double a4 ,      c4 , d4 , e4;
	double      b5 ,      d5 , e5;
	double      b6 , c6 , d6 , e6;
	double a7 , b7 , c7 , d7;
	double      b8 , c8 ,      e8;
	double                d9 ;
	double a10, b10, c10, d10;
	double      b11, c11;
	double           c12, d12;

	a0 = a.x;
	b0 = b.x;
	c0 = a.y;
	d0 = b.y;
	e0 = a.z;
	f0 = b.z;
	g0 = a.w;
	h0 = b.w;
	a1 = two_sum(a0, b0, b1);
	c1 = two_sum(c0, d0, d1);
	e1 = two_sum(e0, f0, f1);
	g1 = two_sum(g0, h0, h1);
	a2 = quick_two_sum(a1, c1, c2);
	b2 = b1 + h1;
	d2 = two_sum(d1, e1, e2);
	f2 = two_sum(f1, g1, g2);
	b3 = two_sum(b2, g2, g3);
	c3 = quick_two_sum(c2, d2, d3);
	e3 = two_sum(e2, f2, f3);
	a4 = quick_two_sum(a2, c3, c4);
	d4 = quick_two_sum(d3, e3, e4);
	b5 = two_sum(b3, d4, d5);
	e5 = e4 + f3;
	b6 = two_sum(b5, c4, c6);
	d6 = two_sum(d5, e5, e6);
	a7 = quick_two_sum(a4, b6, b7);
	c7 = quick_two_sum(c6, d6, d7);
	e8 = e6 + g3;
	b8 = quick_two_sum(b7, c7, c8);
	d9 = d7 + e8;
	a10 = quick_two_sum(a7, b8, b10);
	c10 = quick_two_sum(c8, d9, d10);
	b11 = quick_two_sum(b10, c10, c11);
	c12 = quick_two_sum(c11, d10, d12);

	//c[0] = a10;
	//c[1] = b11;
	//c[2] = c12;
	//c[3] = d12;
	return make_qd(a10, b11, c12, d12);
}

__host__ __device__
gqd_real operator+(const gqd_real &a, const gqd_real &b)
{
	#ifdef USE_SLOPPY_ADD
		return sloppy_add(a, b);
	#else
		return bf_add(a, b);
	#endif // USE_SLOPPY_ADD
}


/** subtractions */
__host__ __device__
gqd_real negative( const gqd_real &a )
{
        return make_qd( -a.x, -a.y, -a.z, -a.w );
}

__host__ __device__
gqd_real operator-(const gqd_real &a, double b)
{
        return (a + (-b));
}

__host__ __device__
gqd_real operator-(double a, const gqd_real &b)
{
        return (a + negative(b));
}

__host__ __device__
gqd_real operator-(const gqd_real &a, const gqd_real &b)
{
        return (a + negative(b));
}

/** multiplications */
__host__ __device__
gqd_real mul_pwr2(const gqd_real &a, double b) {
        return make_qd(a.x * b, a.y * b, a.z * b, a.w * b);
}


//quad_double * double
 __device__
gqd_real operator*(const gqd_real &a, double b)
{
        double p0, p1, p2, p3;
        double q0, q1, q2;
        double s0, s1, s2, s3, s4;

        p0 = two_prod(a.x, b, q0);
        p1 = two_prod(a.y, b, q1);
        p2 = two_prod(a.z, b, q2);
        p3 = a.w * b;

        s0 = p0;

        s1 = two_sum(q0, p1, s2);

        three_sum(s2, q1, p2);

        three_sum2(q1, q2, p3);
        s3 = q1;

        s4 = q2 + p2;

        renorm(s0, s1, s2, s3, s4);
        return make_qd(s0, s1, s2, s3);
}
//quad_double = double*quad_double
__device__
gqd_real operator*( double a, const gqd_real &b )
{
        return b*a;
}

__device__
gqd_real sloppy_mul(const gqd_real &a, const gqd_real &b)
{
        double p0, p1, p2, p3, p4, p5;
        double q0, q1, q2, q3, q4, q5;
        double t0, t1;
        double s0, s1, s2;

        p0 = two_prod(a.x, b.x, q0);

        p1 = two_prod(a.x, b.y, q1);
        p2 = two_prod(a.y, b.x, q2);

        p3 = two_prod(a.x, b.z, q3);
        p4 = two_prod(a.y, b.y, q4);
        p5 = two_prod(a.z, b.x, q5);


        /* Start Accumulation */
        three_sum(p1, p2, q0);

	//return make_qd(p1, p2, q0, 0.0);

        /* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
        three_sum(p2, q1, q2);
        three_sum(p3, p4, p5);
        /* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
        s0 = two_sum(p2, p3, t0);
        s1 = two_sum(q1, p4, t1);
	s2 = q2 + p5;
        s1 = two_sum(s1, t0, t0);
        s2 += (t0 + t1);

	//return make_qd(s0, s1, t0, t1);

        /* O(eps^3) order terms */
        //!!!s1 = s1 + (a.x*b.w + a.y*b.z + a.z*b.y + a.w*b.x + q0 + q3 + q4 + q5);
	
	s1 = s1 + (__dmul_rn(a.x,b.w) + __dmul_rn(a.y,b.z) + 
			__dmul_rn(a.z,b.y) + __dmul_rn(a.w,b.x) + q0 + q3 + q4 + q5);
	renorm(p0, p1, s0, s1, s2);

        return make_qd(p0, p1, s0, s1);
	
}
// 2025-12-24(Wed) T.Kouya
// Branch free algorithm
// void Mul4(const double x[4], const double y[4], double z[4]) {
//static inline void c_qd_mul_bf(const double *a, const double *b, double *c)
__device__
gqd_real bf_mul(const gqd_real &a, const gqd_real &b)
{
	double a0, b0, c0, d0, e0, f0, g0, h0, i0, j0, k0, l0, m0, n0, o0, p0;
	double         c1, d1, e1, f1, g1,     i1, j1,         m1, n1;
	double     b2, c2,     e2, f2,     h2, i2,             m2;
	double a3, b3, c3, d3, e3, f3, g3, h3;
	double         c4, d4, e4, f4;
	double                 d5;
	double         c6,     d6;
	double     b7, c7, d7;
	double a8, b8, c8, d8;
	double     b9, c9;
	double         c10, d10;

	a0 = two_prod(a.x, b.x, b0);
	c0 = two_prod(a.x, b.y, e0);
	d0 = two_prod(a.y, b.x, f0);
	g0 = two_prod(a.x, b.z, j0);
	h0 = two_prod(a.y, b.y, k0);
	i0 = two_prod(a.z, b.x, l0);
	m0 = a.x * b.w;
	n0 = a.y * b.z;
	o0 = a.z * b.y;
	p0 = a.w * b.x;
	c1 = two_sum(c0, d0, d1);
	e1 = two_sum(e0, f0, f1);
	g1 = two_sum(g0, i0, i1);
	j1 = j0 + l0;
	m1 = m0 + p0;
	n1 = n0 + o0;
	b2 = two_sum(b0, c1, c2);
	e2 = two_sum(e1, h0, h2);
	f2 = f1 + j1;
	i2 = i1 + k0;
	m2 = m1 + n1;
	a3 = quick_two_sum(a0, b2, b3);
	c3 = quick_two_sum(c2, d1, d3);
	e3 = two_sum(e2, g1, g3);
	f3 = f2 + m2;
	h3 = h2 + i2;
	c4 = two_sum(c3, e3, e4);
	d4 = d3 + h3;
	f4 = f3 + g3;
	d5 = d4 + e4;
	c6 = two_sum(c4, d5, d6);
	b7 = two_sum(b3, c6, c7);
	d7 = d6 + f4;
	a8 = quick_two_sum(a3, b7, b8);
	c8 = two_sum(c7, d7, d8);
	b9 = two_sum(b8, c8, c9);
	c10 = quick_two_sum(c9, d8, d10);

	//c[0] = a8;
	//c[1] = b9;
	//c[2] = c10;
	//c[3] = d10;
	return make_qd(a8, b9, c10, d10);
}

 __device__
gqd_real operator*(const gqd_real &a, const gqd_real &b) {
	#ifdef USE_SLOPPY_MUL
        return sloppy_mul(a, b);
	#else
		return bf_mul(a, b);
	#endif // USE_SLOPPY_MUL	
}

 __device__
gqd_real sqr(const gqd_real &a) 
{
	double p0, p1, p2, p3, p4, p5;
	double q0, q1, q2, q3;
	double s0, s1;
	double t0, t1;

	p0 = two_sqr(a.x, q0);
	p1 = two_prod(2.0 * a.x, a.y, q1);
	p2 = two_prod(2.0 * a.x, a.z, q2);
	p3 = two_sqr(a.y, q3);

	p1 = two_sum(q0, p1, q0);

	q0 = two_sum(q0, q1, q1);
	p2 = two_sum(p2, p3, p3);

	s0 = two_sum(q0, p2, t0);
	s1 = two_sum(q1, p3, t1);

	s1 = two_sum(s1, t0, t0);
	t0 += t1;

	s1 = quick_two_sum(s1, t0, t0);
	p2 = quick_two_sum(s0, s1, t1);
	p3 = quick_two_sum(t1, t0, q0);

	p4 = 2.0 * a.x * a.w;
	p5 = 2.0 * a.y * a.z;

	p4 = two_sum(p4, p5, p5);
	q2 = two_sum(q2, q3, q3);

	t0 = two_sum(p4, q2, t1);
	t1 = t1 + p5 + q3;

	p3 = two_sum(p3, t0, p4);
	p4 = p4 + q0 + t1;

	renorm(p0, p1, p2, p3, p4);
	return make_qd(p0, p1, p2, p3);
}

/** divisions */
__device__
gqd_real sloppy_div(const gqd_real &a, const gqd_real &b) 
{
	double q0, q1, q2, q3;

	gqd_real r;

	q0 = a.x / b.x;
	r = a - (b * q0);

	q1 = r.x / b.x;
	r = r - (b * q1);

	q2 = r.x / b.x;
	r = r - (b * q2);

	q3 = r.x / b.x;

	renorm(q0, q1, q2, q3);

	return make_qd(q0, q1, q2, q3);
}

__device__
gqd_real operator/(const gqd_real &a, const gqd_real &b) 
{
	return sloppy_div(a, b);
}

/* double / quad-double */
__device__
gqd_real operator/(double a, const gqd_real &b) 
{
	return make_qd(a) / b;
}

/* quad-double / double */
__device__
gqd_real operator/( const gqd_real &a, double b )
{
	return a/make_qd(b);
}

/********** Miscellaneous **********/
__host__ __device__
gqd_real abs(const gqd_real &a) 
{
	return (a.x < 0.0) ? (negative(a)) : (a);
}

/********************** Simple Conversion ********************/
__host__ __device__
double to_double(const gqd_real &a) 
{
	return a.x;
}

__host__ __device__
gqd_real ldexp(const gqd_real &a, int n) 
{
	return make_qd(ldexp(a.x, n), ldexp(a.y, n), 
		ldexp(a.z, n), ldexp(a.w, n));
}

__device__
gqd_real inv(const gqd_real &qd) 
{
	return 1.0 / qd;
}


/********** Greater-Than Comparison ***********/

__host__ __device__
bool operator>=(const gqd_real &a, const gqd_real &b) 
{
	return (a.x > b.x || 
		(a.x == b.x && (a.y > b.y ||
		(a.y == b.y && (a.z > b.z ||
		(a.z == b.z && a.w >= b.w))))));
}

/********** Greater-Than-Or-Equal-To Comparison **********/
/*
__device__
bool operator>=(const gqd_real &a, double b) {
  return (a.x > b || (a.x == b && a.y >= 0.0));
}

__device__
bool operator>=(double a, const gqd_real &b) {
  return (b <= a);
}

__device__
bool operator>=(const gqd_real &a, const gqd_real &b) {
  return (a.x > b.x || 
          (a.x == b.x && (a.y > b.y ||
                            (a.y == b.y && (a.z > b.z ||
                                              (a.z == b.z && a.w >= b.w))))));
}
*/

/********** Less-Than Comparison ***********/
__host__ __device__
bool operator<(const gqd_real &a, double b) {
	return (a.x < b || (a.x == b && a.y < 0.0));
}

__host__ __device__
bool operator<(const gqd_real &a, const gqd_real &b) {
	return (a.x < b.x ||
		(a.x == b.x && (a.y < b.y ||
		(a.y == b.y && (a.z < b.z ||
		(a.z == b.z && a.w < b.w))))));
}

__host__ __device__
bool operator<=(const gqd_real &a, const gqd_real &b) {
  return (a.x < b.x || 
          (a.x == b.x && (a.y < b.y ||
                            (a.y == b.y && (a.z < b.z ||
                                              (a.z == b.z && a.w <= b.w))))));
}

__host__ __device__
bool operator==(const gqd_real &a, const gqd_real &b) {
  return (a.x == b.x && a.y == b.y && 
          a.z == b.z && a.w == b.w);
}



/********** Less-Than-Or-Equal-To Comparison **********/
__device__
bool operator<=(const gqd_real &a, double b) {
	return (a.x < b || (a.x == b && a.y <= 0.0));
}

/*
__device__
bool operator<=(double a, const gqd_real &b) {
	return (b >= a);
}
*/

/*
__device__
bool operator<=(const gqd_real &a, const gqd_real &b) {
  return (a.x < b.x || 
          (a.x == b.x && (a.y < b.y ||
                            (a.y == b.y && (a.z < b.z ||
                                              (a.z == b.z && a.w <= b.w))))));
}
*/

/********** Greater-Than-Or-Equal-To Comparison **********/
__device__
bool operator>=(const gqd_real &a, double b) {
  return (a.x > b || (a.x == b && a.y >= 0.0));
}

__device__
bool operator<=(double a, const gqd_real &b) {
        return (b >= a);
}


__device__
bool operator>=(double a, const gqd_real &b) {
  return (b <= a);
}


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
bool operator>(const gqd_real &a, double b) {
	return (a.x > b || (a.x == b && a.y > 0.0));
}

__host__ __device__
bool operator<(double a, const gqd_real &b) {
	return (b > a);
}

__host__ __device__
bool operator>(double a, const gqd_real &b) {
	return (b < a);
}

__host__ __device__ 
bool operator>(const gqd_real &a, const gqd_real &b) {
	return (a.x > b.x ||
		(a.x == b.x && (a.y > b.y ||
		(a.y == b.y && (a.z > b.z ||
		(a.z == b.z && a.w > b.w))))));
}


__host__ __device__
bool is_zero( const gqd_real &x ) 
{
	return (x.x == 0.0);
}

__host__ __device__
bool is_one( const gqd_real &x ) 
{
	return (x.x == 1.0 && x.y == 0.0 && x.z == 0.0 && x.w == 0.0);
}

__host__ __device__
bool is_positive( const gqd_real &x )
{
	return (x.x > 0.0);
}

__host__ __device__
bool is_negative( const gqd_real &x ) 
{
	return (x.x < 0.0);
}

__device__
gqd_real nint(const gqd_real &a) {
  double x0, x1, x2, x3;

  x0 = nint(a.x);
  x1 = x2 = x3 = 0.0;

  if (x0 == a.x) {
    /* First double is already an integer. */
    x1 = nint(a.y);

    if (x1 == a.y) {
      /* Second double is already an integer. */
      x2 = nint(a.z);
      
      if (x2 == a.z) {
        /* Third double is already an integer. */
        x3 = nint(a.w);
      } else {
        if (fabs(x2 - a.z) == 0.5 && a.w < 0.0) {
          x2 -= 1.0;
        }
      }

    } else {
      if (fabs(x1 - a.y) == 0.5 && a.z < 0.0) {
          x1 -= 1.0;
      }
    }

  } else {
    /* First double is not an integer. */
      if (fabs(x0 - a.x) == 0.5 && a.y < 0.0) {
          x0 -= 1.0;
      }
  }
  
  renorm(x0, x1, x2, x3);
  return make_qd(x0, x1, x2, x3);
}

__device__
gqd_real fabs(const gqd_real &a) {
	return abs(a);
}


#endif


