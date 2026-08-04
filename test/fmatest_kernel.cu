/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
 *
 * Accuracy test for the branch-free multi-word fused multiply-add added
 * in gdtq-0.0.3:
 *
 *     dw_fma  (double-word : gdd_real, gds_real)
 *     tw_fma  (triple-word : gtd_real, gts_real)
 *     qw_fma  (quad-word   : gqd_real, gqs_real)
 *
 * and for the division and square root built on top of them.
 *
 * The reference value is not another floating-point type: every quantity
 * that appears in  a * b + c  is expanded on the host into an *exact*
 * non-overlapping expansion of doubles (Shewchuk's grow_expansion), the
 * value the GPU produced is subtracted from it exactly, and what remains
 * is the true rounding error.  Products of two limbs are exact in one
 * double for the float-based types and in two doubles (two_prod) for the
 * double-based ones, so the reference is exact for all six classes.
 *
 * The file is self-contained: it defines its own main() and compiles to
 * a standalone binary.
 *
 *     nvcc -arch=sm_XX -I../inc fmatest_kernel.cu -o fmatest
 */
#ifndef __NV_NO_VECTOR_DEPRECATION_DIAG
#define __NV_NO_VECTOR_DEPRECATION_DIAG
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_header.cu"
#include "gqd.cu"
#include "gqs.cu"

#define N_ELEMS 20000
#define BLOCK   128

static int n_fail = 0;

/*==================================================================
 * Device kernels (templated over the multi-word type)
 *==================================================================*/

template <class T>
__global__ void k_fma(const T *a, const T *b, const T *c, T *z, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) z[i] = fma(a[i], b[i], c[i]);
}

template <class T>
__global__ void k_muladd(const T *a, const T *b, const T *c, T *z, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) z[i] = a[i] * b[i] + c[i];
}

template <class T>
__global__ void k_div(const T *a, const T *b, T *z, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) z[i] = a[i] / b[i];
}

template <class T>
__global__ void k_sqrt(const T *a, T *z, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) z[i] = sqrt(a[i]);
}

/*==================================================================
 * Host-side exact expansion arithmetic
 *==================================================================*/

#define EXP_MAX 128

static double h_two_sum(double a, double b, double &err) {
	double s = a + b;
	double bb = s - a;
	err = (a - (s - bb)) + (b - bb);
	return s;
}

static double h_quick_two_sum(double a, double b, double &err) {
	double s = a + b;
	err = b - (s - a);
	return s;
}

static double h_two_prod(double a, double b, double &err) {
	double p = a * b;
	err = fma(a, b, -p);       /* exact: host C99 fma */
	return p;
}

/* Add the double b to the non-overlapping expansion e[0..n-1], kept in
   increasing order of magnitude.  Returns the new length. */
static int grow(double *e, int n, double b) {
	double Q = b;
	int m = 0;
	for (int i = 0; i < n; i++) {
		double h;
		Q = h_two_sum(Q, e[i], h);
		if (h != 0.0) e[m++] = h;
	}
	if (Q != 0.0) e[m++] = Q;
	return m;
}

/* |a*b + c - z| computed exactly, all operands given as limb arrays. */
static double exact_residual(const double *a, int na,
                             const double *b, int nb,
                             const double *c, int nc,
                             const double *z, int nz) {
	double e[EXP_MAX];
	int n = 0;

	for (int i = 0; i < na; i++)
		for (int j = 0; j < nb; j++) {
			double err;
			double p = h_two_prod(a[i], b[j], err);
			n = grow(e, n, err);
			n = grow(e, n, p);
		}
	for (int k = 0; k < nc; k++) n = grow(e, n, c[k]);
	for (int k = 0; k < nz; k++) n = grow(e, n, -z[k]);

	double s = 0.0;
	for (int i = 0; i < n; i++) s += e[i];      /* smallest first */
	return fabs(s);
}

/*==================================================================
 * Limb access: widen any of the six GPU types into doubles
 *==================================================================*/

static int limbs(const gdd_real &a, double *w) { w[0]=a.x; w[1]=a.y; return 2; }
static int limbs(const gtd_real &a, double *w) { w[0]=a.x; w[1]=a.y; w[2]=a.z; return 3; }
static int limbs(const gqd_real &a, double *w) { w[0]=a.x; w[1]=a.y; w[2]=a.z; w[3]=a.w; return 4; }
static int limbs(const gds_real &a, double *w) { w[0]=a.x; w[1]=a.y; return 2; }
static int limbs(const gts_real &a, double *w) { w[0]=a.x; w[1]=a.y; w[2]=a.z; return 3; }
static int limbs(const gqs_real &a, double *w) { w[0]=a.x; w[1]=a.y; w[2]=a.z; w[3]=a.w; return 4; }

template <class T>
static double residual(const T &a, const T &b, const T &c, const T &z) {
	double wa[4], wb[4], wc[4], wz[4];
	int na = limbs(a, wa), nb = limbs(b, wb);
	int nc = limbs(c, wc), nz = limbs(z, wz);
	return exact_residual(wa, na, wb, nb, wc, nc, wz, nz);
}

template <class T>
static double lead(const T &a) { double w[4]; limbs(a, w); return fabs(w[0]); }

/*==================================================================
 * Random operands
 *==================================================================*/

static double urand(void) { return 2.0 * ((double)rand() / RAND_MAX) - 1.0; }

/* Build a random non-overlapping expansion of n limbs, each limb a
   factor 2^-p smaller than the one before it. */
static void make_limbs(double *c, int n, int p) {
	double scale = 1.0;
	for (int i = 0; i < n; i++) { c[i] = urand() * scale; scale = ldexp(scale, -p); }

	/* renormalize, bottom-up then top-down (no zero-shifting branches) */
	double s = c[n-1];
	for (int i = n - 2; i >= 0; i--) s = h_quick_two_sum(c[i], s, c[i+1]);
	c[0] = s;
	for (int i = 0; i < n - 1; i++) c[i] = h_quick_two_sum(c[i], c[i+1], c[i+1]);
}

static void fill(gdd_real *v, int n) { for (int i=0;i<n;i++){ double c[2]; make_limbs(c,2,53); v[i]=make_dd(c[0],c[1]); } }
static void fill(gtd_real *v, int n) { for (int i=0;i<n;i++){ double c[3]; make_limbs(c,3,53); v[i]=make_td(c[0],c[1],c[2]); } }
static void fill(gqd_real *v, int n) { for (int i=0;i<n;i++){ double c[4]; make_limbs(c,4,53); v[i]=make_qd(c[0],c[1],c[2],c[3]); } }
static void fill(gds_real *v, int n) { for (int i=0;i<n;i++){ double c[2]; make_limbs(c,2,24); v[i]=make_ds((float)c[0],(float)c[1]); } }
static void fill(gts_real *v, int n) { for (int i=0;i<n;i++){ double c[3]; make_limbs(c,3,24); v[i]=make_ts((float)c[0],(float)c[1],(float)c[2]); } }
static void fill(gqs_real *v, int n) { for (int i=0;i<n;i++){ double c[4]; make_limbs(c,4,24); v[i]=make_qs((float)c[0],(float)c[1],(float)c[2],(float)c[3]); } }

/* Make every element positive (for sqrt) / bounded away from zero (for div). */
static void bump(gdd_real *v, int n) { for (int i=0;i<n;i++){ v[i].x = fabs(v[i].x) + 2.0; } }
static void bump(gtd_real *v, int n) { for (int i=0;i<n;i++){ v[i].x = fabs(v[i].x) + 2.0; } }
static void bump(gqd_real *v, int n) { for (int i=0;i<n;i++){ v[i].x = fabs(v[i].x) + 2.0; } }
static void bump(gds_real *v, int n) { for (int i=0;i<n;i++){ v[i].x = fabsf(v[i].x) + 2.0f; } }
static void bump(gts_real *v, int n) { for (int i=0;i<n;i++){ v[i].x = fabsf(v[i].x) + 2.0f; } }
static void bump(gqs_real *v, int n) { for (int i=0;i<n;i++){ v[i].x = fabsf(v[i].x) + 2.0f; } }

/*==================================================================
 * Test drivers
 *==================================================================*/

static void report(const char *what, double worst, double tol) {
	int ok = (worst <= tol);
	printf("  %-32s  max = %7.3f eps   (tol %4.1f)  %s\n",
	       what, worst, tol, ok ? "ok" : "FAILED");
	if (!ok) n_fail++;
}

/* eps of the type, as a double.  eps = 2^-(p*limbs - guard). */
template <class T> struct Eps;
template <> struct Eps<gdd_real> { static double v() { return ldexp(1.0, -104); } };
template <> struct Eps<gtd_real> { static double v() { return ldexp(1.0, -156); } };
template <> struct Eps<gqd_real> { static double v() { return ldexp(1.0, -209); } };
template <> struct Eps<gds_real> { static double v() { return ldexp(1.0,  -46); } };
template <> struct Eps<gts_real> { static double v() { return ldexp(1.0,  -69); } };
template <> struct Eps<gqs_real> { static double v() { return ldexp(1.0,  -93); } };

template <class T>
static void test_type(const char *name, double tol) {
	const int n = N_ELEMS;
	const size_t sz = n * sizeof(T);
	const double eps = Eps<T>::v();

	T *ha = (T *)malloc(sz), *hb = (T *)malloc(sz);
	T *hc = (T *)malloc(sz), *hz = (T *)malloc(sz);
	T *da, *db, *dc, *dz;
	GPUMALLOC((void **)&da, sz); GPUMALLOC((void **)&db, sz);
	GPUMALLOC((void **)&dc, sz); GPUMALLOC((void **)&dz, sz);

	fill(ha, n); fill(hb, n); fill(hc, n);
	TOGPU(da, ha, sz); TOGPU(db, hb, sz); TOGPU(dc, hc, sz);

	int grid = (n + BLOCK - 1) / BLOCK;
	char buf[128];

	/* ---- fma(a,b,c) and the unfused a*b+c ---- */
	k_fma<T><<<grid, BLOCK>>>(da, db, dc, dz, n);
	cudaDeviceSynchronize(); cutilCheckMsg("k_fma");
	FROMGPU(hz, dz, sz);

	double worst = 0.0;
	for (int i = 0; i < n; i++) {
		double scale = lead(ha[i]) * lead(hb[i]) + lead(hc[i]);
		if (scale == 0.0) continue;
		double u = residual(ha[i], hb[i], hc[i], hz[i]) / (scale * eps);
		if (u > worst) worst = u;
	}
	sprintf(buf, "%s  fma(a,b,c)", name);
	report(buf, worst, tol);

	k_muladd<T><<<grid, BLOCK>>>(da, db, dc, dz, n);
	cudaDeviceSynchronize(); cutilCheckMsg("k_muladd");
	FROMGPU(hz, dz, sz);

	worst = 0.0;
	for (int i = 0; i < n; i++) {
		double scale = lead(ha[i]) * lead(hb[i]) + lead(hc[i]);
		if (scale == 0.0) continue;
		double u = residual(ha[i], hb[i], hc[i], hz[i]) / (scale * eps);
		if (u > worst) worst = u;
	}
	sprintf(buf, "%s  a*b+c (unfused)", name);
	report(buf, worst, tol);

	/* ---- division:  b*(a/b) must reproduce a ---- */
	bump(hb, n);
	TOGPU(db, hb, sz);
	k_div<T><<<grid, BLOCK>>>(da, db, dz, n);
	cudaDeviceSynchronize(); cutilCheckMsg("k_div");
	FROMGPU(hz, dz, sz);

	worst = 0.0;
	for (int i = 0; i < n; i++) {
		double scale = lead(ha[i]);
		if (scale == 0.0) continue;
		/* residual = q*b - a, computed exactly */
		double wq[4], wb[4], wa[4];
		int nq = limbs(hz[i], wq);
		int nb = limbs(hb[i], wb);
		int na = limbs(ha[i], wa);
		for (int k = 0; k < na; k++) wa[k] = -wa[k];
		double u = exact_residual(wq, nq, wb, nb, wa, na, (double *)0, 0)
		         / (scale * eps);
		if (u > worst) worst = u;
	}
	sprintf(buf, "%s  a/b", name);
	report(buf, worst, tol);

	/* ---- square root:  sqrt(a)^2 must reproduce a ---- */
	bump(ha, n);
	TOGPU(da, ha, sz);
	k_sqrt<T><<<grid, BLOCK>>>(da, dz, n);
	cudaDeviceSynchronize(); cutilCheckMsg("k_sqrt");
	FROMGPU(hz, dz, sz);

	worst = 0.0;
	for (int i = 0; i < n; i++) {
		double scale = lead(ha[i]);
		if (scale == 0.0) continue;
		double wa[4]; int na = limbs(ha[i], wa);
		for (int k = 0; k < na; k++) wa[k] = -wa[k];
		double ws[4]; int ns = limbs(hz[i], ws);
		double u = exact_residual(ws, ns, ws, ns, wa, na, (double *)0, 0)
		         / (scale * eps);
		if (u > worst) worst = u;
	}
	sprintf(buf, "%s  sqrt(a)", name);
	report(buf, worst, tol);

	GPUFREE(da); GPUFREE(db); GPUFREE(dc); GPUFREE(dz);
	free(ha); free(hb); free(hc); free(hz);
}

int main(void) {
	GDDStart(0);
	GTDStart(0);
	GQDStart(0);
	GDSStart(0);
	GTSStart(0);
	GQSStart(0);

	srand(20260804);

	printf("gdtq -- dw_fma / tw_fma / qw_fma accuracy test\n");
	printf("%d elements per case; errors are relative to |a*b| + |c|,\n", N_ELEMS);
	printf("in units of the type's eps.\n\n");

	printf("double-based classes\n");
	test_type<gdd_real>("gdd_real (dw_fma)", 8.0);
	test_type<gtd_real>("gtd_real (tw_fma)", 8.0);
	test_type<gqd_real>("gqd_real (qw_fma)", 8.0);

	printf("\nfloat-based classes\n");
	test_type<gds_real>("gds_real (dw_fma)", 8.0);
	test_type<gts_real>("gts_real (tw_fma)", 8.0);
	test_type<gqs_real>("gqs_real (qw_fma)", 8.0);

	printf("\n%s -- %d failure(s)\n", n_fail == 0 ? "PASS" : "FAIL", n_fail);
	return n_fail == 0 ? 0 : 1;
}
