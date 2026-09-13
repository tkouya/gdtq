/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya.
 *
 * fmaref -- bitwise conformance of the multi-word fused multiply-add with
 * the machine-proved reference formulation (gdtq 0.0.4 test).
 *
 * The host side is an operation-by-operation transcription of the scalar
 * reference fma_ref.c of the ACS2026 paper (T. Kouya, "Branch-free fused
 * multiply-add for multi-component-type multiple-precision arithmetic"),
 * whose DW/TW/QW routines match the FPANVerifier netlists:
 *
 *   DW  17 flops  1 normalization FastTwoSum        |err| <= 35 u^2 (|xy|+|c|)
 *   TW  72 flops  3 passes (FastTwoSum where proved) |err| <= 187 u^3 (|xy|+|c|)
 *   QW 176 flops  5 passes                           |err| <= 822 u^4 (|xy|+|c|)
 *
 * (certified bounds: FPANVerifier anchor-relative 34/184/812 plus the
 * analytic input-relative conversion), and the div/sqrt-safe variants with
 * every FastTwoSum demoted to TwoSum and the same pass counts
 * (20 / 84 / 206 flops).  The reference is written once as a template over
 * the base type, so it serves the double-based and the float-based classes.
 *
 * Checked on the GPU, for all six classes:
 *   fma(a, b, c)          == reference(a, b, c)
 *   fma(a, q, c)          == reference(a, {q, 0, ...}, c)   (zero folding)
 *   Xw_fma_safe(a, q, c)  == safe reference(a, {q, 0, ...}, c)
 * on random non-overlapping operands (wide exponent range, zero words,
 * near-complete cancellation c ~ -a*b).  Any mismatch is an error.
 *
 * Host code must not contract a*b+c: compile with -Xcompiler
 * -ffp-contract=off (the device side uses -fmad=false as everywhere).
 *
 *     nvcc -fmad=false -Xcompiler -ffp-contract=off -arch=sm_XX -I../inc \
 *          fmaref_kernel.cu -o fmaref
 */
#ifndef __NV_NO_VECTOR_DEPRECATION_DIAG
#define __NV_NO_VECTOR_DEPRECATION_DIAG
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "cuda_header.cu"
#include "gqd.cu"
#include "gqs.cu"

#define N_TRIALS 200000
#define BLOCK    128

static int n_fail = 0;

/*==================================================================
 * Host reference (fma_ref.c, templated over the base type)
 *==================================================================*/

static inline double ref_fma1(double a, double b, double c) { return fma(a, b, c); }
static inline float  ref_fma1(float a, float b, float c)    { return fmaf(a, b, c); }

template <class T> static inline void r_two_sum(T a, T b, T *s, T *e) {
	T t;
	*s = a + b;
	t  = *s - a;
	*e = (a - (*s - t)) + (b - t);
}
template <class T> static inline void r_fast_two_sum(T a, T b, T *s, T *e) {
	*s = a + b;
	*e = b - (*s - a);
}
template <class T> static inline void r_two_prod(T a, T b, T *p, T *e) {
	*p = a * b;
	*e = ref_fma1(a, b, -*p);
}
/* SAFE: FastTwoSum demoted to TwoSum (div/sqrt-safe variant) */
template <class T, bool SAFE> static inline void r_f2s(T a, T b, T *s, T *e) {
	if (SAFE) r_two_sum(a, b, s, e); else r_fast_two_sum(a, b, s, e);
}

template <class T, bool SAFE>
static void ref_dw(const T *x, const T *y, const T *c, T *z) {
	T P00, E00, P01, P10, l, v, w, s, t, tp;
	r_two_prod(x[0], y[0], &P00, &E00);
	P01 = x[0] * y[1];
	P10 = x[1] * y[0];
	l   = P01 + P10;
	v   = E00 + c[1];
	w   = v + l;
	r_two_sum(P00, c[0], &s, &t);
	tp  = t + w;
	r_f2s<T, SAFE>(s, tp, &z[0], &z[1]);
}

template <class T, bool SAFE>
static void ref_tw(const T *x, const T *y, const T *c, T *z) {
	T P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
	T A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;
	r_two_prod(x[0], y[0], &P00, &E00);
	r_two_prod(x[0], y[1], &P01, &E01);
	r_two_prod(x[1], y[0], &P10, &E10);
	P02 = x[0] * y[2];
	P11 = x[1] * y[1];
	P20 = x[2] * y[0];
	sg  = (P02 + P20) + P11;
	G   = ((E01 + E10) + sg) + c[2];
	r_two_sum(P01, P10, &A, &q1);
	r_two_sum(A,   E00, &A, &q2);
	r_two_sum(A,   c[1], &A, &q3);
	G   = G + ((q1 + q2) + q3);
	r_two_sum(P00, c[0], &B, &r);
	r_two_sum(r, A, &m1, &m2);
	m2  = m2 + G;
	r_f2s<T, SAFE>(B, m1, &w0, &w1);            /* pass 1 */
	r_two_sum(w1, m2, &w1, &w2);
	r_two_sum(w0, w1, &w0, &w1);                /* pass 2 */
	r_f2s<T, SAFE>(w1, w2, &w1, &w2);
	r_f2s<T, SAFE>(w0, w1, &z[0], &w1);         /* pass 3 */
	r_f2s<T, SAFE>(w1, w2, &z[1], &z[2]);
}

template <class T, bool SAFE>
static void ref_qw(const T *x, const T *y, const T *c, T *z) {
	T P00, E00, P01, E01, P10, E10, P02, E02, P11, E11, P20, E20;
	T P03, P12, P21, P30, D, B, r;
	T A1, f1, f2, f3, f4;
	T A2, g1, g2, g3, g4, g5, g6, g7, g8, g9;
	T A3, t1, t2, t3, t4;
	T w0, w1, w2, w3;
	r_two_prod(x[0], y[0], &P00, &E00);
	r_two_prod(x[0], y[1], &P01, &E01);
	r_two_prod(x[1], y[0], &P10, &E10);
	r_two_prod(x[0], y[2], &P02, &E02);
	r_two_prod(x[1], y[1], &P11, &E11);
	r_two_prod(x[2], y[0], &P20, &E20);
	P03 = x[0] * y[3];
	P12 = x[1] * y[2];
	P21 = x[2] * y[1];
	P30 = x[3] * y[0];
	D   = (P03 + P30) + (P12 + P21);
	r_two_sum(P00, c[0], &B, &r);
	r_two_sum(P01, P10,  &A1, &f1);
	r_two_sum(A1,  E00,  &A1, &f2);
	r_two_sum(A1,  c[1], &A1, &f3);
	r_two_sum(A1,  r,    &A1, &f4);
	r_two_sum(P02, P20,  &A2, &g1);
	r_two_sum(A2,  P11,  &A2, &g2);
	r_two_sum(E01, E10,  &E01, &g4);
	r_two_sum(A2,  E01,  &A2, &g3);
	r_two_sum(A2,  c[2], &A2, &g5);
	r_two_sum(A2,  f1,   &A2, &g6);
	r_two_sum(A2,  f2,   &A2, &g7);
	r_two_sum(A2,  f3,   &A2, &g8);
	r_two_sum(A2,  f4,   &A2, &g9);
	t1 = E02 + E20;
	t2 = E11 + D;
	t3 = (t1 + t2) + c[3];
	t1 = g1 + g2;
	t2 = g3 + g4;
	t1 = t1 + t2;
	t2 = g6 + g7;
	t4 = g8 + g9;
	t2 = t2 + t4;
	t1 = t1 + t2;
	t1 = t1 + g5;
	A3 = t3 + t1;
	r_f2s<T, SAFE>(B, A1, &w0, &w1);            /* pass 1 */
	r_two_sum(w1, A2, &w1, &w2);
	r_two_sum(w2, A3, &w2, &w3);
	r_two_sum(w0, w1, &w0, &w1);                /* pass 2 */
	r_two_sum(w1, w2, &w1, &w2);
	r_f2s<T, SAFE>(w2, w3, &w2, &w3);
	r_two_sum(w0, w1, &w0, &w1);                /* pass 3 */
	r_f2s<T, SAFE>(w1, w2, &w1, &w2);
	r_f2s<T, SAFE>(w2, w3, &w2, &w3);
	r_f2s<T, SAFE>(w0, w1, &w0, &w1);           /* pass 4 */
	r_f2s<T, SAFE>(w1, w2, &w1, &w2);
	r_f2s<T, SAFE>(w2, w3, &w2, &w3);
	r_f2s<T, SAFE>(w0, w1, &z[0], &w1);         /* pass 5 */
	r_f2s<T, SAFE>(w1, w2, &z[1], &w2);
	r_f2s<T, SAFE>(w2, w3, &z[2], &z[3]);
}

/*==================================================================
 * Type traits: limb access, reference, word count
 *==================================================================*/

template <class G> struct FT;
#define FT_DEF(G, TT, KK)                                                     \
	template <> struct FT<G> {                                                \
		typedef TT T; static const int K = KK;                                \
		static const char *name() { return #G; }                               \
	};
FT_DEF(gdd_real, double, 2) FT_DEF(gtd_real, double, 3) FT_DEF(gqd_real, double, 4)
FT_DEF(gds_real, float, 2)  FT_DEF(gts_real, float, 3)  FT_DEF(gqs_real, float, 4)
#undef FT_DEF

template <class G> static void get(const G &g, typename FT<G>::T *w) {
	memcpy(w, &g, FT<G>::K * sizeof(typename FT<G>::T));
}
template <class G> static G put(const typename FT<G>::T *w) {
	G g;
	memset(&g, 0, sizeof g);
	memcpy(&g, w, FT<G>::K * sizeof(typename FT<G>::T));
	return g;
}

template <class T, bool SAFE>
static void ref(int K, const T *x, const T *y, const T *c, T *z) {
	if (K == 2) ref_dw<T, SAFE>(x, y, c, z);
	else if (K == 3) ref_tw<T, SAFE>(x, y, c, z);
	else ref_qw<T, SAFE>(x, y, c, z);
}

/*==================================================================
 * Device kernels
 *==================================================================*/

__device__ gdd_real fma_safe_(const gdd_real &a, double q, const gdd_real &c) { return dw_fma_safe(a, q, c); }
__device__ gtd_real fma_safe_(const gtd_real &a, double q, const gtd_real &c) { return tw_fma_safe(a, q, c); }
__device__ gqd_real fma_safe_(const gqd_real &a, double q, const gqd_real &c) { return qw_fma_safe(a, q, c); }
__device__ gds_real fma_safe_(const gds_real &a, float q, const gds_real &c)  { return dw_fma_safe(a, q, c); }
__device__ gts_real fma_safe_(const gts_real &a, float q, const gts_real &c)  { return tw_fma_safe(a, q, c); }
__device__ gqs_real fma_safe_(const gqs_real &a, float q, const gqs_real &c)  { return qw_fma_safe(a, q, c); }

__device__ gdd_real xw_fma(const gdd_real &a, const gdd_real &b, const gdd_real &c) { return dw_fma(a, b, c); }
__device__ gtd_real xw_fma(const gtd_real &a, const gtd_real &b, const gtd_real &c) { return tw_fma(a, b, c); }
__device__ gqd_real xw_fma(const gqd_real &a, const gqd_real &b, const gqd_real &c) { return qw_fma(a, b, c); }
__device__ gds_real xw_fma(const gds_real &a, const gds_real &b, const gds_real &c) { return dw_fma(a, b, c); }
__device__ gts_real xw_fma(const gts_real &a, const gts_real &b, const gts_real &c) { return tw_fma(a, b, c); }
__device__ gqs_real xw_fma(const gqs_real &a, const gqs_real &b, const gqs_real &c) { return qw_fma(a, b, c); }

__device__ gdd_real xw_fma(const gdd_real &a, double q, const gdd_real &c) { return dw_fma(a, q, c); }
__device__ gtd_real xw_fma(const gtd_real &a, double q, const gtd_real &c) { return tw_fma(a, q, c); }
__device__ gqd_real xw_fma(const gqd_real &a, double q, const gqd_real &c) { return qw_fma(a, q, c); }
__device__ gds_real xw_fma(const gds_real &a, float q, const gds_real &c)  { return dw_fma(a, q, c); }
__device__ gts_real xw_fma(const gts_real &a, float q, const gts_real &c)  { return tw_fma(a, q, c); }
__device__ gqs_real xw_fma(const gqs_real &a, float q, const gqs_real &c)  { return qw_fma(a, q, c); }

template <class G, class T>
__global__ void k_fmas(const G *a, const G *b, const T *q, const G *c,
                       G *z_full, G *z_scal, G *z_safe, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	z_full[i] = xw_fma(a[i], b[i], c[i]);
	z_scal[i] = xw_fma(a[i], q[i], c[i]);
	z_safe[i] = fma_safe_(a[i], q[i], c[i]);
}

/*==================================================================
 * Random non-overlapping operands
 *==================================================================*/

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rnd64(void) {           /* xorshift64* */
	rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25; rng_state ^= rng_state >> 27;
	return rng_state * 2685821657736338717ull;
}
static double urand(void) { return (double)(rnd64() >> 11) * 0x1p-53; }   /* [0,1) */

/* K-word expansion with |w[i+1]| <= ulp(w[i]) / 2; now and then a zero
   word or an exact power of two */
template <class T>
static void rand_expansion(int K, int emin, int emax, T *w) {
	const int p = (sizeof(T) == 8) ? 53 : 24;
	const int e = emin + (int)(urand() * (emax - emin + 1));
	T lead = (T)ldexp(0.5 + 0.5 * urand(), e);
	if (rnd64() & 1) lead = -lead;
	if ((rnd64() & 63) == 0) lead = (T)ldexp(1.0, e);
	w[0] = lead;
	for (int i = 1; i < K; i++) {
		const T prev = w[i - 1];
		if (prev == 0 || (rnd64() & 31) == 0) { w[i] = 0; continue; }
		int ex; frexp((double)prev, &ex);
		/* |w[i]| < 2^(ex-p-1) <= ulp(prev)/2, sometimes much smaller */
		const int drop = (rnd64() & 7) == 0 ? (int)(urand() * 10) : 0;
		T v = (T)ldexp(urand() - 0.5, ex - p - drop);
		/* keep it exactly representable and strictly below ulp/2 */
		w[i] = v;
	}
}

/*==================================================================
 * One class
 *==================================================================*/

template <class G>
static void test_class(void) {
	typedef typename FT<G>::T T;
	const int K = FT<G>::K;
	const int n = N_TRIALS;
	const bool fl = sizeof(T) == 4;
	const int er = fl ? 20 : 200;         /* exponent range of the operands */

	G *ha = (G *)malloc(n * sizeof(G)), *hb = (G *)malloc(n * sizeof(G)), *hc = (G *)malloc(n * sizeof(G));
	T *hq = (T *)malloc(n * sizeof(T));
	G *hz[3];
	for (int k = 0; k < 3; k++) hz[k] = (G *)malloc(n * sizeof(G));

	for (int i = 0; i < n; i++) {
		T a[4], b[4], c[4], z[4], zero[4] = {0, 0, 0, 0};
		rand_expansion<T>(K, -er, er, a);
		rand_expansion<T>(K, -er, er, b);
		const int kind = (int)(rnd64() % 4);
		if (kind == 0) {
			rand_expansion<T>(K, -2 * er, 2 * er, c);
		} else {
			/* c close to -(a b): complete or partial cancellation */
			ref<T, false>(K, a, b, zero, z);
			for (int j = 0; j < K; j++) c[j] = -z[j];
			if (kind == 2) c[K - 1] = (T)(c[K - 1] * (T)(1 + urand()));  /* perturbed */
			if (kind == 3) c[0] = (T)(c[0] * (T)(1 + ldexp(urand() - 0.5, -10)));
		}
		ha[i] = put<G>(a); hb[i] = put<G>(b); hc[i] = put<G>(c);
		hq[i] = a[0] != 0 ? b[0] : (T)urand();
	}

	G *da, *db, *dc, *dz[3];
	T *dq;
	GPUMALLOC((void **)&da, n * sizeof(G));
	GPUMALLOC((void **)&db, n * sizeof(G));
	GPUMALLOC((void **)&dc, n * sizeof(G));
	GPUMALLOC((void **)&dq, n * sizeof(T));
	for (int k = 0; k < 3; k++) GPUMALLOC((void **)&dz[k], n * sizeof(G));
	TOGPU(da, ha, n * sizeof(G));
	TOGPU(db, hb, n * sizeof(G));
	TOGPU(dc, hc, n * sizeof(G));
	TOGPU(dq, hq, n * sizeof(T));
	k_fmas<G, T><<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(da, db, dq, dc, dz[0], dz[1], dz[2], n);
	cudaDeviceSynchronize(); cutilCheckMsg("k_fmas");
	for (int k = 0; k < 3; k++) FROMGPU(hz[k], dz[k], n * sizeof(G));

	long mism[3] = {0, 0, 0};
	for (int i = 0; i < n; i++) {
		T a[4], b[4], c[4], y[4] = {0, 0, 0, 0}, r[3][4], g[4];
		get(ha[i], a); get(hb[i], b); get(hc[i], c);
		y[0] = hq[i];
		ref<T, false>(K, a, b, c, r[0]);
		ref<T, false>(K, a, y, c, r[1]);
		ref<T, true >(K, a, y, c, r[2]);
		for (int k = 0; k < 3; k++) {
			get(hz[k][i], g);
			if (memcmp(g, r[k], K * sizeof(T)) != 0) {
				if (mism[k] < 3) {
					printf("    mismatch %s variant %d i=%d:", FT<G>::name(), k, i);
					for (int j = 0; j < K; j++) printf(" %a/%a", (double)g[j], (double)r[k][j]);
					printf("\n");
				}
				mism[k]++;
			}
		}
	}
	static const char *vname[3] = {"fma(a,b,c)", "fma(a,q,c)", "fma_safe(a,q,c)"};
	for (int k = 0; k < 3; k++) {
		printf("  %-9s %-16s %d trials  mismatches = %ld  %s\n", FT<G>::name(), vname[k], n,
		       mism[k], mism[k] ? "FAIL" : "ok");
		if (mism[k]) n_fail++;
	}

	GPUFREE(da); GPUFREE(db); GPUFREE(dc); GPUFREE(dq);
	for (int k = 0; k < 3; k++) { GPUFREE(dz[k]); free(hz[k]); }
	free(ha); free(hb); free(hc); free(hq);
}

/* a*b + c as written; returns 0 for the test values unless the host
   compiler contracted it into an FMA */
static double __attribute__((noinline)) host_mul_add(double a, double b, double c) {
	return a * b + c;
}

int main(void) {
	volatile double va = 1.0 + 0x1p-30, vb = 1.0 - 0x1p-30, vc = -1.0;
	if (host_mul_add(va, vb, vc) != 0.0) {
		printf("the host compiler contracts a*b+c: build with -Xcompiler -ffp-contract=off\n");
		return 1;
	}
	printf("\ngdtq -- bitwise conformance of dw/tw/qw_fma with the machine-proved\n");
	printf("reference (ACS2026 fma_ref.c: DW 17 / TW 72 / QW 176 flops, safe 20 / 84 / 206)\n\n");
	test_class<gdd_real>();
	test_class<gtd_real>();
	test_class<gqd_real>();
	test_class<gds_real>();
	test_class<gts_real>();
	test_class<gqs_real>();
	printf("\n%s -- %d failing variant(s)\n", n_fail == 0 ? "PASS" : "FAIL", n_fail);
	return n_fail == 0 ? 0 : 1;
}
