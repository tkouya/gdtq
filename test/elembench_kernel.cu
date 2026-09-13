/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
 *
 * elembench -- GPU counterpart of the dtq-0.0.3 elementary function
 * benchmark (tests/elem_bench.cpp):
 *
 *   [1] Exception / special-value handling checks (exp / log / sqrt)
 *   [2] Accuracy of sqrt / exp / expm1 / log / log10 / sin / cos and
 *       polyeval for all six precision classes, measured against MPFR
 *       references computed on the host; since 0.0.4 also log1p, tan,
 *       asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, atan2
 *       and pow
 *   [3] Performance (ns per call over 2^20 elements, best of 5 runs),
 *       with a native double baseline
 *
 * The file is self-contained: it defines its own main() and compiles to
 * a standalone binary.
 *
 *     nvcc -arch=sm_XX -I../inc elembench_kernel.cu -lmpfr -lgmp -o elembench
 *
 * Usage: ./elembench [n_accuracy_samples]   (default 200)
 */
#ifndef __NV_NO_VECTOR_DEPRECATION_DIAG
#define __NV_NO_VECTOR_DEPRECATION_DIAG
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpfr.h>

#include "cuda_header.cu"
#include "gqd.cu"
#include "gqs.cu"

#define N_TIME  (1 << 20)    /* elements per timing launch */
#define REPS    5            /* timing repetitions, best (min) kept */
#define BLOCK   128
#define NC      16           /* polyeval: degree 15 */

static int n_acc = 200;      /* accuracy samples per function */
static int n_fail = 0;

/* function selector, uniform per kernel launch (no divergence) */
enum ElemOp { OP_SQRT, OP_EXP, OP_EXPM1, OP_LOG, OP_LOG10,
              OP_SIN, OP_COS,
              /* gdtq 0.0.4 */
              OP_LOG1P, OP_TAN, OP_ASIN, OP_ACOS, OP_ATAN, OP_SINH,
              OP_COSH, OP_TANH, OP_ASINH, OP_ACOSH, OP_ATANH,
              OP_ATAN2, OP_POW,          /* binary: z = f(a, b) */
              OP_N };
#define OP_FIRST_BINARY OP_ATAN2

static const char *op_name[OP_N] =
  { "sqrt", "exp", "expm1", "log", "log10", "sin", "cos",
    "log1p", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "asinh", "acosh", "atanh", "atan2", "pow" };

typedef int (*mpfr_fun1)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
static const mpfr_fun1 op_mpfr[OP_N] =
  { mpfr_sqrt, mpfr_exp, mpfr_expm1, mpfr_log, mpfr_log10,
    mpfr_sin, mpfr_cos,
    mpfr_log1p, mpfr_tan, mpfr_asin, mpfr_acos, mpfr_atan, mpfr_sinh,
    mpfr_cosh, mpfr_tanh, mpfr_asinh, mpfr_acosh, mpfr_atanh,
    NULL, NULL };

/* sampling range per op: {lo, hi, logscale}; for the binary ops the
   first argument (atan2: y, pow: base) */
static const struct { double lo, hi; int logscale; } op_range[OP_N] = {
	{ 1e-3, 1e3, 1 },    /* sqrt  */
	{ -20.0, 20.0, 0 },  /* exp   */
	{ -0.5, 0.5, 0 },    /* expm1 */
	{ 1e-6, 1e6, 1 },    /* log   */
	{ 1e-6, 1e6, 1 },    /* log10 */
	{ -6.28, 6.28, 0 },  /* sin   */
	{ -6.28, 6.28, 0 },  /* cos   */
	{ -0.5, 1.0, 0 },    /* log1p */
	{ -1.57, 1.57, 0 },  /* tan   */
	{ -1.0, 1.0, 0 },    /* asin  */
	{ -1.0, 1.0, 0 },    /* acos  */
	{ -100.0, 100.0, 0 },/* atan  */
	{ -10.0, 10.0, 0 },  /* sinh  */
	{ -10.0, 10.0, 0 },  /* cosh  */
	{ -10.0, 10.0, 0 },  /* tanh  */
	{ -10.0, 10.0, 0 },  /* asinh */
	{ 1.0, 100.0, 1 },   /* acosh */
	{ -0.9, 0.9, 0 },    /* atanh */
	{ -10.0, 10.0, 0 },  /* atan2 (y) */
	{ 0.1, 10.0, 1 },    /* pow (base) */
};
/* second argument of the binary ops: atan2 x in [-10,10], pow exponent
   in [-8,8] */
static const struct { double lo, hi; } op_range2[OP_N - OP_FIRST_BINARY] = {
	{ -10.0, 10.0 }, { -8.0, 8.0 } };

/*==================================================================
 * Device kernels
 *==================================================================*/

/* one kernel per function (the op is a template argument), so that the
   timing of a cheap function is not affected by the register allocation of
   an expensive one */
template <class T, int op>
__global__ void k_elem1(const T *a, const T *b, T *z, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	switch (op) {
	case OP_SQRT:  z[i] = sqrt(a[i]);  break;
	case OP_EXP:   z[i] = exp(a[i]);   break;
	case OP_EXPM1: z[i] = expm1(a[i]); break;
	case OP_LOG:   z[i] = log(a[i]);   break;
	case OP_LOG10: z[i] = log10(a[i]); break;
	case OP_SIN:   z[i] = sin(a[i]);   break;
	case OP_COS:   z[i] = cos(a[i]);   break;
	case OP_LOG1P: z[i] = log1p(a[i]); break;
	case OP_TAN:   z[i] = tan(a[i]);   break;
	case OP_ASIN:  z[i] = asin(a[i]);  break;
	case OP_ACOS:  z[i] = acos(a[i]);  break;
	case OP_ATAN:  z[i] = atan(a[i]);  break;
	case OP_SINH:  z[i] = sinh(a[i]);  break;
	case OP_COSH:  z[i] = cosh(a[i]);  break;
	case OP_TANH:  z[i] = tanh(a[i]);  break;
	case OP_ASINH: z[i] = asinh(a[i]); break;
	case OP_ACOSH: z[i] = acosh(a[i]); break;
	case OP_ATANH: z[i] = atanh(a[i]); break;
	case OP_ATAN2: z[i] = atan2(a[i], b[i]); break;
	case OP_POW:   z[i] = pow(a[i], b[i]);   break;
	}
}

template <class T>
static void k_elem(int grid, int block, const T *a, const T *b, T *z, int n, int op) {
#define ELEM_CASE(OP) case OP: k_elem1<T, OP><<<grid, block>>>(a, b, z, n); break;
	switch (op) {
	ELEM_CASE(OP_SQRT)  ELEM_CASE(OP_EXP)   ELEM_CASE(OP_EXPM1) ELEM_CASE(OP_LOG)
	ELEM_CASE(OP_LOG10) ELEM_CASE(OP_SIN)   ELEM_CASE(OP_COS)   ELEM_CASE(OP_LOG1P)
	ELEM_CASE(OP_TAN)   ELEM_CASE(OP_ASIN)  ELEM_CASE(OP_ACOS)  ELEM_CASE(OP_ATAN)
	ELEM_CASE(OP_SINH)  ELEM_CASE(OP_COSH)  ELEM_CASE(OP_TANH)  ELEM_CASE(OP_ASINH)
	ELEM_CASE(OP_ACOSH) ELEM_CASE(OP_ATANH) ELEM_CASE(OP_ATAN2) ELEM_CASE(OP_POW)
	}
#undef ELEM_CASE
}

template <class T>
__global__ void k_poly(const T *c, const T *x, T *z, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) z[i] = polyeval(c, NC - 1, x[i]);
}

/* native double baseline */
__global__ void k_elem_double(const double *a, const double *b, double *z, int n, int op) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	switch (op) {
	case OP_SQRT:  z[i] = sqrt(a[i]);  break;
	case OP_EXP:   z[i] = exp(a[i]);   break;
	case OP_EXPM1: z[i] = expm1(a[i]); break;
	case OP_LOG:   z[i] = log(a[i]);   break;
	case OP_LOG10: z[i] = log10(a[i]); break;
	case OP_SIN:   z[i] = sin(a[i]);   break;
	case OP_COS:   z[i] = cos(a[i]);   break;
	case OP_LOG1P: z[i] = log1p(a[i]); break;
	case OP_TAN:   z[i] = tan(a[i]);   break;
	case OP_ASIN:  z[i] = asin(a[i]);  break;
	case OP_ACOS:  z[i] = acos(a[i]);  break;
	case OP_ATAN:  z[i] = atan(a[i]);  break;
	case OP_SINH:  z[i] = sinh(a[i]);  break;
	case OP_COSH:  z[i] = cosh(a[i]);  break;
	case OP_TANH:  z[i] = tanh(a[i]);  break;
	case OP_ASINH: z[i] = asinh(a[i]); break;
	case OP_ACOSH: z[i] = acosh(a[i]); break;
	case OP_ATANH: z[i] = atanh(a[i]); break;
	case OP_ATAN2: z[i] = atan2(a[i], b[i]); break;
	case OP_POW:   z[i] = pow(a[i], b[i]);   break;
	}
}

/*==================================================================
 * Type traits and limb access
 *==================================================================*/

template <class T> struct RT;
template <> struct RT<gdd_real> { static constexpr const char *name = "gdd_real";
	static constexpr int limbs = 2, limb_bits = 53; };
template <> struct RT<gtd_real> { static constexpr const char *name = "gtd_real";
	static constexpr int limbs = 3, limb_bits = 53; };
template <> struct RT<gqd_real> { static constexpr const char *name = "gqd_real";
	static constexpr int limbs = 4, limb_bits = 53; };
template <> struct RT<gds_real> { static constexpr const char *name = "gds_real";
	static constexpr int limbs = 2, limb_bits = 24; };
template <> struct RT<gts_real> { static constexpr const char *name = "gts_real";
	static constexpr int limbs = 3, limb_bits = 24; };
template <> struct RT<gqs_real> { static constexpr const char *name = "gqs_real";
	static constexpr int limbs = 4, limb_bits = 24; };

template <class T> static int prec_bits() { return RT<T>::limbs * RT<T>::limb_bits; }

static int limbs_of(const gdd_real &a, double *w) { w[0]=a.x; w[1]=a.y; return 2; }
static int limbs_of(const gtd_real &a, double *w) { w[0]=a.x; w[1]=a.y; w[2]=a.z; return 3; }
static int limbs_of(const gqd_real &a, double *w) { w[0]=a.x; w[1]=a.y; w[2]=a.z; w[3]=a.w; return 4; }
static int limbs_of(const gds_real &a, double *w) { w[0]=a.x; w[1]=a.y; return 2; }
static int limbs_of(const gts_real &a, double *w) { w[0]=a.x; w[1]=a.y; w[2]=a.z; return 3; }
static int limbs_of(const gqs_real &a, double *w) { w[0]=a.x; w[1]=a.y; w[2]=a.z; w[3]=a.w; return 4; }

/* exact conversion: unevaluated sum of limbs -> MPFR */
template <class T> static void to_mpfr(mpfr_t r, const T &x) {
	double w[4];
	int n = limbs_of(x, w);
	mpfr_set_zero(r, 1);
	for (int i = 0; i < n; i++)
		mpfr_add_d(r, r, w[i], MPFR_RNDN);
}

template <class T> static double lead(const T &a) { double w[4]; limbs_of(a, w); return w[0]; }

/*==================================================================
 * Random full-precision inputs
 *==================================================================*/

static double urand01(void) { return (double)rand() / RAND_MAX; }

static double base_input(int op) {
	double lo = op_range[op].lo, hi = op_range[op].hi;
	if (op_range[op].logscale)
		return lo * pow(hi / lo, urand01());
	return lo + (hi - lo) * urand01();
}

static double second_input(int op) {
	const int k = op - OP_FIRST_BINARY;
	return op_range2[k].lo + (op_range2[k].hi - op_range2[k].lo) * urand01();
}

/* populate lower limbs with noise so the input uses full precision */
static void fill1(gdd_real &v, double a) {
	v = make_dd(a, a * ldexp(urand01() - 0.5, -53)); }
static void fill1(gtd_real &v, double a) {
	v = make_td(a, a * ldexp(urand01() - 0.5, -53),
	               a * ldexp(urand01() - 0.5, -106)); }
static void fill1(gqd_real &v, double a) {
	v = make_qd(a, a * ldexp(urand01() - 0.5, -53),
	               a * ldexp(urand01() - 0.5, -106),
	               a * ldexp(urand01() - 0.5, -159)); }
static void fill1(gds_real &v, double a) { float h = (float)a;
	v = make_ds(h, (float)((a - h) * 0.5)); }
static void fill1(gts_real &v, double a) { float h = (float)a;
	v = make_ts(h, (float)((a - h) * 0.5),
	               (float)(h * ldexp(urand01() - 0.5, -48))); }
static void fill1(gqs_real &v, double a) { float h = (float)a;
	v = make_qs(h, (float)((a - h) * 0.5),
	               (float)(h * ldexp(urand01() - 0.5, -48)),
	               (float)(h * ldexp(urand01() - 0.5, -72))); }

template <class T>
static void fill(T *v, int n, int op) {
	for (int i = 0; i < n; i++) fill1(v[i], base_input(op));
}

/* second argument of the binary ops (unused, but filled, otherwise) */
template <class T>
static void fill2(T *v, int n, int op) {
	for (int i = 0; i < n; i++)
		fill1(v[i], op >= OP_FIRST_BINARY ? second_input(op) : 1.0);
}

/* leading-limb-only values (polyeval coefficients / inputs) */
static void set_from_double(gdd_real &v, double a) { v = make_dd(a, 0.0); }
static void set_from_double(gtd_real &v, double a) { v = make_td(a, 0.0, 0.0); }
static void set_from_double(gqd_real &v, double a) { v = make_qd(a, 0.0, 0.0, 0.0); }
static void set_from_double(gds_real &v, double a) { v = make_ds((float)a, 0.0f); }
static void set_from_double(gts_real &v, double a) { v = make_ts((float)a, 0.0f, 0.0f); }
static void set_from_double(gqs_real &v, double a) { v = make_qs((float)a, 0.0f, 0.0f, 0.0f); }

/*==================================================================
 * [1] Exception / special-value handling
 *==================================================================*/

static void report_special(const char *type, const char *what,
                           double got, int kind /*0:=val 1:+inf 2:-inf 3:nan*/,
                           double val) {
	int ok = 0;
	const char *expect = "";
	char buf[32];
	switch (kind) {
	case 0: ok = (got == val); sprintf(buf, "%g", val); expect = buf; break;
	case 1: ok = (isinf(got) && got > 0); expect = "+Inf"; break;
	case 2: ok = (isinf(got) && got < 0); expect = "-Inf"; break;
	case 3: ok = (isnan(got) != 0);       expect = "NaN";  break;
	}
	printf("  %-8s %-12s expect=%-6s got=%-12.6g %s\n",
	       type, what, expect, got, ok ? "PASS" : "FAIL");
	if (!ok) n_fail++;
}

template <class T>
static void special_tests(void) {
	/* inputs: exp(big), exp(-big), exp(0), log(0), log(-1), sqrt(0) */
	const double big = 1e5;
	T hin[6], hout[6];
	set_from_double(hin[0], big);
	set_from_double(hin[1], -big);
	set_from_double(hin[2], 0.0);
	set_from_double(hin[3], 0.0);
	set_from_double(hin[4], -1.0);
	set_from_double(hin[5], 0.0);

	T *d_in, *d_out;
	GPUMALLOC((void **)&d_in, 6 * sizeof(T));
	GPUMALLOC((void **)&d_out, 6 * sizeof(T));
	TOGPU(d_in, hin, 6 * sizeof(T));

	k_elem<T>(1, 32, d_in,     d_in,     d_out,     3, OP_EXP);
	k_elem<T>(1, 32, d_in + 3, d_in + 3, d_out + 3, 2, OP_LOG);
	k_elem<T>(1, 32, d_in + 5, d_in + 5, d_out + 5, 1, OP_SQRT);
	cudaDeviceSynchronize(); cutilCheckMsg("special");
	FROMGPU(hout, d_out, 6 * sizeof(T));

	const char *tn = RT<T>::name;
	report_special(tn, "exp(+1e5)", lead(hout[0]), 1, 0.0);
	report_special(tn, "exp(-1e5)", lead(hout[1]), 0, 0.0);
	report_special(tn, "exp(0)",    lead(hout[2]), 0, 1.0);
	report_special(tn, "log(0)",    lead(hout[3]), 2, 0.0);
	report_special(tn, "log(-1)",   lead(hout[4]), 3, 0.0);
	report_special(tn, "sqrt(0)",   lead(hout[5]), 0, 0.0);

	GPUFREE(d_in); GPUFREE(d_out);
}

/*==================================================================
 * Timing helper
 *==================================================================*/

template <class F>
static float time_best_ms(F launch) {
	cudaEvent_t t0, t1;
	cudaEventCreate(&t0); cudaEventCreate(&t1);
	float best = 1e30f;
	for (int r = 0; r < REPS; r++) {
		cudaEventRecord(t0);
		launch();
		cudaEventRecord(t1);
		cudaEventSynchronize(t1);
		float ms; cudaEventElapsedTime(&ms, t0, t1);
		if (ms < best) best = ms;
	}
	cudaEventDestroy(t0); cudaEventDestroy(t1);
	return best;
}

/*==================================================================
 * [2]+[3] Accuracy (vs MPFR) and performance, per type
 *==================================================================*/

template <class T>
static void test_type(void) {
	const int pb = prec_bits<T>();
	const int wp = 2 * pb + 64;             /* MPFR working precision */
	const double bits_cap = pb + 24.0;
	const size_t sz_time = N_TIME * sizeof(T);

	T *ha = (T *)malloc(sz_time), *hb = (T *)malloc(sz_time), *hz = (T *)malloc(sz_time);
	T *da, *db, *dz;
	GPUMALLOC((void **)&da, sz_time);
	GPUMALLOC((void **)&db, sz_time);
	GPUMALLOC((void **)&dz, sz_time);

	mpfr_t mx, my, mref, mcomp, mtmp;
	mpfr_inits2(wp, mx, my, mref, mcomp, mtmp, (mpfr_ptr) 0);

	printf("--- %s (theoretical %d bits, N=%d) ---\n", RT<T>::name, pb, n_acc);
	printf("  %-8s %12s %12s %10s %13s %5s %10s\n",
	       "func", "max_err[ulp]", "avg_err[ulp]", "min_bits",
	       "worst_x", "anom", "ns/call");

	for (int op = 0; op < OP_N; op++) {
		/* ---- accuracy vs MPFR ---- */
		fill(ha, n_acc, op);
		fill2(hb, n_acc, op);
		TOGPU(da, ha, n_acc * sizeof(T));
		TOGPU(db, hb, n_acc * sizeof(T));
		int grid = (n_acc + BLOCK - 1) / BLOCK;
		k_elem<T>(grid, BLOCK, da, db, dz, n_acc, op);
		cudaDeviceSynchronize(); cutilCheckMsg("k_elem");
		FROMGPU(hz, dz, n_acc * sizeof(T));

		double max_ulp = 0.0, avg_ulp = 0.0, min_bits = bits_cap, worst_x = 0.0;
		int anom = 0;
		for (int i = 0; i < n_acc; i++) {
			double wl = lead(hz[i]);
			if (isnan(wl) || isinf(wl)) { anom++; continue; }

			to_mpfr(mx, ha[i]);
			if (op == OP_ATAN2) {
				to_mpfr(my, hb[i]);
				mpfr_atan2(mref, mx, my, MPFR_RNDN);
			} else if (op == OP_POW) {
				to_mpfr(my, hb[i]);
				mpfr_pow(mref, mx, my, MPFR_RNDN);
			} else
				op_mpfr[op](mref, mx, MPFR_RNDN);
			to_mpfr(mcomp, hz[i]);
			mpfr_sub(mtmp, mcomp, mref, MPFR_RNDN);
			double relerr;
			if (mpfr_zero_p(mref)) {
				relerr = fabs(mpfr_get_d(mtmp, MPFR_RNDN));
			} else {
				mpfr_div(mtmp, mtmp, mref, MPFR_RNDN);
				relerr = fabs(mpfr_get_d(mtmp, MPFR_RNDN));
			}
			double bits = (relerr > 0.0) ? -log2(relerr) : bits_cap;
			if (bits > bits_cap) bits = bits_cap;
			double ulp = relerr * ldexp(1.0, pb);
			if (ulp > max_ulp) { max_ulp = ulp; worst_x = lead(ha[i]); }
			if (bits < min_bits) min_bits = bits;
			avg_ulp += ulp;
		}
		if (n_acc - anom > 0) avg_ulp /= (n_acc - anom);

		/* ---- timing ---- */
		fill(ha, N_TIME, op);
		fill2(hb, N_TIME, op);
		TOGPU(da, ha, sz_time);
		TOGPU(db, hb, sz_time);
		int tgrid = (N_TIME + BLOCK - 1) / BLOCK;
		float ms = time_best_ms([&]() {
			k_elem<T>(tgrid, BLOCK, da, db, dz, N_TIME, op); });

		printf("  %-8s %12.3g %12.3g %10.1f %13.6g %5d %10.2f\n",
		       op_name[op], max_ulp, avg_ulp, min_bits, worst_x, anom,
		       ms * 1e6 / N_TIME);
	}

	/* ---- polyeval (degree NC-1, coefficients and x in [-1,1]) ----
	   The error is measured against the Horner recurrence evaluated in
	   MPFR, relative to the term-magnitude sum (the quantity the fused
	   multiply-add bounds), so a near-root x does not inflate it. */
	{
		T hc[NC];
		T *dc;
		GPUMALLOC((void **)&dc, NC * sizeof(T));
		for (int k = 0; k < NC; k++)
			set_from_double(hc[k], 2.0 * urand01() - 1.0);
		for (int i = 0; i < n_acc; i++)
			set_from_double(ha[i], 2.0 * urand01() - 1.0);
		TOGPU(dc, hc, NC * sizeof(T));
		TOGPU(da, ha, n_acc * sizeof(T));
		int grid = (n_acc + BLOCK - 1) / BLOCK;
		k_poly<T><<<grid, BLOCK>>>(dc, da, dz, n_acc);
		cudaDeviceSynchronize(); cutilCheckMsg("k_poly");
		FROMGPU(hz, dz, n_acc * sizeof(T));

		double max_ulp = 0.0, avg_ulp = 0.0, min_bits = bits_cap, worst_x = 0.0;
		for (int i = 0; i < n_acc; i++) {
			to_mpfr(mref, hc[NC-1]);
			to_mpfr(mx, ha[i]);
			double scale = fabs(lead(hc[NC-1]));
			double ax = fabs(lead(ha[i]));
			for (int k = NC - 2; k >= 0; k--) {
				mpfr_mul(mref, mref, mx, MPFR_RNDN);
				to_mpfr(mtmp, hc[k]);
				mpfr_add(mref, mref, mtmp, MPFR_RNDN);
				scale = scale * ax + fabs(lead(hc[k]));
			}
			to_mpfr(mcomp, hz[i]);
			mpfr_sub(mtmp, mcomp, mref, MPFR_RNDN);
			double relerr = fabs(mpfr_get_d(mtmp, MPFR_RNDN)) / scale;
			double bits = (relerr > 0.0) ? -log2(relerr) : bits_cap;
			if (bits > bits_cap) bits = bits_cap;
			double ulp = relerr * ldexp(1.0, pb);
			if (ulp > max_ulp) { max_ulp = ulp; worst_x = lead(ha[i]); }
			if (bits < min_bits) min_bits = bits;
			avg_ulp += ulp;
		}
		avg_ulp /= n_acc;

		for (int i = 0; i < N_TIME; i++)
			set_from_double(ha[i], 2.0 * urand01() - 1.0);
		TOGPU(da, ha, sz_time);
		int tgrid = (N_TIME + BLOCK - 1) / BLOCK;
		float ms = time_best_ms([&]() {
			k_poly<T><<<tgrid, BLOCK>>>(dc, da, dz, N_TIME); });

		printf("  %-8s %12.3g %12.3g %10.1f %13.6g %5d %10.2f\n",
		       "poly15", max_ulp, avg_ulp, min_bits, worst_x, 0,
		       ms * 1e6 / N_TIME);
		GPUFREE(dc);
	}
	printf("\n");

	mpfr_clears(mx, my, mref, mcomp, mtmp, (mpfr_ptr) 0);
	GPUFREE(da); GPUFREE(db); GPUFREE(dz);
	free(ha); free(hb); free(hz);
}

/*==================================================================
 * main
 *==================================================================*/

int main(int argc, char **argv) {
	if (argc > 1) n_acc = atoi(argv[1]);
	if (n_acc < 1) n_acc = 200;

	GDDStart(0); GTDStart(0); GQDStart(0);
	GDSStart(0); GTSStart(0); GQSStart(0);

	srand(20260811);

	printf("\ngdtq -- elementary function benchmark (GPU)\n");
	printf("accuracy: N=%d samples per function vs MPFR references;\n", n_acc);
	printf("err[ulp] = relative error scaled by 2^prec_bits;\n");
	printf("timing: %d elements, best of %d runs, kernel time / element.\n\n",
	       N_TIME, REPS);

	printf("[1] Exception / special-value handling\n");
	special_tests<gdd_real>();
	special_tests<gtd_real>();
	special_tests<gqd_real>();
	special_tests<gds_real>();
	special_tests<gts_real>();
	special_tests<gqs_real>();
	printf("\n");

	printf("[2]+[3] Accuracy and performance\n\n");

	/* native double baseline (timing only) */
	{
		double *ha = (double *)malloc(N_TIME * sizeof(double));
		double *hb = (double *)malloc(N_TIME * sizeof(double));
		double *da, *db, *dz;
		GPUMALLOC((void **)&da, N_TIME * sizeof(double));
		GPUMALLOC((void **)&db, N_TIME * sizeof(double));
		GPUMALLOC((void **)&dz, N_TIME * sizeof(double));
		printf("--- double (baseline, timing only) ---\n");
		printf("  %-8s %10s\n", "func", "ns/call");
		int grid = (N_TIME + BLOCK - 1) / BLOCK;
		for (int op = 0; op < OP_N; op++) {
			for (int i = 0; i < N_TIME; i++) {
				ha[i] = base_input(op);
				hb[i] = op >= OP_FIRST_BINARY ? second_input(op) : 1.0;
			}
			TOGPU(da, ha, N_TIME * sizeof(double));
			TOGPU(db, hb, N_TIME * sizeof(double));
			float ms = time_best_ms([&]() {
				k_elem_double<<<grid, BLOCK>>>(da, db, dz, N_TIME, op); });
			printf("  %-8s %10.2f\n", op_name[op], ms * 1e6 / N_TIME);
		}
		printf("\n");
		GPUFREE(da); GPUFREE(db); GPUFREE(dz); free(ha); free(hb);
	}

	printf("double-based classes\n");
	test_type<gdd_real>();
	test_type<gtd_real>();
	test_type<gqd_real>();

	printf("float-based classes\n");
	test_type<gds_real>();
	test_type<gts_real>();
	test_type<gqs_real>();

	printf("%s -- %d special-value failure(s)\n",
	       n_fail == 0 ? "PASS" : "FAIL", n_fail);
	return n_fail == 0 ? 0 : 1;
}
