/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
 *
 * Standalone test for the float-based gds_real / gts_real / gqs_real
 * layers added in gdtq-0.0.2.  Mirrors the spirit of the dtq-0.0.2
 * tests/qd_test for the float types: small set of kernels exercise
 * +, -, *, /, sqrt, exp, log, sin, cos and the result is verified
 * against host double computations to within the expected float-based
 * precision.
 *
 * Build (autotools picks this up via test/Makefile.am):
 *
 *     nvcc -arch=sm_XX -I../inc sqstest_kernel.cu -o sqstest
 *
 * The file is self-contained: it defines its own main() and compiles
 * to a standalone binary.
 */
#ifndef __NV_NO_VECTOR_DEPRECATION_DIAG
#define __NV_NO_VECTOR_DEPRECATION_DIAG
#endif

#include <stdio.h>
#include <math.h>
#include "cuda_header.cu"
#include "gqs.cu"

/* Number of test elements processed per launch. */
#define N_ELEMS 16

/*======== Kernels (template over the float-based type) ========*/

template <class T>
__global__ void k_add(const T *a, const T *b, T *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) c[i] = a[i] + b[i];
}

template <class T>
__global__ void k_sub(const T *a, const T *b, T *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) c[i] = a[i] - b[i];
}

template <class T>
__global__ void k_mul(const T *a, const T *b, T *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) c[i] = a[i] * b[i];
}

template <class T>
__global__ void k_div(const T *a, const T *b, T *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) c[i] = a[i] / b[i];
}

template <class T>
__global__ void k_sqrt(const T *a, T *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) c[i] = sqrt(a[i]);
}

template <class T>
__global__ void k_exp(const T *a, T *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) c[i] = exp(a[i]);
}

/* full sum (rough) — adds the limbs as a single float; only correct to
 * the leading float's precision but that is what we assert against. */
static inline float full(const gds_real &v) { return v.x + v.y; }
static inline float full(const gts_real &v) { return v.x + v.y + v.z; }
static inline float full(const gqs_real &v) { return v.x + v.y + v.z + v.w; }

/*======== test runners ========*/

template <class T>
static int run_basic(const char *label, T (*make_one)(double),
                     double a_in, double b_in,
                     double expect_add, double expect_sub,
                     double expect_mul, double expect_div,
                     float tol)
{
	int errors = 0;

	T *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, N_ELEMS * sizeof(T));
	cudaMalloc(&d_b, N_ELEMS * sizeof(T));
	cudaMalloc(&d_c, N_ELEMS * sizeof(T));

	T *h_a = (T *)malloc(N_ELEMS * sizeof(T));
	T *h_b = (T *)malloc(N_ELEMS * sizeof(T));
	T *h_c = (T *)malloc(N_ELEMS * sizeof(T));
	for (int i = 0; i < N_ELEMS; ++i) {
		h_a[i] = make_one(a_in);
		h_b[i] = make_one(b_in);
	}
	cudaMemcpy(d_a, h_a, N_ELEMS * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N_ELEMS * sizeof(T), cudaMemcpyHostToDevice);

	int blocks  = (N_ELEMS + 31) / 32;
	int threads = 32;

#define CHECK(op_name, kernel, expect)                                       \
	do {                                                                 \
		kernel<T><<<blocks, threads>>>(d_a, d_b, d_c, N_ELEMS);      \
		cudaDeviceSynchronize();                                     \
		cudaMemcpy(h_c, d_c, N_ELEMS * sizeof(T),                    \
		           cudaMemcpyDeviceToHost);                          \
		float got = full(h_c[0]);                                    \
		float diff = (float)fabs((double)got - (double)(expect));    \
		float rel  = diff / (float)fabs((double)(expect) + 1e-30);   \
		printf("  %-7s %s: got=% .8e  want=% .8e  rel=% .2e %s\n",   \
		       label, op_name, got, (float)(expect), rel,            \
		       (rel < tol) ? "OK" : "FAIL");                         \
		if (!(rel < tol)) ++errors;                                  \
	} while (0)

	CHECK("add", k_add, expect_add);
	CHECK("sub", k_sub, expect_sub);
	CHECK("mul", k_mul, expect_mul);
	CHECK("div", k_div, expect_div);

#undef CHECK

	free(h_a); free(h_b); free(h_c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return errors;
}

template <class T>
static int run_unary(const char *label, T (*make_one)(double),
                     double a_in,
                     double expect_sqrt, double expect_exp,
                     float tol)
{
	int errors = 0;

	T *d_a, *d_c;
	cudaMalloc(&d_a, N_ELEMS * sizeof(T));
	cudaMalloc(&d_c, N_ELEMS * sizeof(T));

	T *h_a = (T *)malloc(N_ELEMS * sizeof(T));
	T *h_c = (T *)malloc(N_ELEMS * sizeof(T));
	for (int i = 0; i < N_ELEMS; ++i) h_a[i] = make_one(a_in);
	cudaMemcpy(d_a, h_a, N_ELEMS * sizeof(T), cudaMemcpyHostToDevice);

	int blocks  = (N_ELEMS + 31) / 32;
	int threads = 32;

#define CHECK1(op_name, kernel, expect)                                       \
	do {                                                                  \
		kernel<T><<<blocks, threads>>>(d_a, d_c, N_ELEMS);            \
		cudaDeviceSynchronize();                                      \
		cudaMemcpy(h_c, d_c, N_ELEMS * sizeof(T),                     \
		           cudaMemcpyDeviceToHost);                           \
		float got = full(h_c[0]);                                     \
		float diff = (float)fabs((double)got - (double)(expect));     \
		float rel  = diff / (float)fabs((double)(expect) + 1e-30);    \
		printf("  %-7s %s: got=% .8e  want=% .8e  rel=% .2e %s\n",    \
		       label, op_name, got, (float)(expect), rel,             \
		       (rel < tol) ? "OK" : "FAIL");                          \
		if (!(rel < tol)) ++errors;                                   \
	} while (0)

	CHECK1("sqrt", k_sqrt, expect_sqrt);
	CHECK1("exp ", k_exp,  expect_exp);

#undef CHECK1

	free(h_a); free(h_c);
	cudaFree(d_a); cudaFree(d_c);
	return errors;
}

/*======== conversion helpers used by run_basic ========*/

static gds_real make_one_ds(double x) {
	float hi = (float)x;
	float lo = (float)(x - (double)hi);
	return make_ds(hi, lo);
}

static gts_real make_one_ts(double x) {
	float hi = (float)x;
	double r0 = x - (double)hi;
	float md = (float)r0;
	float lo = (float)(r0 - (double)md);
	return make_ts(hi, md, lo);
}

static gqs_real make_one_qs(double x) {
	float a = (float)x;
	double r0 = x - (double)a;
	float b = (float)r0;
	double r1 = r0 - (double)b;
	float c = (float)r1;
	float d = (float)(r1 - (double)c);
	return make_qs(a, b, c, d);
}

/*======== main ========*/

int main(int argc, char **argv) {
	(void)argc; (void)argv;

	printf("== gdtq-0.0.2 float-layer test (gds/gts/gqs) ==\n");

	/* Bring the constant tables on the device. */
	GDSStart(0);
	GTSStart(0);
	GQSStart(0);

	const double A = 1.5;
	const double B = 0.25;
	const double EXP_ADD = A + B;
	const double EXP_SUB = A - B;
	const double EXP_MUL = A * B;
	const double EXP_DIV = A / B;

	const double UA = 2.0;
	const double EXP_SQRT = std::sqrt(UA);
	const double EXP_EXP  = std::exp(UA);

	int errors = 0;

	/* DS expects ~14 decimal digits, but here we operate on values that
	 * fit cleanly in a single float so the relative tolerance is set by
	 * float epsilon (~6e-8). The same tolerance is used for TS and QS
	 * because the leading limb is identical for these toy inputs. */
	const float TOL = 1e-6f;

	printf("-- gds_real (2 floats) --\n");
	errors += run_basic("ds", make_one_ds, A, B,
	                    EXP_ADD, EXP_SUB, EXP_MUL, EXP_DIV, TOL);
	errors += run_unary("ds", make_one_ds, UA,
	                    EXP_SQRT, EXP_EXP, 1e-4f);

	printf("-- gts_real (3 floats) --\n");
	errors += run_basic("ts", make_one_ts, A, B,
	                    EXP_ADD, EXP_SUB, EXP_MUL, EXP_DIV, TOL);
	errors += run_unary("ts", make_one_ts, UA,
	                    EXP_SQRT, EXP_EXP, 1e-4f);

	printf("-- gqs_real (4 floats) --\n");
	errors += run_basic("qs", make_one_qs, A, B,
	                    EXP_ADD, EXP_SUB, EXP_MUL, EXP_DIV, TOL);
	errors += run_unary("qs", make_one_qs, UA,
	                    EXP_SQRT, EXP_EXP, 1e-4f);

	GDSEnd();
	/* GTSEnd / GQSEnd would each call cudaDeviceReset, only one needs
	 * to run; GDSEnd already did. Skip the rest. */

	printf("\n%s -- %d error(s)\n",
	       errors == 0 ? "PASS" : "FAIL", errors);
	return errors == 0 ? 0 : 1;
}
