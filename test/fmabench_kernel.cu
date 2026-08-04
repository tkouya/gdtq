/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
 *
 * GPU benchmark for the branch-free multi-word fused multiply-add added
 * in gdtq-0.0.3 (dw_fma / tw_fma / qw_fma) and for the division and
 * square root built on top of it.
 *
 *   "old"  the 0.0.2 code path: a separate multiply followed by an
 *          addition, sloppy_div / standard_div, sqrt_legacy.
 *   "new"  the fma-based code path that 0.0.3 uses by default.
 *
 * Each thread runs CHAIN dependent operations so the kernel is compute
 * bound rather than memory bound, and the dependency is carried by the
 * operation under test itself, so no helper arithmetic pollutes the
 * measurement:
 *
 *     multiply-and-add   s = a*b + s        /  s = fma(a, b, s)
 *     division           s = a / s
 *     square root        s = sqrt(s)
 *
 * Kernels are timed with CUDA events and the fastest of REPEAT runs is
 * reported.
 *
 * The file is self-contained: it defines its own main().
 *
 *     nvcc -arch=sm_XX -I../inc fmabench_kernel.cu -o fmabench
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

#define N_ELEMS (1 << 20)
#define BLOCK   128
#define CHAIN   32
#define REPEAT  5

static int n_faster = 0, n_slower = 0;

/*==================================================================
 * The 0.0.2 division, under one name for every type
 *==================================================================*/
__device__ gdd_real old_div(const gdd_real &a, const gdd_real &b) { return sloppy_div(a, b); }
__device__ gtd_real old_div(const gtd_real &a, const gtd_real &b) { return standard_div(a, b); }
__device__ gqd_real old_div(const gqd_real &a, const gqd_real &b) { return sloppy_div(a, b); }
__device__ gds_real old_div(const gds_real &a, const gds_real &b) { return sloppy_div(a, b); }
__device__ gts_real old_div(const gts_real &a, const gts_real &b) { return standard_div(a, b); }
__device__ gqs_real old_div(const gqs_real &a, const gqs_real &b) { return sloppy_div(a, b); }

/*==================================================================
 * Kernels
 *==================================================================*/

/* Each thread carries ACC independent chains, the way a dot product is
   written with several accumulators.  Without that, a multi-word
   multiply-and-add is limited by the length of its dependency chain
   rather than by its operation count, and a cheaper algorithm cannot
   show up at all. */
#define ACC 4

/* Dot-product inner loop with ACC accumulators.  Every accumulator gets
   its own pair of operands, so the product is never loop invariant and
   the unfused path cannot hoist it out of the loop -- both paths really
   perform CHAIN*ACC multiply-and-adds.  The operand window is small, so
   the (warp-uniform) loads stay in L1 and the kernel measures
   arithmetic, not bandwidth. */
#define WINDOW 1024

template <class T>
__global__ void k_muladd_old(const T *a, const T *b, T *z, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	T s[ACC];
#pragma unroll
	for (int j = 0; j < ACC; j++) s[j] = z[(i + j) & (n - 1)];
	for (int k = 0; k < CHAIN; k++) {
#pragma unroll
		for (int j = 0; j < ACC; j++) {
			int m = (k * ACC + j) & (WINDOW - 1);
			s[j] = a[m] * b[m] + s[j];
		}
	}
	T t = s[0];
#pragma unroll
	for (int j = 1; j < ACC; j++) t = t + s[j];
	z[i] = t;
}

template <class T>
__global__ void k_muladd_new(const T *a, const T *b, T *z, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	T s[ACC];
#pragma unroll
	for (int j = 0; j < ACC; j++) s[j] = z[(i + j) & (n - 1)];
	for (int k = 0; k < CHAIN; k++) {
#pragma unroll
		for (int j = 0; j < ACC; j++) {
			int m = (k * ACC + j) & (WINDOW - 1);
			s[j] = fma(a[m], b[m], s[j]);
		}
	}
	T t = s[0];
#pragma unroll
	for (int j = 1; j < ACC; j++) t = t + s[j];
	z[i] = t;
}

template <class T>
__global__ void k_div_old(const T *a, T *z, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	T x = a[i];
	T s[ACC];
#pragma unroll
	for (int j = 0; j < ACC; j++) s[j] = z[(i + j) & (n - 1)];
	for (int k = 0; k < CHAIN; k++)
#pragma unroll
		for (int j = 0; j < ACC; j++) s[j] = old_div(x, s[j]);
	T t = s[0];
#pragma unroll
	for (int j = 1; j < ACC; j++) t = t + s[j];
	z[i] = t;
}

template <class T>
__global__ void k_div_new(const T *a, T *z, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	T x = a[i];
	T s[ACC];
#pragma unroll
	for (int j = 0; j < ACC; j++) s[j] = z[(i + j) & (n - 1)];
	for (int k = 0; k < CHAIN; k++)
#pragma unroll
		for (int j = 0; j < ACC; j++) s[j] = fma_div(x, s[j]);
	T t = s[0];
#pragma unroll
	for (int j = 1; j < ACC; j++) t = t + s[j];
	z[i] = t;
}

template <class T>
__global__ void k_sqrt_old(const T *a, T *z, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	T s[ACC];
#pragma unroll
	for (int j = 0; j < ACC; j++) s[j] = a[(i + j) & (n - 1)];
	for (int k = 0; k < CHAIN; k++)
#pragma unroll
		for (int j = 0; j < ACC; j++) s[j] = sqrt_legacy(s[j]);
	T t = s[0];
#pragma unroll
	for (int j = 1; j < ACC; j++) t = t + s[j];
	z[i] = t;
}

template <class T>
__global__ void k_sqrt_new(const T *a, T *z, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	T s[ACC];
#pragma unroll
	for (int j = 0; j < ACC; j++) s[j] = a[(i + j) & (n - 1)];
	for (int k = 0; k < CHAIN; k++)
#pragma unroll
		for (int j = 0; j < ACC; j++) s[j] = sqrt(s[j]);
	T t = s[0];
#pragma unroll
	for (int j = 1; j < ACC; j++) t = t + s[j];
	z[i] = t;
}

/*==================================================================
 * Host helpers
 *==================================================================*/

static double urand(void) { return (double)rand() / RAND_MAX; }

static void fill(gdd_real *v, int n) { for (int i=0;i<n;i++) v[i] = make_dd(1.0 + urand(), 0.0); }
static void fill(gtd_real *v, int n) { for (int i=0;i<n;i++) v[i] = make_td(1.0 + urand(), 0.0, 0.0); }
static void fill(gqd_real *v, int n) { for (int i=0;i<n;i++) v[i] = make_qd(1.0 + urand(), 0.0, 0.0, 0.0); }
static void fill(gds_real *v, int n) { for (int i=0;i<n;i++) v[i] = make_ds(1.0f + (float)urand(), 0.0f); }
static void fill(gts_real *v, int n) { for (int i=0;i<n;i++) v[i] = make_ts(1.0f + (float)urand(), 0.0f, 0.0f); }
static void fill(gqs_real *v, int n) { for (int i=0;i<n;i++) v[i] = make_qs(1.0f + (float)urand(), 0.0f, 0.0f, 0.0f); }

static float time_ms(cudaEvent_t s, cudaEvent_t e) {
	float ms = 0.0f;
	cudaEventElapsedTime(&ms, s, e);
	return ms;
}

static void row(const char *label, double t_old, double t_new) {
	printf("  %-28s %10.3f %10.3f   x%5.2f%s\n", label, t_old, t_new,
	       t_old / t_new, (t_new < t_old) ? "" : "   <-- no gain");
	fflush(stdout);
	if (t_new < t_old) n_faster++; else n_slower++;
}

static void header(const char *title) {
	printf("\n%s\n", title);
	printf("  %-28s %10s %10s   %7s\n", "kernel", "old ms", "new ms", "speedup");
}

/*==================================================================
 * Benchmark driver
 *==================================================================*/

#define TIME_KERNEL(BEST, LAUNCH)                                          \
	do {                                                               \
		BEST = 1e30;                                               \
		LAUNCH;  cudaDeviceSynchronize();       /* warm up */      \
		for (int r = 0; r < REPEAT; r++) {                         \
			cudaEventRecord(ev0, 0);                           \
			LAUNCH;                                            \
			cudaEventRecord(ev1, 0);                           \
			cudaEventSynchronize(ev1);                         \
			double t = time_ms(ev0, ev1);                      \
			if (t < BEST) BEST = t;                            \
		}                                                          \
	} while (0)

template <class T>
static void bench_type(const char *name,
                       double *r_ma, double *r_div, double *r_sqrt) {
	const int n = N_ELEMS;
	const size_t sz = n * sizeof(T);
	const int grid = (n + BLOCK - 1) / BLOCK;

	T *ha = (T *)malloc(sz), *hb = (T *)malloc(sz), *hz = (T *)malloc(sz);
	T *da, *db, *dz;
	GPUMALLOC((void **)&da, sz); GPUMALLOC((void **)&db, sz);
	GPUMALLOC((void **)&dz, sz);

	fill(ha, n); fill(hb, n); fill(hz, n);
	TOGPU(da, ha, sz); TOGPU(db, hb, sz);

	cudaEvent_t ev0, ev1;
	cudaEventCreate(&ev0); cudaEventCreate(&ev1);

	double t_old, t_new;
	char label[64];

	TOGPU(dz, hz, sz);
	TIME_KERNEL(t_old, (k_muladd_old<T><<<grid, BLOCK>>>(da, db, dz, n)));
	TOGPU(dz, hz, sz);
	TIME_KERNEL(t_new, (k_muladd_new<T><<<grid, BLOCK>>>(da, db, dz, n)));
	cutilCheckMsg("muladd");
	*r_ma = t_old / t_new;
	sprintf(label, "%s  a*b+c", name);
	row(label, t_old, t_new);

	TOGPU(dz, hz, sz);
	TIME_KERNEL(t_old, (k_div_old<T><<<grid, BLOCK>>>(da, dz, n)));
	TOGPU(dz, hz, sz);
	TIME_KERNEL(t_new, (k_div_new<T><<<grid, BLOCK>>>(da, dz, n)));
	cutilCheckMsg("div");
	*r_div = t_old / t_new;
	sprintf(label, "%s  a/b", name);
	row(label, t_old, t_new);

	TIME_KERNEL(t_old, (k_sqrt_old<T><<<grid, BLOCK>>>(da, dz, n)));
	TIME_KERNEL(t_new, (k_sqrt_new<T><<<grid, BLOCK>>>(da, dz, n)));
	cutilCheckMsg("sqrt");
	*r_sqrt = t_old / t_new;
	sprintf(label, "%s  sqrt(a)", name);
	row(label, t_old, t_new);

	cudaEventDestroy(ev0); cudaEventDestroy(ev1);
	GPUFREE(da); GPUFREE(db); GPUFREE(dz);
	free(ha); free(hb); free(hz);
}

int main(void) {
	GDDStart(0);
	GTDStart(0);
	GQDStart(0);
	GDSStart(0);
	GTSStart(0);
	GQSStart(0);

	srand(20260804);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	printf("\ngdtq -- branch-free multi-word FMA benchmark\n");
	printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
	printf("%d elements, %d dependent operations per thread, best of %d runs\n",
	       N_ELEMS, CHAIN, REPEAT);

	double ma[6], dv[6], sq[6];
	const char *names[6] = { "gdd_real (dw_fma)", "gtd_real (tw_fma)",
	                         "gqd_real (qw_fma)", "gds_real (dw_fma)",
	                         "gts_real (tw_fma)", "gqs_real (qw_fma)" };

	header("per type: multiply-and-add, then division, then square root");
	bench_type<gdd_real>(names[0], &ma[0], &dv[0], &sq[0]);
	bench_type<gtd_real>(names[1], &ma[1], &dv[1], &sq[1]);
	bench_type<gqd_real>(names[2], &ma[2], &dv[2], &sq[2]);
	bench_type<gds_real>(names[3], &ma[3], &dv[3], &sq[3]);
	bench_type<gts_real>(names[4], &ma[4], &dv[4], &sq[4]);
	bench_type<gqs_real>(names[5], &ma[5], &dv[5], &sq[5]);

	printf("\nsummary (speedup of the fma-based path over the 0.0.2 path)\n");
	printf("  %-20s %10s %10s %10s\n", "type", "a*b+c", "a/b", "sqrt(a)");
	for (int i = 0; i < 6; i++)
		printf("  %-20s %9.2fx %9.2fx %9.2fx\n", names[i], ma[i], dv[i], sq[i]);

	printf("\n%d kernel(s) faster, %d kernel(s) not faster.\n",
	       n_faster, n_slower);

	return 0;
}
