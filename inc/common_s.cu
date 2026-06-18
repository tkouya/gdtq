/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
 *
 * Float-precision counterpart to common.cu.  Defines make_ds / make_ts /
 * make_qs and initialises the on-device constant-memory tables for the
 * float-based gds_real / gts_real / gqs_real layers.
 *
 * The constant tables (inv_fact and sin/cos) are derived from the
 * reference double values by truncating each high-precision component
 * down to the float granularity, propagating the residual into the
 * next limb.  Sin/cos tables are computed at GxSStart() time using the
 * host's double-precision sin/cos so we do not have to transcribe the
 * 256-entry tables a second time.
 */
#ifndef __GQS_COMMON_CU__
#define __GQS_COMMON_CU__

#include "common_s.cuh"
#include <cmath>

/* Storage for the constant-memory tables — only emitted in the
 * multi-TU (-rdc=true) build; in the single-TU build the storage is
 * `static` inside common_s.cuh.  See common_s.cuh for the rationale.
 * common_s.cu is included by exactly one TU (gdslinear.cu, via gqs.cu),
 * so this is the sole definition; all other TUs reference the externs
 * declared in common_s.cuh and `nvcc -dlink` resolves them. */
#ifdef __CUDACC_RDC__
__device__ __constant__ gds_real ds_inv_fact[n_ds_inv_fact];
__device__ __constant__ gds_real d_ds_sin_table[4];
__device__ __constant__ gds_real d_ds_cos_table[4];

__device__ __constant__ gts_real ts_inv_fact[n_ts_inv_fact];
__device__ __constant__ gts_real d_ts_sin_table[256];
__device__ __constant__ gts_real d_ts_cos_table[256];

__device__ __constant__ gqs_real qs_inv_fact[15];
__device__ __constant__ gqs_real d_qs_sin_table[256];
__device__ __constant__ gqs_real d_qs_cos_table[256];
#endif

/*======== type construction ========*/
__device__ __host__
gds_real make_ds( const float x, const float y ) {
	return make_float2( x, y );
}

__device__ __host__
gds_real make_ds( const float x ) {
	return make_float2( x, 0.0f );
}

__device__ __host__
gts_real make_ts( const float x, const float y, const float z ) {
	return make_float3( x, y, z );
}

__device__ __host__
gts_real make_ts( const float x ) {
	return make_float3( x, 0.0f, 0.0f );
}

__device__ __host__
gqs_real make_qs( const float x,
                  const float y,
                  const float z,
                  const float w ) {
	return make_float4( x, y, z, w );
}

__device__ __host__
gqs_real make_qs( const float x ) {
	return make_qs( x, 0.0f, 0.0f, 0.0f );
}

/*======== host helpers: convert a high-precision double into floats =====
 * Each helper takes the components of the original gdd/gtd/gqd constant
 * and produces an equivalent gds/gts/gqs.  The trick mirrors ds_from_dd
 * in dtq-0.0.2: round the running double residual down to a float, then
 * subtract that float (in double arithmetic) and continue to the next
 * limb.  This loses only the bits below the float-precision target.
 *
 * The single-double overloads (taking just one double) are used by the
 * sin/cos table generators; only the leading limb is supplied so the
 * staircase is anchored in pure double precision.
 */
static gds_real ds_from_d2(double d0, double d1) {
	float f0 = (float)d0;
	double r = (d0 - (double)f0) + d1;
	float f1 = (float)r;
	return make_ds(f0, f1);
}

static gds_real ds_from_d(double d0) {
	float f0 = (float)d0;
	double r = d0 - (double)f0;
	float f1 = (float)r;
	return make_ds(f0, f1);
}

static gts_real ts_from_d3(double d0, double d1, double d2) {
	float f0 = (float)d0;
	double r0 = (d0 - (double)f0) + d1;
	float f1 = (float)r0;
	double r1 = (r0 - (double)f1) + d2;
	float f2 = (float)r1;
	return make_ts(f0, f1, f2);
}

static gts_real ts_from_d(double d0) {
	float f0 = (float)d0;
	double r0 = d0 - (double)f0;
	float f1 = (float)r0;
	double r1 = r0 - (double)f1;
	float f2 = (float)r1;
	return make_ts(f0, f1, f2);
}

static gqs_real qs_from_d4(double d0, double d1, double d2, double d3) {
	float f0 = (float)d0;
	double r0 = (d0 - (double)f0) + d1;
	float f1 = (float)r0;
	double r1 = (r0 - (double)f1) + d2;
	float f2 = (float)r1;
	double r2 = (r1 - (double)f2) + d3;
	float f3 = (float)r2;
	return make_qs(f0, f1, f2, f3);
}

static gqs_real qs_from_d(double d0) {
	float f0 = (float)d0;
	double r0 = d0 - (double)f0;
	float f1 = (float)r0;
	double r1 = r0 - (double)f1;
	float f2 = (float)r1;
	double r2 = r1 - (double)f2;
	float f3 = (float)r2;
	return make_qs(f0, f1, f2, f3);
}

/*======== sin/cos table generation =======================================
 * The original GQD/GTD tables hold sin(k*pi/1024) / cos(k*pi/1024) at
 * 0-based index i = k-1 for k = 1..256 (the 256 entries cover one
 * eighth of the unit circle, i.e. 0 < angle <= pi/4).  We reproduce the
 * SAME index convention so that the converted gqs/gts_sincos kernels
 * (which use abs_k-1 lookups) work without modification.  Each entry
 * uses the staircase float decomposition (ts_from_d / qs_from_d) so
 * that all 3/4 float limbs carry useful precision rather than only the
 * leading limb.
 */
static const double K_PI = 3.14159265358979323846;

static void fill_sin_cos_d3(gts_real *sin_tab, gts_real *cos_tab) {
	for (int i = 0; i < 256; ++i) {
		double angle = (double)(i + 1) * K_PI / 1024.0;
		sin_tab[i] = ts_from_d(std::sin(angle));
		cos_tab[i] = ts_from_d(std::cos(angle));
	}
}

static void fill_sin_cos_d4(gqs_real *sin_tab, gqs_real *cos_tab) {
	for (int i = 0; i < 256; ++i) {
		double angle = (double)(i + 1) * K_PI / 1024.0;
		sin_tab[i] = qs_from_d(std::sin(angle));
		cos_tab[i] = qs_from_d(std::cos(angle));
	}
}

/*======== inv_fact tables =================================================
 * 1/k! for k = 3..17.  Same numerical source as common.cu, just reduced
 * to the float layer.
 */
static const double s_inv_fact_d[15][4] = {
	{ 1.66666666666666657e-01,  9.25185853854297066e-18,  5.13581318503262866e-34,  2.85094902409834186e-50},
	{ 4.16666666666666644e-02,  2.31296463463574266e-18,  1.28395329625815716e-34,  7.12737256024585466e-51},
	{ 8.33333333333333322e-03,  1.15648231731787138e-19,  1.60494162032269652e-36,  2.22730392507682967e-53},
	{ 1.38888888888888894e-03, -5.30054395437357706e-20, -1.73868675534958776e-36, -1.63335621172300840e-52},
	{ 1.98412698412698413e-04,  1.72095582934207053e-22,  1.49269123913941271e-40,  1.29470326746002471e-58},
	{ 2.48015873015873016e-05,  2.15119478667758816e-23,  1.86586404892426588e-41,  1.61837908432503088e-59},
	{ 2.75573192239858925e-06, -1.85839327404647208e-22,  8.49175460488199287e-39, -5.72661640789429621e-55},
	{ 2.75573192239858883e-07,  2.37677146222502973e-23, -3.26318890334088294e-40,  1.61435111860404415e-56},
	{ 2.50521083854417202e-08, -1.44881407093591197e-24,  2.04267351467144546e-41, -8.49632672007163175e-58},
	{ 2.08767569878681002e-09, -1.20734505911325997e-25,  1.70222792889287100e-42,  1.41609532150396700e-58},
	{ 1.60590438368216133e-10,  1.25852945887520981e-26, -5.31334602762985031e-43,  3.54021472597605528e-59},
	{ 1.14707455977297245e-11,  2.06555127528307454e-28,  6.88907923246664603e-45,  5.72920002655109095e-61},
	{ 7.64716373181981641e-13,  7.03872877733453001e-30, -7.82753927716258345e-48,  1.92138649443790242e-64},
	{ 4.77947733238738525e-14,  4.39920548583408126e-31, -4.89221204822661465e-49,  1.20086655902368901e-65},
	{ 2.81145725434552060e-15,  1.65088427308614326e-31, -2.87777179307447918e-50,  4.27110689256293549e-67}
};

/*======== initialization functions ========*/

void GDSStart(const int device) {
	printf("GDS turns on...\n");
	CUDA_SAFE_CALL( cudaSetDevice(device) );

	gds_real h_inv_fact[n_ds_inv_fact];
	for (int i = 0; i < n_ds_inv_fact; ++i) {
		h_inv_fact[i] = ds_from_d2(s_inv_fact_d[i][0], s_inv_fact_d[i][1]);
	}
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(ds_inv_fact, h_inv_fact, sizeof(gds_real)*n_ds_inv_fact) );

	/* sin/cos for pi/16, pi/8, 3pi/16, pi/4 (DD layer convention) */
	gds_real h_sin_table[4];
	gds_real h_cos_table[4];
	for (int i = 0; i < 4; ++i) {
		double angle = (double)(i + 1) * K_PI / 16.0;
		h_sin_table[i] = ds_from_d(std::sin(angle));
		h_cos_table[i] = ds_from_d(std::cos(angle));
	}
	cutilSafeCall( cudaMemcpyToSymbol(d_ds_sin_table, h_sin_table, sizeof(gds_real)*4) );
	cutilSafeCall( cudaMemcpyToSymbol(d_ds_cos_table, h_cos_table, sizeof(gds_real)*4) );

	printf("\tdone.\n");
}

void GDSEnd() {
	printf("GDS turns off...\n");
	CUDA_SAFE_CALL( cudaDeviceReset() );
	printf("\tdone.\n");
}

void GTSStart(const int device) {
	printf("GTS turns on...\n");
	CUDA_SAFE_CALL( cudaSetDevice(device) );

	gts_real h_inv_fact[n_ts_inv_fact];
	for (int i = 0; i < n_ts_inv_fact; ++i) {
		h_inv_fact[i] = ts_from_d3(s_inv_fact_d[i][0],
		                           s_inv_fact_d[i][1],
		                           s_inv_fact_d[i][2]);
	}
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(ts_inv_fact, h_inv_fact, sizeof(gts_real)*n_ts_inv_fact) );

	gts_real *h_sin_table = (gts_real*)malloc(sizeof(gts_real)*256);
	gts_real *h_cos_table = (gts_real*)malloc(sizeof(gts_real)*256);
	fill_sin_cos_d3(h_sin_table, h_cos_table);
	cutilSafeCall( cudaMemcpyToSymbol(d_ts_sin_table, h_sin_table, sizeof(gts_real)*256) );
	cutilSafeCall( cudaMemcpyToSymbol(d_ts_cos_table, h_cos_table, sizeof(gts_real)*256) );
	free(h_sin_table);
	free(h_cos_table);

	printf("\tdone.\n");
}

void GTSEnd() {
	printf("GTS turns off...\n");
	CUDA_SAFE_CALL( cudaDeviceReset() );
	printf("\tdone.\n");
}

void GQSStart(const int device) {
	printf("GQS turns on...\n");
	CUDA_SAFE_CALL( cudaSetDevice(device) );

	gqs_real *h_inv_fact = (gqs_real*)malloc(sizeof(gqs_real)*n_qs_inv_fact);
	for (int i = 0; i < n_qs_inv_fact; ++i) {
		h_inv_fact[i] = qs_from_d4(s_inv_fact_d[i][0],
		                           s_inv_fact_d[i][1],
		                           s_inv_fact_d[i][2],
		                           s_inv_fact_d[i][3]);
	}
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(qs_inv_fact, h_inv_fact, sizeof(gqs_real)*n_qs_inv_fact) );
	free(h_inv_fact);

	gqs_real *h_sin_table = (gqs_real*)malloc(sizeof(gqs_real)*256);
	gqs_real *h_cos_table = (gqs_real*)malloc(sizeof(gqs_real)*256);
	fill_sin_cos_d4(h_sin_table, h_cos_table);
	cutilSafeCall( cudaMemcpyToSymbol(d_qs_sin_table, h_sin_table, sizeof(gqs_real)*256) );
	cutilSafeCall( cudaMemcpyToSymbol(d_qs_cos_table, h_cos_table, sizeof(gqs_real)*256) );
	free(h_sin_table);
	free(h_cos_table);

	printf("\tdone.\n");
}

void GQSEnd() {
	printf("GQS turns off...\n");
	CUDA_SAFE_CALL( cudaDeviceReset() );
	printf("\tdone.\n");
}

#endif /* __GQS_COMMON_CU__ */
