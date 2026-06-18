/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GQD_COMMON_CUH__
#define __GQD_COMMON_CUH__

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "gqd_type.h"		//type definitions for gdd_real and gqd_real
#include "cuda_header.cu"
//#include "inline.cu" 		//basic functions used by both gdd_real and gqd_real
//#include "inline.cuh" 		//basic functions used by both gdd_real and gqd_real

/* type definitions, defined in the type.h */
//defined in gqd_type.h

//extern "C" {

/* type construction */
extern __device__ __host__
gdd_real make_dd( const double x, const double y );

extern __device__ __host__
gdd_real make_dd( const double x );

extern __device__ __host__
gtd_real make_td( const double x, const double y, const double z );

extern __device__ __host__
gtd_real make_td( const double x );

extern __device__ __host__
gqd_real make_qd( const double x,
                const double y,
                const double z,
                const double w );

extern __device__ __host__
gqd_real make_qd( const double x );

/* constants */
#define _dd_eps (4.93038065763132e-32)  // 2^-104
#define _dd_e make_dd(2.718281828459045091e+00, 1.445646891729250158e-16)
#define _dd_log2 make_dd(6.931471805599452862e-01, 2.319046813846299558e-17)
#define _dd_2pi make_dd(6.283185307179586232e+00, 2.449293598294706414e-16)
#define _dd_pi make_dd(3.141592653589793116e+00, 1.224646799147353207e-16)
#define _dd_pi2 make_dd(1.570796326794896558e+00, 6.123233995736766036e-17)
#define _dd_pi16 make_dd(1.963495408493620697e-01, 7.654042494670957545e-18)
#define _dd_pi4 make_dd(7.853981633974482790e-01, 3.061616997868383018e-17)
#define _dd_3pi4 make_dd(2.356194490192344837e+00, 9.1848509936051484375e-17)

/* triple-double constants -- top three components of the QD constants below */
#define _td_eps  (8.673617379884035e-48)  /* ~ 2^-156, three-double rounding */
#define _td_e    make_td(2.718281828459045091e+00, 1.445646891729250158e-16, -2.127717108038176765e-33)
#define _td_log2 make_td(6.931471805599452862e-01, 2.319046813846299558e-17,  5.707708438416212066e-34)
#define _td_2pi  make_td(6.283185307179586232e+00, 2.449293598294706414e-16, -5.989539619436679332e-33)
#define _td_pi   make_td(3.141592653589793116e+00, 1.224646799147353207e-16, -2.994769809718339666e-33)
#define _td_pi2  make_td(1.570796326794896558e+00, 6.123233995736766036e-17, -1.497384904859169833e-33)
#define _td_pi4  make_td(7.853981633974482790e-01, 3.061616997868383018e-17, -7.486924524295849165e-34)
#define _td_pi1024 make_td(3.067961575771282340e-03, 1.195944139792337116e-19, -2.924579892303066080e-36)

#define _qd_e make_qd(2.718281828459045091e+00, 1.445646891729250158e-16,  -2.127717108038176765e-33, 1.515630159841218954e-49)
#define _qd_log2 make_qd(6.931471805599452862e-01, 2.319046813846299558e-17,5.707708438416212066e-34,-3.582432210601811423e-50)
#define _qd_eps (1.21543267145725e-63) // = 2^-209
#define _qd_2pi make_qd(6.283185307179586232e+00, 2.449293598294706414e-16, -5.989539619436679332e-33, 2.224908441726730563e-49)
#define _qd_pi make_qd(3.141592653589793116e+00, 1.224646799147353207e-16, -2.994769809718339666e-33, 1.112454220863365282e-49)
#define _qd_pi2 make_qd(1.570796326794896558e+00, 6.123233995736766036e-17, -1.497384904859169833e-33, 5.562271104316826408e-50)
#define _qd_pi1024 make_qd( 3.067961575771282340e-03, 1.195944139792337116e-19,  -2.924579892303066080e-36, 1.086381075061880158e-52)
#define _qd_pi4 make_qd(7.853981633974482790e-01, 3.061616997868383018e-17, -7.486924524295849165e-34, 2.781135552158413204e-50)
#define _qd_3pi4 make_qd(2.356194490192344837e+00, 9.1848509936051484375e-17, 3.9168984647504003225e-33, -2.5867981632704860386e-49)


/* data in the constant memory
 *
 * Two declaration modes depending on how this header is being used:
 *
 *   (A) Single-TU build (gdtq's stand-alone benchmark / sqstest):
 *       compiled WITHOUT -rdc=true.  Use `static __device__ __constant__`
 *       so the storage lives privately in whichever single TU includes
 *       common.cu via gqd.cu.  This is the original gdtq behaviour.
 *
 *   (B) Multi-TU build (e.g. bncmatmul's libbncmm_cuda.a) compiled
 *       WITH -rdc=true.  Use `extern __device__ __constant__` here and
 *       put the actual storage definitions in common.cu (its own
 *       conditional block).  The owner TU is whichever .cu pulls in
 *       common.cu via gqd.cu; all other TUs reference the externs and
 *       the device link step (`nvcc -dlink`) resolves them.  Static
 *       per-TU storage would otherwise produce
 *       ".nv.constant3 section size mismatch" at nvlink time.
 *
 * `__CUDACC_RDC__` is defined automatically by nvcc when -rdc=true is
 * passed, so we use it to discriminate the two modes.
 */
#define n_dd_inv_fact (15)
#define n_td_inv_fact (15)

#ifdef __CUDACC_RDC__
/* (B) -rdc=true: extern only, storage lives in common.cu. */
extern __device__ __constant__ gdd_real dd_inv_fact[n_dd_inv_fact];
extern __device__ __constant__ gdd_real d_dd_sin_table[4];
extern __device__ __constant__ gdd_real d_dd_cos_table[4];

extern __device__ __constant__ gtd_real td_inv_fact[n_td_inv_fact];
extern __device__ __constant__ gtd_real d_td_sin_table[256];
extern __device__ __constant__ gtd_real d_td_cos_table[256];

extern __device__ __constant__ gqd_real inv_fact[15];
extern __device__ __constant__ gqd_real d_sin_table[256];
extern __device__ __constant__ gqd_real d_cos_table[256];
#else
/* (A) Single-TU: per-TU static storage (original gdtq pattern). */
static __device__ __constant__ gdd_real dd_inv_fact[n_dd_inv_fact];
static __device__ __constant__ gdd_real d_dd_sin_table[4];
static __device__ __constant__ gdd_real d_dd_cos_table[4];

static __device__ __constant__ gtd_real td_inv_fact[n_td_inv_fact];
static __device__ __constant__ gtd_real d_td_sin_table[256];
static __device__ __constant__ gtd_real d_td_cos_table[256];

static __device__ __constant__ gqd_real inv_fact[15];
/*
 * Note (CUDA 13 port):
 * The original source wrapped d_sin_table/d_cos_table in
 * `#ifdef USE_GQD_SIN`, but the functions in gqd_sincos.cu reference
 * these tables unconditionally, so without the macro the compilation
 * failed with "identifier d_sin_table/d_cos_table is undefined".
 */
static __device__ __constant__ gqd_real d_sin_table[256];
static __device__ __constant__ gqd_real d_cos_table[256];
#endif

static const int n_inv_fact = 15;

/** initialization function */
extern void GDDStart(const int device);

extern void GDDEnd();

extern void GTDStart(const int device);

extern void GTDEnd();

extern void GQDEnd();

extern void GQDStart(const int device);

//}

#endif /* __GQD_COMMON_CUH__ */
