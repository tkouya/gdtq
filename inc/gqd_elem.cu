/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. */
/*
 * gqd_elem.cu -- elementary functions of gdd_real, gtd_real and gqd_real
 *                (gdtq 0.0.4), by the generic multi-word kernels in mw/
 *                (see gdtq_mw.cuh).
 *
 * Maximum errors against MPFR on the evaluation domains (GB10, u = 2^-53):
 * DD 4.3 u^2, TD 8.3 u^3, QD 11.6 u^4, against up to 10^6-10^7 u^K for
 * 0.0.3 (sin/cos reduction, log near 1); speed-up over 0.0.3 (geometric
 * means) DD 2.7x, TD 4.0x, QD 4.8x.  Compiled with -DMW_CASCADE_GUARD the
 * accumulators keep a guard word: <= 5.6 u^K, but 1.4-2.3x slower.
 *
 * New in 0.0.4: log1p and pow for all three classes; atan2 ... atanh also
 * for gtd_real, and for gqd_real without ALL_MATH.
 *
 * Compiled with -DGDTQ_LEGACY_ELEMENTARY (configure --enable-legacy-
 * elementary), the 0.0.3 implementations in gXd_exp.cu / gXd_log.cu /
 * gXd_sincos.cu are used for the functions they provide (with the 0.0.3
 * meaning of ALL_MATH), and the kernels below only for the others.
 */
#ifndef __GQD_ELEM_CU__
#define __GQD_ELEM_CU__

#include "gqd.cuh"
#include "gdtq_mw.cuh"

/* ---- gdd_real ---- */
#if !defined(GDTQ_LEGACY_ELEMENTARY)
GDTQ_MW_BASIC(gdd_real)
#endif
#if !defined(GDTQ_LEGACY_ELEMENTARY) || defined(ALL_MATH)
GDTQ_MW_ADVANCED(gdd_real)
#endif
GDTQ_MW_NEW(gdd_real)

/* ---- gtd_real ---- */
#if !defined(GDTQ_LEGACY_ELEMENTARY)
GDTQ_MW_BASIC(gtd_real)
#endif
GDTQ_MW_ADVANCED(gtd_real)
GDTQ_MW_NEW(gtd_real)

/* ---- gqd_real ---- */
#if !defined(GDTQ_LEGACY_ELEMENTARY)
GDTQ_MW_BASIC(gqd_real)
#endif
#if !defined(GDTQ_LEGACY_ELEMENTARY) || !defined(ALL_MATH)
GDTQ_MW_ADVANCED(gqd_real)
#endif
GDTQ_MW_NEW(gqd_real)

#endif /* __GQD_ELEM_CU__ */
