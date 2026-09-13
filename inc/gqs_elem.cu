/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. */
/*
 * gqs_elem.cu -- elementary functions of gds_real, gts_real and gqs_real
 *                (gdtq 0.0.4), by the generic multi-word kernels in mw/
 *                (see gdtq_mw.cuh) on float words.
 *
 * Maximum errors against MPFR on the evaluation domains (GB10, u = 2^-24):
 * DS 4.4 u^2, TS 7.7 u^3, QS 13.7 u^4; speed-up over 0.0.3 (geometric
 * means) DS 2.5x, TS 10.7x, QS 9.8x.
 * Only the Payne-Hanek reduction of huge sin/cos/tan arguments uses double
 * arithmetic (rare; see gdtq_twopass.cuh for arrays with such inputs).
 *
 * New in 0.0.4: log1p and pow for all three classes; atan2 ... atanh also
 * for gts_real, and for gqs_real without ALL_MATH.
 *
 * Compiled with -DGDTQ_LEGACY_ELEMENTARY (configure --enable-legacy-
 * elementary), the 0.0.3 implementations in gXs_exp.cu / gXs_log.cu /
 * gXs_sincos.cu are used for the functions they provide (with the 0.0.3
 * meaning of ALL_MATH), and the kernels below only for the others.
 */
#ifndef __GQS_ELEM_CU__
#define __GQS_ELEM_CU__

#include "gqs.cuh"
#include "gdtq_mw.cuh"

/* ---- gds_real ---- */
#if !defined(GDTQ_LEGACY_ELEMENTARY)
GDTQ_MW_BASIC(gds_real)
#endif
#if !defined(GDTQ_LEGACY_ELEMENTARY) || defined(ALL_MATH)
GDTQ_MW_ADVANCED(gds_real)
#endif
GDTQ_MW_NEW(gds_real)

/* ---- gts_real ---- */
#if !defined(GDTQ_LEGACY_ELEMENTARY)
GDTQ_MW_BASIC(gts_real)
#endif
GDTQ_MW_ADVANCED(gts_real)
GDTQ_MW_NEW(gts_real)

/* ---- gqs_real ---- */
#if !defined(GDTQ_LEGACY_ELEMENTARY)
GDTQ_MW_BASIC(gqs_real)
#endif
#if !defined(GDTQ_LEGACY_ELEMENTARY) || !defined(ALL_MATH)
GDTQ_MW_ADVANCED(gqs_real)
#endif
GDTQ_MW_NEW(gqs_real)

#endif /* __GQS_ELEM_CU__ */
