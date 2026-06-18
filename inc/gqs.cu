/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
 *
 * Float-precision implementation umbrella.  Mirrors gqd.cu but pulls in
 * the float-based gds_real / gts_real / gqs_real translation units.
 *
 * Include this file from a single .cu translation unit alongside (or
 * instead of) gqd.cu when the application wants the float layers.
 */
#ifndef __GQS_CU__
#define __GQS_CU__

#include "inline_s.cu"
#include "common_s.cu"

/* GDS (single-double) */
#include "gds_basic.cu"
#include "gds_sqrt.cu"
#include "gds_exp.cu"
#include "gds_log.cu"
#include "gds_sincos.cu"

/* GTS (triple-single) -- shared helpers in inline_s.cu */
#include "gts_basic.cu"
#include "gts_sqrt.cu"
#include "gts_exp.cu"
#include "gts_log.cu"
#include "gts_sincos.cu"

/* GQS (quadruple-single) */
#include "gqs_basic.cu"
#include "gqs_sqrt.cu"
#include "gqs_exp.cu"
#include "gqs_log.cu"
#include "gqs_sincos.cu"

#endif /* __GQS_CU__ */
