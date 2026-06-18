/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
 *
 * Float-precision API umbrella header.  Mirrors gqd.cuh but pulls in
 * the gds_real / gts_real / gqs_real declarations.
 */
#ifndef __GQS_CUH__
#define __GQS_CUH__

/* Float-overload Two-Sum / Two-Prod helpers must come before the
 * type-specific headers, since they reference these helpers. */
#include "inline_s.cuh"
#include "common_s.cuh"

/* GDS (single-double, 2 floats) */
#include "gds_basic.cuh"
#include "gds_sqrt.cuh"
#include "gds_exp.cuh"
#include "gds_log.cuh"
#include "gds_sincos.cuh"

/* GTS (triple-single, 3 floats) -- before GQS because GQS shares
 * helpers with the triple-single layer. */
#include "gts_basic.cuh"
#include "gts_sqrt.cuh"
#include "gts_exp.cuh"
#include "gts_log.cuh"
#include "gts_sincos.cuh"

/* GQS (quadruple-single, 4 floats) */
#include "gqs_basic.cuh"
#include "gqs_sqrt.cuh"
#include "gqs_exp.cuh"
#include "gqs_log.cuh"
#include "gqs_sincos.cuh"

#endif /* __GQS_CUH__ */
