/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

// CUH files for GDD class

#ifndef __GQD_CUH__
#define __GQD_CUH__

// common.cu
#include "inline.cuh" 		//basic functions used by both gdd_real and gqd_real
#include "common.cuh"

// GDD
#include "gdd_basic.cuh"
#include "gdd_sqrt.cuh"
#include "gdd_exp.cuh"
#include "gdd_log.cuh"
#include "gdd_sincos.cuh"

// GTD (must come before GQD because gtd uses make_td/inline helpers
//      shared with the GQD layer)
#include "gtd_basic.cuh"
#include "gtd_sqrt.cuh"
#include "gtd_exp.cuh"
#include "gtd_log.cuh"
#include "gtd_sincos.cuh"

// GQD
#include "gqd_basic.cuh"
#include "gqd_sqrt.cuh"
#include "gqd_exp.cuh"
#include "gqd_log.cuh"
#include "gqd_sincos.cuh"

#endif // __GQD_CUH__
