/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef __GTS_SQRT_CUH__
#define __GTS_SQRT_CUH__

__device__
gts_real sqrt(const gts_real &a);

/* Reference (0.0.2) implementation, kept for benchmarking. */
__device__
gts_real sqrt_legacy(const gts_real &a);

#endif /* __GTS_SQRT_CUH__ */
