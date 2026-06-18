## SPDX-License-Identifier: BSD-3-Clause
## Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).

ACLOCAL_AMFLAGS = -I m4

SUBDIRS = inc docs

if ENABLE_BENCHMARK
SUBDIRS += test
endif

## pkg-config descriptor for the consolidated libgqd library.
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = gqd.pc gqd-shared.pc

EXTRA_DIST = LICENSE README README.md bootstrap m4/ax_cuda.m4 \
	gqd.pc.in gqd-shared.pc.in
