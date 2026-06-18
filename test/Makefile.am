## SPDX-License-Identifier: BSD-3-Clause
## Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
##
## Builds the benchmark program for the double-precision layers and the
## standalone test program for the float-precision layers (gds/gts/gqs).
## CUDA kernel translation units are compiled by nvcc via explicit rules
## because automake does not have a built-in .cu language.

AM_CPPFLAGS = \
	-I$(top_srcdir)/inc \
	@CUDA_CPPFLAGS@ \
	@CUDA_SDK_INCLUDE@ \
	@QD_CPPFLAGS@ \
	@TDLIB_CPPFLAGS@

AM_CXXFLAGS = -O3 $(OPENMP_CXXFLAGS)

noinst_PROGRAMS = benchmark sqstest

## ======== benchmark (double layers) =================================
benchmark_SOURCES = \
	benchmark.cpp \
	test_util.cpp \
	test_util.h \
	test_common.h \
	gqdtest.h

benchmark_LDADD = gqdtest_kernel.$(OBJEXT) \
	@QD_LIBS@ @TDLIB_LIBS@ @CUDA_LIBS@ $(OPENMP_CXXFLAGS)

benchmark_LDFLAGS = \
	$(OPENMP_CXXFLAGS) \
	@CUDA_SDK_LDFLAGS@ \
	@CUDA_LDFLAGS@ \
	@QD_LDFLAGS@ \
	@TDLIB_LDFLAGS@

benchmark_DEPENDENCIES = gqdtest_kernel.$(OBJEXT)

## ======== sqstest (float layers, standalone) ========================
## sqstest_kernel.cu contains its own main(), so we link it directly
## (no separate .cpp host driver).  We list a placeholder C++ source
## via nodist_EXTRA_sqstest_SOURCES so that automake selects the C++
## linker (g++) rather than the C linker (gcc); the kernel object
## references C++ ABI symbols (__cxa_guard_acquire, etc.) that live
## in libstdc++.
sqstest_SOURCES =
nodist_EXTRA_sqstest_SOURCES = dummy.cxx
sqstest_LDADD   = sqstest_kernel.$(OBJEXT) @CUDA_LIBS@
sqstest_LDFLAGS = @CUDA_SDK_LDFLAGS@ @CUDA_LDFLAGS@
sqstest_DEPENDENCIES = sqstest_kernel.$(OBJEXT)

## ======== nvcc rules ================================================

NVCC_INCLUDES = \
	-I$(top_srcdir)/inc \
	@CUDA_SDK_INCLUDE@

NVCC_DEFS = -D__NV_NO_VECTOR_DEPRECATION_DIAG

KERNEL_DEPS = \
	$(top_srcdir)/inc/gqd.cu        $(top_srcdir)/inc/gqd.cuh \
	$(top_srcdir)/inc/common.cu     $(top_srcdir)/inc/common.cuh \
	$(top_srcdir)/inc/cuda_header.cu \
	$(top_srcdir)/inc/inline.cu     $(top_srcdir)/inc/inline.cuh \
	$(top_srcdir)/inc/gdd_basic.cu  $(top_srcdir)/inc/gdd_basic.cuh \
	$(top_srcdir)/inc/gdd_sqrt.cu   $(top_srcdir)/inc/gdd_sqrt.cuh \
	$(top_srcdir)/inc/gdd_exp.cu    $(top_srcdir)/inc/gdd_exp.cuh \
	$(top_srcdir)/inc/gdd_log.cu    $(top_srcdir)/inc/gdd_log.cuh \
	$(top_srcdir)/inc/gdd_sincos.cu $(top_srcdir)/inc/gdd_sincos.cuh \
	$(top_srcdir)/inc/gtd_basic.cu  $(top_srcdir)/inc/gtd_basic.cuh \
	$(top_srcdir)/inc/gtd_sqrt.cu   $(top_srcdir)/inc/gtd_sqrt.cuh \
	$(top_srcdir)/inc/gtd_exp.cu    $(top_srcdir)/inc/gtd_exp.cuh \
	$(top_srcdir)/inc/gtd_log.cu    $(top_srcdir)/inc/gtd_log.cuh \
	$(top_srcdir)/inc/gtd_sincos.cu $(top_srcdir)/inc/gtd_sincos.cuh \
	$(top_srcdir)/inc/gqd_basic.cu  $(top_srcdir)/inc/gqd_basic.cuh \
	$(top_srcdir)/inc/gqd_sqrt.cu   $(top_srcdir)/inc/gqd_sqrt.cuh \
	$(top_srcdir)/inc/gqd_exp.cu    $(top_srcdir)/inc/gqd_exp.cuh \
	$(top_srcdir)/inc/gqd_log.cu    $(top_srcdir)/inc/gqd_log.cuh \
	$(top_srcdir)/inc/gqd_sincos.cu $(top_srcdir)/inc/gqd_sincos.cuh \
	$(top_srcdir)/inc/gqs.cu        $(top_srcdir)/inc/gqs.cuh \
	$(top_srcdir)/inc/common_s.cu   $(top_srcdir)/inc/common_s.cuh \
	$(top_srcdir)/inc/inline_s.cu   $(top_srcdir)/inc/inline_s.cuh \
	$(top_srcdir)/inc/gds_basic.cu  $(top_srcdir)/inc/gds_basic.cuh \
	$(top_srcdir)/inc/gds_sqrt.cu   $(top_srcdir)/inc/gds_sqrt.cuh \
	$(top_srcdir)/inc/gds_exp.cu    $(top_srcdir)/inc/gds_exp.cuh \
	$(top_srcdir)/inc/gds_log.cu    $(top_srcdir)/inc/gds_log.cuh \
	$(top_srcdir)/inc/gds_sincos.cu $(top_srcdir)/inc/gds_sincos.cuh \
	$(top_srcdir)/inc/gts_basic.cu  $(top_srcdir)/inc/gts_basic.cuh \
	$(top_srcdir)/inc/gts_sqrt.cu   $(top_srcdir)/inc/gts_sqrt.cuh \
	$(top_srcdir)/inc/gts_exp.cu    $(top_srcdir)/inc/gts_exp.cuh \
	$(top_srcdir)/inc/gts_log.cu    $(top_srcdir)/inc/gts_log.cuh \
	$(top_srcdir)/inc/gts_sincos.cu $(top_srcdir)/inc/gts_sincos.cuh \
	$(top_srcdir)/inc/gqs_basic.cu  $(top_srcdir)/inc/gqs_basic.cuh \
	$(top_srcdir)/inc/gqs_sqrt.cu   $(top_srcdir)/inc/gqs_sqrt.cuh \
	$(top_srcdir)/inc/gqs_exp.cu    $(top_srcdir)/inc/gqs_exp.cuh \
	$(top_srcdir)/inc/gqs_log.cu    $(top_srcdir)/inc/gqs_log.cuh \
	$(top_srcdir)/inc/gqs_sincos.cu $(top_srcdir)/inc/gqs_sincos.cuh \
	$(top_srcdir)/inc/gqd_type.h

KERNEL_S_DEPS = \
	$(top_srcdir)/inc/gqs.cu        $(top_srcdir)/inc/gqs.cuh \
	$(top_srcdir)/inc/common_s.cu   $(top_srcdir)/inc/common_s.cuh \
	$(top_srcdir)/inc/cuda_header.cu \
	$(top_srcdir)/inc/inline_s.cu   $(top_srcdir)/inc/inline_s.cuh \
	$(top_srcdir)/inc/gds_basic.cu  $(top_srcdir)/inc/gds_basic.cuh \
	$(top_srcdir)/inc/gds_sqrt.cu   $(top_srcdir)/inc/gds_sqrt.cuh \
	$(top_srcdir)/inc/gds_exp.cu    $(top_srcdir)/inc/gds_exp.cuh \
	$(top_srcdir)/inc/gds_log.cu    $(top_srcdir)/inc/gds_log.cuh \
	$(top_srcdir)/inc/gds_sincos.cu $(top_srcdir)/inc/gds_sincos.cuh \
	$(top_srcdir)/inc/gts_basic.cu  $(top_srcdir)/inc/gts_basic.cuh \
	$(top_srcdir)/inc/gts_sqrt.cu   $(top_srcdir)/inc/gts_sqrt.cuh \
	$(top_srcdir)/inc/gts_exp.cu    $(top_srcdir)/inc/gts_exp.cuh \
	$(top_srcdir)/inc/gts_log.cu    $(top_srcdir)/inc/gts_log.cuh \
	$(top_srcdir)/inc/gts_sincos.cu $(top_srcdir)/inc/gts_sincos.cuh \
	$(top_srcdir)/inc/gqs_basic.cu  $(top_srcdir)/inc/gqs_basic.cuh \
	$(top_srcdir)/inc/gqs_sqrt.cu   $(top_srcdir)/inc/gqs_sqrt.cuh \
	$(top_srcdir)/inc/gqs_exp.cu    $(top_srcdir)/inc/gqs_exp.cuh \
	$(top_srcdir)/inc/gqs_log.cu    $(top_srcdir)/inc/gqs_log.cuh \
	$(top_srcdir)/inc/gqs_sincos.cu $(top_srcdir)/inc/gqs_sincos.cuh \
	$(top_srcdir)/inc/gqd_type.h

gqdtest_kernel.$(OBJEXT): gqdtest_kernel.cu $(KERNEL_DEPS)
	@NVCC@ @NVCC_FLAGS@ -arch=@CUDA_ARCH@ \
	    $(NVCC_DEFS) $(NVCC_INCLUDES) \
	    -c $(srcdir)/gqdtest_kernel.cu -o $@

sqstest_kernel.$(OBJEXT): sqstest_kernel.cu $(KERNEL_S_DEPS)
	@NVCC@ @NVCC_FLAGS@ -arch=@CUDA_ARCH@ \
	    $(NVCC_DEFS) $(NVCC_INCLUDES) \
	    -c $(srcdir)/sqstest_kernel.cu -o $@

CLEANFILES = gqdtest_kernel.$(OBJEXT) sqstest_kernel.$(OBJEXT)

EXTRA_DIST = \
	gqdtest_kernel.cu \
	sqstest_kernel.cu \
	clean.sh \
	make.sh \
	README_CUDA13_FIX.md
