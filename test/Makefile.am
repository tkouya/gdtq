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

noinst_PROGRAMS = benchmark sqstest fmatest fmabench elembench fmaref

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

## ======== fmatest / fmabench (branch-free FMA, standalone) ==========
## Both .cu files contain their own main(), like sqstest above, and
## exercise the double- and the float-based layers together.
##
##   fmatest   accuracy of dw_fma / tw_fma / qw_fma and of the division
##             and square root built on them, checked on the GPU against
##             an exact expansion-arithmetic reference on the host.
##   fmabench  old (0.0.2) vs new (0.0.3) timings for the same.
fmatest_SOURCES =
nodist_EXTRA_fmatest_SOURCES = dummy.cxx
fmatest_LDADD   = fmatest_kernel.$(OBJEXT) @CUDA_LIBS@
fmatest_LDFLAGS = @CUDA_SDK_LDFLAGS@ @CUDA_LDFLAGS@
fmatest_DEPENDENCIES = fmatest_kernel.$(OBJEXT)

## fmaref (0.0.4): bitwise conformance of dw/tw/qw_fma and the
## div/sqrt-safe variants with a host transcription of the machine-proved
## reference fma_ref.c (DW 17 / TW 72 / QW 176 flops, safe 20 / 84 / 206).
fmaref_SOURCES =
nodist_EXTRA_fmaref_SOURCES = dummy.cxx
fmaref_LDADD   = fmaref_kernel.$(OBJEXT) @CUDA_LIBS@
fmaref_LDFLAGS = @CUDA_SDK_LDFLAGS@ @CUDA_LDFLAGS@
fmaref_DEPENDENCIES = fmaref_kernel.$(OBJEXT)

fmabench_SOURCES =
nodist_EXTRA_fmabench_SOURCES = dummy.cxx
fmabench_LDADD   = fmabench_kernel.$(OBJEXT) @CUDA_LIBS@
fmabench_LDFLAGS = @CUDA_SDK_LDFLAGS@ @CUDA_LDFLAGS@
fmabench_DEPENDENCIES = fmabench_kernel.$(OBJEXT)

## ======== elembench (elementary functions, standalone) ==============
## GPU counterpart of the dtq-0.0.3 elem_bench: accuracy of the
## elementary functions and polyeval against MPFR references, plus
## per-call timings.  Needs MPFR/GMP on the host side only.
elembench_SOURCES =
nodist_EXTRA_elembench_SOURCES = dummy.cxx
elembench_LDADD   = elembench_kernel.$(OBJEXT) @CUDA_LIBS@ -lmpfr -lgmp
elembench_LDFLAGS = @CUDA_SDK_LDFLAGS@ @CUDA_LDFLAGS@
elembench_DEPENDENCIES = elembench_kernel.$(OBJEXT)

## ======== nvcc rules ================================================

NVCC_INCLUDES = \
	-I$(top_srcdir)/inc \
	@CUDA_SDK_INCLUDE@

NVCC_DEFS = -D__NV_NO_VECTOR_DEPRECATION_DIAG

## elementary functions (gdtq 0.0.4): mw kernels behind gqd_elem.cu / gqs_elem.cu
ELEM_DEPS = \
	$(top_srcdir)/inc/gqd_elem.cu   $(top_srcdir)/inc/gqs_elem.cu \
	$(top_srcdir)/inc/gdtq_mw.cuh   $(top_srcdir)/inc/gdtq_twopass.cuh \
	$(top_srcdir)/inc/mw/mw_arith.h $(top_srcdir)/inc/mw/mw_elem.h \
	$(top_srcdir)/inc/mw/mw_tables.h $(top_srcdir)/inc/mw/mw_twopass.cuh

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
	$(ELEM_DEPS) \
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
	$(ELEM_DEPS) \
	$(top_srcdir)/inc/gqd_type.h

gqdtest_kernel.$(OBJEXT): gqdtest_kernel.cu $(KERNEL_DEPS)
	@NVCC@ @NVCC_FLAGS@ -arch=@CUDA_ARCH@ \
	    $(NVCC_DEFS) $(NVCC_INCLUDES) \
	    -c $(srcdir)/gqdtest_kernel.cu -o $@

sqstest_kernel.$(OBJEXT): sqstest_kernel.cu $(KERNEL_S_DEPS)
	@NVCC@ @NVCC_FLAGS@ -arch=@CUDA_ARCH@ \
	    $(NVCC_DEFS) $(NVCC_INCLUDES) \
	    -c $(srcdir)/sqstest_kernel.cu -o $@

fmatest_kernel.$(OBJEXT): fmatest_kernel.cu $(KERNEL_DEPS)
	@NVCC@ @NVCC_FLAGS@ -arch=@CUDA_ARCH@ \
	    $(NVCC_DEFS) $(NVCC_INCLUDES) \
	    -c $(srcdir)/fmatest_kernel.cu -o $@

## the host reference must not be contracted into FMAs
fmaref_kernel.$(OBJEXT): fmaref_kernel.cu $(KERNEL_DEPS)
	@NVCC@ @NVCC_FLAGS@ -arch=@CUDA_ARCH@ -Xcompiler -ffp-contract=off \
	    $(NVCC_DEFS) $(NVCC_INCLUDES) \
	    -c $(srcdir)/fmaref_kernel.cu -o $@

fmabench_kernel.$(OBJEXT): fmabench_kernel.cu $(KERNEL_DEPS)
	@NVCC@ @NVCC_FLAGS@ -arch=@CUDA_ARCH@ \
	    $(NVCC_DEFS) $(NVCC_INCLUDES) \
	    -c $(srcdir)/fmabench_kernel.cu -o $@

elembench_kernel.$(OBJEXT): elembench_kernel.cu $(KERNEL_DEPS) $(KERNEL_S_DEPS)
	@NVCC@ @NVCC_FLAGS@ -arch=@CUDA_ARCH@ \
	    $(NVCC_DEFS) $(NVCC_INCLUDES) \
	    -c $(srcdir)/elembench_kernel.cu -o $@

## Convenience targets: run the FMA accuracy test / benchmark.
fmacheck: fmatest$(EXEEXT) fmaref$(EXEEXT)
	./fmatest$(EXEEXT)
	./fmaref$(EXEEXT)

bench: fmabench$(EXEEXT)
	./fmabench$(EXEEXT)

elembench-run: elembench$(EXEEXT)
	./elembench$(EXEEXT)

.PHONY: fmacheck bench elembench-run

CLEANFILES = gqdtest_kernel.$(OBJEXT) sqstest_kernel.$(OBJEXT) \
	fmatest_kernel.$(OBJEXT) fmabench_kernel.$(OBJEXT) \
	elembench_kernel.$(OBJEXT) fmaref_kernel.$(OBJEXT)

EXTRA_DIST = \
	gqdtest_kernel.cu \
	sqstest_kernel.cu \
	fmatest_kernel.cu \
	fmabench_kernel.cu \
	elembench_kernel.cu \
	fmaref_kernel.cu \
	clean.sh \
	make.sh \
	README_CUDA13_FIX.md
