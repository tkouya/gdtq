## SPDX-License-Identifier: BSD-3-Clause
## Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
##
## Two ways to consume gdtq are supported, both installed by `make install`:
##
##  (1) Consolidated compiled library  $(libdir)/libgqd.a
##      Built here from the two umbrella translation units gqd.cu (the
##      double-based DD/TD/QD layers) and gqs.cu (the float-based
##      DS/TS/QS layers) with relocatable device code (-rdc=true), then
##      archived into a single static library.  Client code includes the
##      *declaration* headers (gqd.cuh / gqs.cuh), compiles with
##      -rdc=true, and device-links against -lgqd.  See README.md.
##
##  (2) Header / source include (the original 0.0.1 model)
##      Every .cu/.cuh translation unit is installed under
##      $(includedir)/gdtq/ so existing code that does `#include "gqd.cu"`
##      (or "gqs.cu") from a single .cu file keeps working unchanged.

gdtqincludedir = $(includedir)/gdtq

gdtqinclude_HEADERS = \
	gqd_type.h

gdtqinclude_DATA = \
	common.cu        common.cuh \
	cuda_header.cu \
	inline.cu        inline.cuh \
	gdd_basic.cu     gdd_basic.cuh \
	gdd_sqrt.cu      gdd_sqrt.cuh \
	gdd_exp.cu       gdd_exp.cuh \
	gdd_log.cu       gdd_log.cuh \
	gdd_sincos.cu    gdd_sincos.cuh \
	gtd_basic.cu     gtd_basic.cuh \
	gtd_sqrt.cu      gtd_sqrt.cuh \
	gtd_exp.cu       gtd_exp.cuh \
	gtd_log.cu       gtd_log.cuh \
	gtd_sincos.cu    gtd_sincos.cuh \
	gqd.cu           gqd.cuh \
	gqd_basic.cu     gqd_basic.cuh \
	gqd_sqrt.cu      gqd_sqrt.cuh \
	gqd_exp.cu       gqd_exp.cuh \
	gqd_log.cu       gqd_log.cuh \
	gqd_sincos.cu    gqd_sincos.cuh \
	common_s.cu      common_s.cuh \
	inline_s.cu      inline_s.cuh \
	gds_basic.cu     gds_basic.cuh \
	gds_sqrt.cu      gds_sqrt.cuh \
	gds_exp.cu       gds_exp.cuh \
	gds_log.cu       gds_log.cuh \
	gds_sincos.cu    gds_sincos.cuh \
	gts_basic.cu     gts_basic.cuh \
	gts_sqrt.cu      gts_sqrt.cuh \
	gts_exp.cu       gts_exp.cuh \
	gts_log.cu       gts_log.cuh \
	gts_sincos.cu    gts_sincos.cuh \
	gqs.cu           gqs.cuh \
	gqs_basic.cu     gqs_basic.cuh \
	gqs_sqrt.cu      gqs_sqrt.cuh \
	gqs_exp.cu       gqs_exp.cuh \
	gqs_log.cu       gqs_log.cuh \
	gqs_sincos.cu    gqs_sincos.cuh

EXTRA_DIST = $(gdtqinclude_DATA) convert_to_float.py

## ====================================================================
## Consolidated library: libgqd.a (static) and libgqd.so (shared)
## ====================================================================
## automake has no native CUDA language, so the library is built with
## explicit nvcc rules and installed through the *-local hooks rather
## than via lib_LIBRARIES.
##
## Both umbrella TUs are compiled once, as position-independent
## relocatable device code (-rdc=true -Xcompiler -fPIC).  The same two
## objects feed both artifacts:
##
##   libgqd.a   static archive that keeps the relocatable __device__
##              code intact, so a consumer can DEVICE-LINK the operators
##              into its own kernels (-rdc=true on the consumer side).
##              This is the library to use for GPU kernel code.
##
##   libgqd.so  shared library; nvcc device-links it internally, so it
##              exports only the HOST-callable symbols (GxxStart/GxxEnd
##              and the __host__ __device__ functions usable on the CPU).
##              Device code cannot be device-linked out of a .so -- that
##              is a CUDA limitation, not a packaging choice.

LIBGQD    = libgqd.a
LIBGQD_SO = libgqd.so

## Every installed .cu/.cuh plus gqd_type.h triggers a rebuild.
LIBGQD_SRCDEPS = $(gdtqinclude_DATA) $(gdtqinclude_HEADERS)

NVCC = @NVCC@
LIBGQD_NVCCFLAGS = @NVCC_FLAGS@ -arch=@CUDA_ARCH@ -rdc=true \
	-Xcompiler -fPIC @CUDA_SDK_INCLUDE@ -I$(srcdir)

all-local: $(LIBGQD) $(LIBGQD_SO)

libgqd_dd.$(OBJEXT): $(srcdir)/gqd.cu $(LIBGQD_SRCDEPS)
	$(NVCC) $(LIBGQD_NVCCFLAGS) -c $(srcdir)/gqd.cu -o $@

libgqd_qs.$(OBJEXT): $(srcdir)/gqs.cu $(LIBGQD_SRCDEPS)
	$(NVCC) $(LIBGQD_NVCCFLAGS) -c $(srcdir)/gqs.cu -o $@

$(LIBGQD): libgqd_dd.$(OBJEXT) libgqd_qs.$(OBJEXT)
	rm -f $@
	$(AR) cr $@ libgqd_dd.$(OBJEXT) libgqd_qs.$(OBJEXT)
	$(RANLIB) $@

$(LIBGQD_SO): libgqd_dd.$(OBJEXT) libgqd_qs.$(OBJEXT)
	$(NVCC) -shared -arch=@CUDA_ARCH@ \
	    libgqd_dd.$(OBJEXT) libgqd_qs.$(OBJEXT) -o $@

install-exec-local: $(LIBGQD) $(LIBGQD_SO)
	$(MKDIR_P) "$(DESTDIR)$(libdir)"
	$(INSTALL_DATA) $(LIBGQD) "$(DESTDIR)$(libdir)/$(LIBGQD)"
	$(RANLIB) "$(DESTDIR)$(libdir)/$(LIBGQD)"
	$(INSTALL_PROGRAM) $(LIBGQD_SO) "$(DESTDIR)$(libdir)/$(LIBGQD_SO)"

uninstall-local:
	rm -f "$(DESTDIR)$(libdir)/$(LIBGQD)" "$(DESTDIR)$(libdir)/$(LIBGQD_SO)"

clean-local:
	rm -f $(LIBGQD) $(LIBGQD_SO) libgqd_dd.$(OBJEXT) libgqd_qs.$(OBJEXT)
