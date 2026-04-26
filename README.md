# gdtq
----------------------------------------
QD library running on CUDA, originated by GQD, but including gtd_real class
Copyright (c) 2026 Tomonori Kouya

*** DIRECTORIES ***

inc/  : the inlined functions can be used in CUDA kernels
        (also installed to $(prefix)/include/gdtq/)
test/ : test cases and examples using the library
m4/   : autoconf macros (ax_cuda.m4)


*** BUILD AND INSTALL (autotools) ***

In a fresh checkout, generate the configure script first:

    ./bootstrap                    # runs autoreconf --install

Then configure, build, and install:

    ./configure [options]
    make
    sudo make install

Common configure options:

    --prefix=DIR              install prefix (default /usr/local)
    --with-cuda=PATH          CUDA toolkit prefix (default /usr/local/cuda)
    --with-cuda-arch=ARCH     compute arch, e.g. sm_80, sm_90, sm_121
    --with-nvcc=PROG          override nvcc path
    --with-nvcc-flags=FLAGS   extra flags for nvcc
    --with-cuda-sdk=PATH      legacy CUDA Samples common/ path (optional)

    --with-qd=PATH            QD library install prefix
    --with-qd-include=DIR     header dir explicitly
    --with-qd-lib=DIR         lib dir explicitly

    --with-tdlib=PATH         CPU triple-double library prefix
    --with-tdlib-include=DIR  TD library header dir
    --with-tdlib-lib=DIR      TD library lib dir
    --with-tdlib-name=NAME    link library name (default: td -> -ltd)
    --with-tdlib-header=HDR   include header (default: td/td_real.h)
    --with-tdlib-type=NAME    C++ TD scalar type (default: td_real)

    --disable-benchmark       skip building test/benchmark

Example for DGX Spark with the user's TD library installed under /opt/tdlib:

    ./configure --with-cuda=/usr/local/cuda \
                --with-cuda-arch=sm_121 \
                --with-tdlib=/opt/tdlib \
                --with-tdlib-name=td_real

`make install` copies the inc/ headers to $(includedir)/gdtq/. There is
no compiled library: client code includes "gqd.cu" from a .cu file and
nvcc handles everything in one translation unit.


*** NOTE ON THE LEGACY Makefile ***

The original hand-rolled top-level Makefile and inc-style gdtq.inc are
preserved for reference, but `./configure` will overwrite the top-level
Makefile with the autotools-generated one. Save a copy first if you
need it.

