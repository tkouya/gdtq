# gdtq: Double-double, Triple-double, and Quadruple-double precision arithmetic on GPU (CUDA)
================================================
GDTQ Version 0.0.2
Copyright (C) 2026 Tomonori Kouya
based on the GQD library (Mian Lu) and QD library (Yozo Hida, Xiaoye S. Li,
David H. Bailey; LBNL)
================================================

gdtq is an extension of the GQD library that runs multi-precision
floating-point arithmetic inside CUDA kernels. It exposes six
unevaluated-sum types in two families:

  Double-based (each limb is an IEEE-754 double):

  * gdd_real - double-double  (~32 decimal digits, ~106-bit significand)
  * gtd_real - triple-double  (~47 decimal digits, ~156-bit significand)
  * gqd_real - quad-double    (~63 decimal digits, ~212-bit significand)

  Float-based (each limb is an IEEE-754 single, added in 0.0.2):

  * gds_real - double-single  (~14 decimal digits)
  * gts_real - triple-single  (~21 decimal digits)
  * gqs_real - quad-single    (~28 decimal digits)

The gtd_real / gts_real types are new in this package; the other types
follow the original GQD interface so existing GQD-based kernels should
compile unchanged.

A short usage guide is shipped under [docs/](docs/):

  * [docs/GDTQ_QUICKREF.en.md](docs/GDTQ_QUICKREF.en.md) - English
  * [docs/GDTQ_QUICKREF.ja.md](docs/GDTQ_QUICKREF.ja.md) - Japanese


-----------------------------------------------------------------------
Requirements
-----------------------------------------------------------------------
  * CUDA Toolkit (10.x or later; 13.0 has been verified).
  * A C++ compiler that nvcc accepts as the host compiler (GCC, Clang).
  * GNU make.
  * The CPU-side QD library (https://www.davidhbailey.com/dhbsoftware/),
    used by the benchmark for the gold reference.
  * (Optional) A CPU triple-double library such as the dtq package, to
    cross-check the gtd_real path against a true CPU TD reference.
  * (Optional) autoconf / automake, only if you regenerate ./configure
    from configure.ac.


-----------------------------------------------------------------------
Build and install
-----------------------------------------------------------------------
Standard autotools flow:

    ./bootstrap                # first checkout only; runs autoreconf -i
    ./configure --with-cuda=/usr/local/cuda \
                --with-cuda-arch=sm_90 \
                --with-qd=/usr/local
    make
    sudo make install

`make` builds the consolidated library from the two umbrella
translation units `gqd.cu` (DD/TD/QD) and `gqs.cu` (DS/TS/QS),
compiled once as position-independent relocatable device code, into
both a static and a shared form. `make install` then installs:

    $(libdir)/libgqd.a                 static library (device-linkable)
    $(libdir)/libgqd.so                shared library (host API only)
    $(includedir)/gdtq/*.cuh *.cu      headers + sources
    $(libdir)/pkgconfig/gqd.pc         pkg-config: static/device link
    $(libdir)/pkgconfig/gqd-shared.pc  pkg-config: shared/host link

Static vs shared -- which one to use:

  * `libgqd.a` is the one you want for **GPU kernel code**. The static
    archive keeps the relocatable device code intact, so the `__device__`
    operators can be device-linked into your own kernels.
  * `libgqd.so` exports only the **host-callable** API (`GxxStart` /
    `GxxEnd` plus the `__host__ __device__` functions, usable for CPU-side
    DD/TD/QD arithmetic and device initialization). Device code cannot be
    device-linked out of a `.so` -- that is a CUDA limitation, so the
    shared library is not a drop-in replacement for kernel use.

Local (non-root) install -- pick a prefix under your home directory so
no `sudo` is needed:

    ./configure --prefix=$HOME/.local \
                --with-cuda=/usr/local/cuda --with-cuda-arch=sm_90
    make
    make install                       # no sudo

    # let pkg-config and the linker find the local install:
    export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH
    export LIBRARY_PATH=$HOME/.local/lib:$LIBRARY_PATH

Useful configure options:

    --prefix=DIR              install under DIR (default /usr/local)
    --with-cuda=PATH          CUDA toolkit prefix
    --with-cuda-arch=ARCH     compute arch, e.g. sm_80, sm_90, sm_121
    --with-nvcc=PROG          override the nvcc path
    --with-nvcc-flags=FLAGS   extra flags forwarded to nvcc
    --with-cuda-sdk=PATH      legacy CUDA Samples common/ path (optional)
    --with-qd=PATH            CPU QD library install prefix
    --with-tdlib=PATH         CPU triple-double library prefix
    --with-tdlib-name=NAME    link library name (default: td -> -ltd)
    --with-tdlib-header=HDR   include header (default: td/td_real.h)
    --with-tdlib-type=NAME    C++ TD scalar type (default: td_real)
    --disable-benchmark       skip building test/benchmark

Notes:

  * `make install` builds and installs the compiled libraries
    `$(libdir)/libgqd.a` and `$(libdir)/libgqd.so`, copies the
    headers/sources under `inc/` to `$(includedir)/gdtq/`, installs
    `gqd.pc` / `gqd-shared.pc` for pkg-config, and copies the markdown
    guides under `docs/` to `$(docdir)`. Three consumption models are
    supported (see "Using the library" below): device-link the static
    `libgqd.a`, dynamically link the host-only `libgqd.so`, or keep the
    original single-translation-unit style of `#include "gqd.cu"`.
  * `make dist` produces a single `.tar.xz` tarball (gzip output is
    disabled by default; use `make dist-gzip` if you need a `.tar.gz`).

If `./configure` does not exist (e.g. fresh checkout) or you modified
`configure.ac`, regenerate it with:

    autoreconf -fi


-----------------------------------------------------------------------
Using the library from C++ (CUDA)
-----------------------------------------------------------------------
There are three ways to consume gdtq.

(A) Device-link the static libgqd.a  (recommended for kernel code)

Your kernels include only the *declaration* headers (`gqd.cuh` and/or
`gqs.cuh`), compile with `-rdc=true`, and device-link against `-lgqd`.
The `__device__` implementations live in `libgqd.a`, so they are
compiled once instead of in every translation unit.

app_kernel.cu:

    #include "cuda_header.cu"
    #include "gqd.cuh"                       // declarations only (DD/TD/QD)
    // ... your __global__ kernels using gdd_real / gtd_real / gqd_real ...

Build with pkg-config (preferred):

    nvcc -arch=sm_90 -rdc=true $(pkg-config --cflags gqd) \
         -c app_kernel.cu -o app_kernel.o
    nvcc -arch=sm_90 app_kernel.o $(pkg-config --libs gqd) \
         -lcudadevrt -o app

or spelling the paths out by hand:

    nvcc -arch=sm_90 -rdc=true -I$PREFIX/include/gdtq \
         -D__NV_NO_VECTOR_DEPRECATION_DIAG -c app_kernel.cu -o app_kernel.o
    nvcc -arch=sm_90 app_kernel.o -L$PREFIX/lib -lgqd -lcudadevrt -o app

Notes:
  * Use the *same* `-arch=sm_XX` you configured the library with --
    relocatable device code in a static archive is architecture-specific.
  * `-rdc=true` is required when compiling your kernels, and the final
    link must go through `nvcc`.
  * Link the **static** archive for kernels. `pkg-config --libs gqd`
    expands to the archive's full path (`$PREFIX/lib/libgqd.a
    -lcudadevrt`) on purpose: a bare `-lgqd` would resolve to
    `libgqd.so` and the device link would fail with "Undefined
    reference to ... in your .o". If you link by hand, pass the archive
    explicitly: `nvcc app.o $PREFIX/lib/libgqd.a -lcudadevrt -o app`.

(B) Link the shared libgqd.so for host-side use

If you only need the host API -- CPU-side DD/TD/QD arithmetic via the
`__host__ __device__` functions, and `GxxStart`/`GxxEnd` to initialise
the device -- you can dynamically link the shared library from ordinary
C++ (no nvcc needed):

    g++ -O2 $(pkg-config --cflags gqd-shared) -I/usr/local/cuda/include \
        host.cpp $(pkg-config --libs gqd-shared) -lcudart -o host

Your own GPU kernels still need model (A) or (C) for their device code.

(C) Single-translation-unit include  (the original 0.0.1 style)

The kernel side is one `.cu` translation unit that includes `gqd.cu`
(the definitions); the host side declares the launcher with
`extern "C"`. No library is linked -- nvcc handles everything in one
translation unit.

example_kernel.cu:

    #define __NV_NO_VECTOR_DEPRECATION_DIAG
    #include "cuda_header.cu"
    #include "gqd.cu"                       // brings in DD/TD/QD

    template <class T>
    __global__ void axpy(const T* a, const T* x, const T* y, T* z, unsigned n) {
        unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) z[i] = (*a) * x[i] + y[i];
    }

    extern "C" void run_dd_axpy(const gdd_real* a, const gdd_real* x,
                                const gdd_real* y, gdd_real* z, unsigned n) {
        dim3 b(128), g((n + 127) / 128);
        axpy<gdd_real><<<g, b>>>(a, x, y, z, n);
        cudaDeviceSynchronize();
    }

example.cpp (host):

    #define __NV_NO_VECTOR_DEPRECATION_DIAG
    #include <cuda.h>
    #include <qd/qd_real.h>
    #include <qd/fpu.h>
    #include "gqd_type.h"

    extern "C" void run_dd_axpy(const gdd_real*, const gdd_real*,
                                const gdd_real*, gdd_real*, unsigned);

    int main() {
        unsigned int cw; fpu_fix_start(&cw);
        GDDStart(0);                        // upload constant tables
        // ... cudaMalloc / cudaMemcpy / launch via run_dd_axpy ...
        GDDEnd();                           // also calls cudaDeviceReset()
        fpu_fix_end(&cw);
    }

Build and link:

    nvcc -arch=sm_90 -I/usr/local/include/gdtq \
         -D__NV_NO_VECTOR_DEPRECATION_DIAG \
         -c example_kernel.cu -o example_kernel.o
    g++ -O2 example.cpp example_kernel.o -lqd -lcudart -o example

Always pass `-D__NV_NO_VECTOR_DEPRECATION_DIAG` *before* any include
under CUDA 13+, otherwise the deprecated `double4` warning floods the
build. See [docs/GDTQ_QUICKREF.en.md](docs/GDTQ_QUICKREF.en.md) for the
full list of operators, math functions, and constants.


-----------------------------------------------------------------------
Using the library from C
-----------------------------------------------------------------------
The gdtq API is built around C++ operator overloads and templates and
cannot be used from plain C directly. The standard approach is to wrap
the GPU entry points in `extern "C"` functions defined in C++:

    /* mylib.h - includable from both C and C++ */
    typedef struct { double x, y; } mylib_dd;     /* layout-compatible
                                                     with gdd_real (double2) */

    void mylib_init(void);
    void mylib_shutdown(void);
    void mylib_dd_axpy(const mylib_dd* a, const mylib_dd* x,
                       const mylib_dd* y, mylib_dd* z, size_t n);

The C wrapper layer can `reinterpret_cast<gdd_real*>` user pointers and
forward to the kernel launcher above. Always link the final binary with
the C++ linker (`g++` or `nvcc`); the plain C linker will not pull in
the libstdc++ symbols required by the kernel translation unit. A full
template (`mylib_kernel.cu` / `mylib_wrap.cpp` / `mylib.h` / `mycode.c`)
is shown in section 7 of
[docs/GDTQ_QUICKREF.en.md](docs/GDTQ_QUICKREF.en.md).


-----------------------------------------------------------------------
Programs in test/
-----------------------------------------------------------------------
  * benchmark   - exercises +, *, /, sqrt, exp, log, sin, tan for every
                  precision (DD/TD/QD/DS/TS/QS), comparing the GPU
                  result against the corresponding CPU reference
  * sqstest     - small standalone test for the float-based gds/gts/gqs
                  layers, with its own main()

Run after `make`:

    cd test
    ./benchmark
    ./sqstest


-----------------------------------------------------------------------
Pitfalls
-----------------------------------------------------------------------
A short summary; see section 8 of the quick reference for more.

  * `GxxEnd()` calls `cudaDeviceReset()`, which wipes the constant
    tables of every precision currently on the device. When mixing
    precisions, call End exactly once at the very end.
  * Build the CPU-side QD / dd_real with `-ffp-contract=off`; otherwise
    FMA fusion silently breaks Two-Sum / Two-Prod and your CPU gold
    reference loses ~12 digits.
  * Defining `ALL_MATH` in `gqd_type.h` enables asin/acos/atan/sinh/...
    but compile time can stretch into hours. Leave it off unless needed.


-----------------------------------------------------------------------
Reporting issues
-----------------------------------------------------------------------
Please send bug reports and patches to <tkouya@gmail.com>.

References:

  M. Lu, B. He, Q. Luo, "Supporting extended precision on graphics
  processors", DaMoN '10, 2010 -- the original GQD library.

  Y. Hida, X. S. Li, D. H. Bailey, "Algorithms for Quad-Double
  Precision Floating Point Arithmetic", Proc. 15th IEEE Symposium on
  Computer Arithmetic (ARITH-15), 2001 -- the algorithms used by the
  CPU QD library on which both GQD and gdtq are based.
