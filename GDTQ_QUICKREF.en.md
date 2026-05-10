# gdtq Quick Reference (C++ / C Usage Guide)

A short reference for using `gdtq`, a header-only collection that provides
**double-double / triple-double / quadruple-double** (DD/TD/QD) and
**double-single / triple-single / quadruple-single** (DS/TS/QS) multi-precision
arithmetic on CUDA, from your own project.

Target version: **gdtq-0.0.2** (derived from QD 2.3 / GQD, with `gtd_real` added)

---

## 1. Library shape

`gdtq` ships **no compiled library**. Everything is provided as headers (`.cuh`)
and inline definitions (`.cu`); a user `.cu` translation unit pulls the whole
stack in via **`#include "gqd.cu"`** (or `gqs.cu` for the float layers).
`nvcc` then compiles and links it as a single translation unit.

```
inc/gqd.cu    … one-stop include for DD/TD/QD (host + device)
inc/gqs.cu    … one-stop include for DS/TS/QS (host + device)
inc/gqd.cuh   … types and prototypes (when only the header is needed)
inc/gqd_type.h… type definitions (gdd_real …) and GxxStart/End declarations
```

After `make install` the contents of `inc/` are copied to
`$(prefix)/include/gdtq/`.

---

## 2. Type list

| Type | Underlying | Approx. precision | Notes |
|---|---|---|---|
| `gdd_real` | `double2` | ~32 digits (2×53 bit) | DD; from QD |
| `gtd_real` | `double3` | ~48 digits (3×53 bit) | TD; added in this package |
| `gqd_real` | `double4` | ~64 digits (4×53 bit) | QD |
| `gds_real` | `float2`  | ~14 digits (2×24 bit) | DS; float-based |
| `gts_real` | `float3`  | ~21 digits (3×24 bit) | TS |
| `gqs_real` | `float4`  | ~28 digits (4×24 bit) | QS |

CUDA 13+ marks `double4` (and friends) as deprecated. Define
`__NV_NO_VECTOR_DEPRECATION_DIAG` **before any include** to suppress the
warning. The headers do this themselves, but if a host `.cpp` pulls
`<cuda.h>` in directly, define it at the top of that file too for safety.

---

## 3. Build and autoconf options

```sh
./bootstrap                   # first time only; runs autoreconf --install
./configure --with-cuda=/usr/local/cuda \
            --with-cuda-arch=sm_90 \
            --with-qd=/usr/local
make
sudo make install
```

Main `configure` options (see `README`):

| Option | Meaning |
|---|---|
| `--with-cuda=PATH` | CUDA toolkit root |
| `--with-cuda-arch=sm_XX` | Target compute capability (sm_80, sm_90, sm_121, …) |
| `--with-qd=PATH` | CPU-side QD library prefix |
| `--with-tdlib=PATH` | CPU-side TD library prefix (if available) |
| `--disable-benchmark` | Skip building `benchmark` |

When using `gdtq` from your **own** project, the minimum `nvcc` invocation is:

```sh
nvcc -arch=sm_90 \
     -I/usr/local/include/gdtq \
     -D__NV_NO_VECTOR_DEPRECATION_DIAG \
     mycode.cu -o mycode \
     -lqd                     # only if you also use qd_real on the host
```

---

## 4. Initialization (important)

Each precision class needs sin/cos / inverse-factorial tables uploaded to
`__constant__` memory, so **the host must call the matching init function**:

```cpp
GDDStart();   GDDEnd();    // double-double
GTDStart();   GTDEnd();    // triple-double
GQDStart();   GQDEnd();    // quad-double
GDSStart();   GDSEnd();    // double-single
GTSStart();   GTSEnd();    // triple-single
GQSStart();   GQSEnd();    // quad-single
```

The argument is `int device` (default 0). **`GxxEnd()` calls `cudaDeviceReset()`
internally**, which wipes the constant tables of *every* precision currently on
the device. So when you mix precisions, **call End exactly once at the very
end**. The `benchmark.cpp` sequence makes this concrete:

```cpp
GQDStart();
GTDStart();   // start TD before any End() that would reset the device
... compute ...
GTDEnd();     // this single call already does cudaDeviceReset();
              // calling GQDEnd() afterwards would find nothing left
```

---

## 5. Host-side usage (C++)

The `test/test_util.h` helpers convert between the CPU `qd_real` / `dd_real`
(from the QD library) and the GPU `gqd_real` / `gdd_real`. The same pattern
is the recommended one for your own code.

```cpp
#define __NV_NO_VECTOR_DEPRECATION_DIAG
#include <cuda.h>
#include <qd/qd_real.h>
#include <qd/fpu.h>
#include "gqd_type.h"        // gdd_real, GDDStart, ...

// keep the kernel in its own .cu and just declare the launcher here
extern "C" void run_dd_add(const gdd_real* a, const gdd_real* b,
                           gdd_real* c, unsigned n);

int main() {
    unsigned int cw;
    fpu_fix_start(&cw);                     // pin x87 to 64-bit rounding
    GDDStart(0);

    const unsigned N = 1 << 20;
    dd_real *ha = new dd_real[N];
    dd_real *hb = new dd_real[N];
    for (unsigned i = 0; i < N; ++i) { ha[i] = "1.5"; hb[i] = "2.5"; }

    // dd_real -> gdd_real (just copy the (hi, lo) pair)
    gdd_real *hga = new gdd_real[N];
    gdd_real *hgb = new gdd_real[N];
    for (unsigned i = 0; i < N; ++i) {
        hga[i] = make_double2(ha[i].x[0], ha[i].x[1]);
        hgb[i] = make_double2(hb[i].x[0], hb[i].x[1]);
    }

    gdd_real *dA, *dB, *dC;
    cudaMalloc(&dA, N * sizeof(gdd_real));
    cudaMalloc(&dB, N * sizeof(gdd_real));
    cudaMalloc(&dC, N * sizeof(gdd_real));
    cudaMemcpy(dA, hga, N * sizeof(gdd_real), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hgb, N * sizeof(gdd_real), cudaMemcpyHostToDevice);

    run_dd_add(dA, dB, dC, N);              // launches the kernel below

    gdd_real *hgc = new gdd_real[N];
    cudaMemcpy(hgc, dC, N * sizeof(gdd_real), cudaMemcpyDeviceToHost);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    GDDEnd();
    fpu_fix_end(&cw);
}
```

You can replace the manual element-wise copy with the `qd2gqd` / `gqd2qd`
helpers in `test/test_util.cpp`.

---

## 6. Device-side usage (kernels)

Inside a kernel translation unit (`.cu`), include `gqd.cu` (or `gqs.cu`)
exactly once. **Operator overloads are already in place**, so the code reads
like ordinary scalar arithmetic.

```cpp
#define __NV_NO_VECTOR_DEPRECATION_DIAG
#include "cuda_header.cu"
#include "gqd.cu"            // brings in DD/TD/QD

template <class T>
__global__
void axpy_kernel(const T* a, const T* x, const T* y, T* z, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    z[i] = (*a) * x[i] + y[i];   // operator* / operator+ are defined for DD/TD/QD
}

extern "C"
void run_dd_add(const gdd_real* a, const gdd_real* b, gdd_real* c, unsigned n) {
    dim3 block(128), grid((n + 127) / 128);
    axpy_kernel<gdd_real><<<grid, block>>>(a, a, b, c, n);
    cudaDeviceSynchronize();
}
```

### 6.1 Supported operators (shown for DD; TD/QD/DS/TS/QS are analogous)

- Arithmetic: `+ - * /` (`gdd_real op gdd_real`, `gdd_real op double`, `double op gdd_real`)
- Unary: `negative(a)` / `-a`, `fabs(a)`
- Powers: `sqr(a)`, `sqrt(a)`, `mul_pwr2(a, p2)`, `ldexp(a, n)`
- Comparison: `== != < <= > >=` (some host-and-device, some device-only)
- Predicates: `is_zero / is_one / is_positive / is_negative`
- Conversion: `to_double(a)`, `make_dd / make_td / make_qd / make_ds / make_ts / make_qs`
- Functions: `exp / log / sin / cos / tan` for every precision
- Defining `ALL_MATH` in `gqd_type.h` adds `asin/acos/atan/sinh/cosh/tanh/...`
  (**warning: compile time can stretch into hours**)

### 6.2 Predefined constants

```
_dd_eps  _dd_e  _dd_log2  _dd_pi  _dd_pi2  _dd_pi4  _dd_2pi  _dd_3pi4
_td_eps  _td_e  _td_log2  _td_pi  _td_pi2  _td_pi4  _td_2pi  _td_pi1024
_qd_eps  _qd_e  _qd_log2  _qd_pi  _qd_pi2  _qd_pi4  _qd_2pi  _qd_3pi4 _qd_pi1024
_ds_eps  _ds_e  _ds_log2  ...                       # same shape for the float layers
```

---

## 7. Calling from C

The `gdtq` API leans heavily on **C++ operator overloads and templates**, so
it cannot be called from plain C directly. The standard fix is to wrap the
hot paths in `extern "C"` functions defined in C++.

### 7.1 Suggested file layout

```
mylib_kernel.cu     // includes gqd.cu, defines templated/overloaded kernels
mylib_wrap.cpp      // exposes extern "C" host entry points
mylib.h             // C-visible declarations (includable from both C and C++)
mycode.c            // pure C
```

### 7.2 mylib.h (includable from C and C++)

```c
#ifndef MYLIB_H
#define MYLIB_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Pass DD as two doubles. The layout matches gdd_real (= double2). */
typedef struct { double x, y; } mylib_dd;

void mylib_init(void);
void mylib_shutdown(void);

void mylib_dd_axpy(const mylib_dd* a,
                   const mylib_dd* x,
                   const mylib_dd* y,
                   mylib_dd*       z,
                   size_t          n);

#ifdef __cplusplus
}
#endif
#endif
```

### 7.3 mylib_wrap.cpp

```cpp
#define __NV_NO_VECTOR_DEPRECATION_DIAG
#include <cuda.h>
#include "gqd_type.h"
#include "mylib.h"

extern "C" void mylib_init(void)     { GDDStart(0); }
extern "C" void mylib_shutdown(void) { GDDEnd();    }

/* mylib_dd and gdd_real (double2) are layout-compatible, so a
   reinterpret_cast is enough to hand the pointer to the device. */
extern void run_dd_axpy_kernel(const gdd_real*, const gdd_real*,
                               const gdd_real*, gdd_real*, size_t);

extern "C" void mylib_dd_axpy(const mylib_dd* a, const mylib_dd* x,
                              const mylib_dd* y, mylib_dd* z, size_t n) {
    run_dd_axpy_kernel(reinterpret_cast<const gdd_real*>(a),
                       reinterpret_cast<const gdd_real*>(x),
                       reinterpret_cast<const gdd_real*>(y),
                       reinterpret_cast<gdd_real*>(z), n);
}
```

### 7.4 mycode.c (pure C)

```c
#include "mylib.h"

int main(void) {
    mylib_init();
    /* ... cudaMalloc / cudaMemcpy via further C wrappers, same pattern ... */
    mylib_shutdown();
    return 0;
}
```

Link with the **C++ linker** (`g++` or `nvcc`). Driving the link with the
plain C linker fails to resolve `libstdc++` symbols such as
`__cxa_guard_acquire`. (This is the reason `test/Makefile.am` adds the
`dummy.cxx` placeholder for the `sqstest` link line.)

---

## 8. Common pitfalls

| Symptom | Cause / fix |
|---|---|
| Storm of `double4 is deprecated` warnings | `#define __NV_NO_VECTOR_DEPRECATION_DIAG` *before* any include |
| `d_sin_table is undefined` and friends | You forgot to call `GxxStart()` |
| Second precision class produces garbage | An earlier `GxxEnd()` ran `cudaDeviceReset()` and wiped the `__constant__` tables. **Call End exactly once, at the very end.** |
| exp/log lose ~10¹² eps when comparing CPU vs GPU | The CPU-side QD/dd_real was built with FMA contraction. Rebuild it with `-ffp-contract=off` (see the matching dtq note) |
| Host link error from QD | `fpu_fix_start/end` not called, or `-lqd` missing |
| Undefined symbols at link with the C linker | Switch to `g++` / `nvcc` for linking; with Automake add `nodist_EXTRA_xxx_SOURCES = dummy.cxx` |
| sin/cos/tan extremely slow / compile that never finishes | `ALL_MATH` is enabled in `gqd_type.h`; turn it off if you don't need the extra functions |

---

## 9. Files worth reading

- `test/benchmark.cpp` — canonical host-side example exercising every precision
- `test/gqdtest_kernel.cu` — canonical kernel translation unit
- `test/test_util.cpp` / `test_util.h` — `qd_real ↔ g*_real` conversion helpers
- `test/sqstest_kernel.cu` — small standalone float-layer test with its own `main()`
- `test/README_CUDA13_FIX.md` — notes on the CUDA 13 migration
- `inc/gqd_type.h` — type aliases and `GxxStart/End` declarations
- `inc/common.cuh` / `inc/common_s.cuh` — constants and table declarations
