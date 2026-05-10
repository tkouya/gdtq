# gdtq クイックリファレンス（C++ / C 利用ガイド）

CUDA 上で **double-double / triple-double / quadruple-double** （DD/TD/QD）と
**double-single / triple-single / quadruple-single**（DS/TS/QS）の多倍長精度演算を
提供するヘッダ群 `gdtq` を、自分のプロジェクトから利用するときの簡易リファレンスです。

対象バージョン: **gdtq-0.0.2**（QD 2.3 / GQD 由来、`gtd_real` を追加）

---

## 1. ライブラリの形

`gdtq` には **コンパイル済みライブラリは存在しません**。すべてヘッダ（`.cuh`）と
インライン定義（`.cu`）で提供され、ユーザの `.cu` 翻訳単位に **`#include "gqd.cu"`**
（または float 系なら `gqs.cu`）することで取り込みます。`nvcc` が単一翻訳単位として
コンパイル・リンクします。

```
inc/gqd.cu    … DD/TD/QD 一式の取り込み口（ホスト＋デバイス）
inc/gqs.cu    … DS/TS/QS 一式の取り込み口（ホスト＋デバイス）
inc/gqd.cuh   … 型と関数プロトタイプ（ヘッダだけが必要なら）
inc/gqd_type.h… 型定義（gdd_real など）と GxxStart/End の宣言
```

`make install` 後は `$(prefix)/include/gdtq/` 以下に `inc/` 内容が複製されます。

---

## 2. 型一覧

| 型 | 実体 | 概略精度 | 備考 |
|---|---|---|---|
| `gdd_real` | `double2`  | ~32桁 (2×53bit) | DD：QD 由来 |
| `gtd_real` | `double3`  | ~48桁 (3×53bit) | TD：本パッケージで追加 |
| `gqd_real` | `double4`  | ~64桁 (4×53bit) | QD |
| `gds_real` | `float2`   | ~14桁 (2×24bit) | DS：float 版 |
| `gts_real` | `float3`   | ~21桁 (3×24bit) | TS |
| `gqs_real` | `float4`   | ~28桁 (4×24bit) | QS |

CUDA 13 以降は `double4` 等が deprecated 警告を出します。
`__NV_NO_VECTOR_DEPRECATION_DIAG` を **すべての include より前** に定義して抑止します
（ヘッダ内で自動定義していますが、ホスト `.cpp` から `<cuda.h>` を直接読む場合は
利用側でも先頭で定義しておくと安全）。

---

## 3. ビルドと autoconf 設定

```sh
./bootstrap                   # 初回のみ。autoreconf --install
./configure --with-cuda=/usr/local/cuda \
            --with-cuda-arch=sm_90 \
            --with-qd=/usr/local
make
sudo make install
```

主な `configure` オプション（`README` 参照）:

| オプション | 意味 |
|---|---|
| `--with-cuda=PATH` | CUDA toolkit ルート |
| `--with-cuda-arch=sm_XX` | 対応アーキ（sm_80, sm_90, sm_121 など） |
| `--with-qd=PATH` | CPU 側 QD ライブラリ |
| `--with-tdlib=PATH` | CPU 側 TD ライブラリ（あれば） |
| `--disable-benchmark` | benchmark をビルドしない |

**自前のプロジェクトから利用するとき**、`nvcc` への必須フラグは:

```sh
nvcc -arch=sm_90 \
     -I/usr/local/include/gdtq \
     -D__NV_NO_VECTOR_DEPRECATION_DIAG \
     mycode.cu -o mycode \
     -lqd                     # CPU 側で qd_real を使う場合
```

---

## 4. 初期化（重要）

各精度クラスは sin/cos などのテーブルを `__constant__` メモリに転送するため、
**ホスト側で必ず初期化関数を呼ぶ** 必要があります。

```cpp
GDDStart();   GDDEnd();    // double-double
GTDStart();   GTDEnd();    // triple-double
GQDStart();   GQDEnd();    // quad-double
GDSStart();   GDSEnd();    // double-single
GTSStart();   GTSEnd();    // triple-single
GQSStart();   GQSEnd();    // quad-single
```

引数は `int device`（既定 0）。`GxxEnd()` は内部で `cudaDeviceReset()` を呼ぶため、
**複数の精度クラスを併用する場合は最後に 1 回だけ End を呼ぶ** こと。
`benchmark.cpp` のコメントにあるように

```cpp
GQDStart();
GTDStart();   // QD のテーブルを上書きしない順序で先に
... 計算 ...
GTDEnd();     // この呼び出しで cudaDeviceReset がかかる
              // GQDEnd() を後に呼んでも何も残っていない
```

---

## 5. ホスト側の使い方（C++）

CPU 側の `qd_real` / `dd_real`（QD ライブラリ）と GPU 側の `gqd_real` / `gdd_real`
の間に変換ヘルパが用意されています（`test/test_util.h`）。プロジェクトでも同じ
パターンを使うのがおすすめです。

```cpp
#define __NV_NO_VECTOR_DEPRECATION_DIAG
#include <cuda.h>
#include <qd/qd_real.h>
#include <qd/fpu.h>
#include "gqd_type.h"        // gdd_real, GDDStart など

// kernel ラッパは別 .cu に分けて宣言だけ extern する
extern "C" void run_dd_add(const gdd_real* a, const gdd_real* b,
                           gdd_real* c, unsigned n);

int main() {
    unsigned int cw;
    fpu_fix_start(&cw);                     // x87 を 64bit 丸めに
    GDDStart(0);

    const unsigned N = 1 << 20;
    dd_real *ha = new dd_real[N];
    dd_real *hb = new dd_real[N];
    for (unsigned i = 0; i < N; ++i) { ha[i] = "1.5"; hb[i] = "2.5"; }

    // dd_real → gdd_real（hi, lo の 2 成分にコピーするだけ）
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

    run_dd_add(dA, dB, dC, N);              // ↓ のカーネルを呼ぶ

    gdd_real *hgc = new gdd_real[N];
    cudaMemcpy(hgc, dC, N * sizeof(gdd_real), cudaMemcpyDeviceToHost);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    GDDEnd();
    fpu_fix_end(&cw);
}
```

`test/test_util.cpp` の `qd2gqd` / `gqd2qd` を流用すれば、上の手書きコピーは
1 行ヘルパで済みます。

---

## 6. デバイス側の使い方（カーネル）

カーネル翻訳単位（`.cu`）では `gqd.cu`（または `gqs.cu`）を 1 度だけ取り込みます。
**演算子オーバーロードが装備済み** なので、組み込み型と同じ感覚で書けます。

```cpp
#define __NV_NO_VECTOR_DEPRECATION_DIAG
#include "cuda_header.cu"
#include "gqd.cu"            // DD/TD/QD すべて入る

template <class T>
__global__
void axpy_kernel(const T* a, const T* x, const T* y, T* z, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    z[i] = (*a) * x[i] + y[i];   // operator* / operator+ は DD/TD/QD すべて定義済み
}

extern "C"
void run_dd_add(const gdd_real* a, const gdd_real* b, gdd_real* c, unsigned n) {
    dim3 block(128), grid((n + 127) / 128);
    axpy_kernel<gdd_real><<<grid, block>>>(/* スカラ a 省略 */ b, b, b, c, n);
    cudaDeviceSynchronize();
}
```

### 6.1 サポート演算子（DD を例に。TD/QD/DS/TS/QS も同様）

- 算術: `+ - * /`（`gdd_real op gdd_real`、`gdd_real op double`、`double op gdd_real`）
- 単項: `negative(a)` / `-a`、`fabs(a)`
- 平方: `sqr(a)`、`sqrt(a)`、`mul_pwr2(a, p2)`、`ldexp(a, n)`
- 比較: `== != < <= > >=`（一部 host/device 両対応）
- 述語: `is_zero / is_one / is_positive / is_negative`
- 変換: `to_double(a)`、`make_dd / make_td / make_qd / make_ds / make_ts / make_qs`
- 関数: `exp / log / sin / cos / tan`（DD/TD/QD/DS/TS/QS すべて）
- `ALL_MATH` を `gqd_type.h` で有効にすると `asin/acos/atan/sinh/cosh/tanh/...` も入る
  （**コンパイル時間が数時間に達することがある** ので注意）

### 6.2 主な定数

```
_dd_eps  _dd_e  _dd_log2  _dd_pi  _dd_pi2  _dd_pi4  _dd_2pi  _dd_3pi4
_td_eps  _td_e  _td_log2  _td_pi  _td_pi2  _td_pi4  _td_2pi  _td_pi1024
_qd_eps  _qd_e  _qd_log2  _qd_pi  _qd_pi2  _qd_pi4  _qd_2pi  _qd_3pi4 _qd_pi1024
_ds_eps  _ds_e  _ds_log2  ...                       # float 系も同形
```

---

## 7. C コードから使うには

`gdtq` の API は **C++ オーバーロード演算子とテンプレート** を多用しているため、
純粋な C からそのまま呼ぶことはできません。**C++ で extern "C" 関数を切り出して
ラップする** のが定石です。

### 7.1 ラッパを置くファイル構成

```
mylib_kernel.cu     // gqd.cu を include、テンプレート/演算子を使うカーネル
mylib_wrap.cpp      // ホスト関数を extern "C" で公開
mylib.h             // C から見える関数宣言（C/C++ の両方から include 可）
mycode.c            // 純 C
```

### 7.2 mylib.h（C と C++ の両方から include）

```c
#ifndef MYLIB_H
#define MYLIB_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* DD を 2 つの double で受け渡す。gdd_real の中身は double2 と等価。 */
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

/* mylib_dd と gdd_real (double2) はメモリレイアウト互換なので
   reinterpret_cast でデバイスに渡せる。              */
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

### 7.4 mycode.c（純 C）

```c
#include "mylib.h"

int main(void) {
    mylib_init();
    /* ... cudaMalloc / cudaMemcpy などは別の C ラッパに同様に分離する ... */
    mylib_shutdown();
    return 0;
}
```

リンクは C++ リンカ（`g++` または `nvcc` 経由）で行うこと。
C リンカを直接使うと `libstdc++` のシンボル（`__cxa_guard_acquire` 等）が
解決できず失敗します（`test/Makefile.am` の `sqstest` の `dummy.cxx` の理由）。

---

## 8. よくある落とし穴

| 症状 | 原因と対処 |
|---|---|
| `double4 is deprecated` の山 | include 前に `#define __NV_NO_VECTOR_DEPRECATION_DIAG` |
| `d_sin_table is undefined` 等 | `GxxStart()` の呼び忘れ |
| 2 つ目の精度で結果が壊れる | 先に呼んだ `GxxEnd()` で `cudaDeviceReset()` が走り `__constant__` テーブルが消えた。**End は最後に 1 回だけ** |
| Two-Sum が壊れて exp/log の精度が～10¹² eps 落ちる | ホスト側 CPU 比較に使う QD/dd_real のビルドで FMA 融合が起きている。CPU 側を `-ffp-contract=off` でビルドする（dtq 側のメモも参照） |
| ホスト `.cpp` でリンクエラー | `qd` ライブラリの `fpu_fix_start/end` を呼んでいない、または `-lqd` の付け忘れ |
| C リンカで未定義シンボル | リンカを `g++`／`nvcc` に切替。Automake なら `nodist_EXTRA_xxx_SOURCES = dummy.cxx` |
| sin/cos/tan が遅すぎ／コンパイルが終わらない | `gqd_type.h` の `ALL_MATH` を有効にしたまま。要らなければ無効化 |

---

## 9. 参考ファイル

- `test/benchmark.cpp` … ホスト側の典型例（DD/TD/QD/DS/TS/QS をすべて回す）
- `test/gqdtest_kernel.cu` … カーネル翻訳単位の典型例
- `test/test_util.cpp` / `test_util.h` … `qd_real ↔ g*_real` 変換ヘルパ
- `test/sqstest_kernel.cu` … 自前 `main()` を持つ小さな float 系単独テスト
- `test/README_CUDA13_FIX.md` … CUDA 13 移行時の修正点
- `inc/gqd_type.h` … 型と `GxxStart/End` の宣言
- `inc/common.cuh` / `inc/common_s.cuh` … 定数とテーブルの宣言
