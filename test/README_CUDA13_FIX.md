# CUDA 13 対応 GQD ライブラリ + テスト/ベンチマーク修正サマリ

CUDA Toolkit 13 (13.0 / 13.1 / 13.2) で警告なくビルドできるよう、
ライブラリ本体とテスト/ベンチマーク側を合計 9 ファイル修正しました。
その他のファイルは無変更です。

---

## 変更のあったファイル一覧

### ライブラリ本体

| ファイル          | 変更理由                                                                         |
| ----------------- | -------------------------------------------------------------------------------- |
| `gqd_type.h`      | CUDA 13 で `double4` が非推奨化（削除は CUDA 14 予定）。抑止マクロを定義         |
| `cuda_header.cu`  | Runtime API `[[nodiscard]]` 対応、タイマー/エラーチェックを CUDA Events で再実装 |
| `common.cu`       | `cudaThreadExit()` を `cudaDeviceReset()` に置換、戻り値キャスト                 |
| `inline.cu`       | `#ifdef GQD_FMS` 分岐のバグ修正（`*err` → `err`、`QD_FMS` → `GQD_FMS`）          |
| `gdd_exp.cu`      | `for` ループ内で外側の `i` を隠蔽するシャドウ警告を解消                          |
| `gqd_exp.cu`      | 同上                                                                             |
| `gqd_basic.cu`    | 5 引数 `renorm` 内のデッドストアを除去                                           |

### テスト/ベンチマーク

| ファイル            | 変更理由                                                                              |
| ------------------- | ------------------------------------------------------------------------------------- |
| `gqdtest_kernel.cu` | `cudaThreadSynchronize()` → `cudaDeviceSynchronize()`、先頭で非推奨警告抑止マクロ定義 |
| `benchmark.cpp`     | `<cuda.h>` より先に `__NV_NO_VECTOR_DEPRECATION_DIAG` を定義                          |

### 無変更ファイル

`common.cuh`, `gdd_basic.cu`, `gdd_basic.cuh`, `gdd_exp.cuh`, `gdd_log.cu`,
`gdd_log.cuh`, `gdd_sincos.cu`, `gdd_sincos.cuh`, `gdd_sqrt.cu`, `gdd_sqrt.cuh`,
`gqd.cu`, `gqd.cuh`, `gqd_basic.cuh`, `gqd_exp.cuh`, `gqd_log.cu`, `gqd_log.cuh`,
`gqd_sincos.cu`, `gqd_sincos.cuh`, `gqd_sqrt.cu`, `gqd_sqrt.cuh`, `inline.cuh`,
`test_util.h`, `test_util.cpp`, `test_common.h`, `gqdtest.h`

---

## 変更の詳細

### 1. `gqd_type.h` — `double4` 非推奨警告の抑止

CUDA 13.0 以降、`double4` / `long4` / `ulong4` / `longlong4` / `ulonglong4` は
非推奨となり、アラインメントを明示した新型 `*_16a` / `*_32a` への移行が推奨
されています（CUDA 14 で削除予定）。本ライブラリは `typedef double4 gqd_real`
としているため、全翻訳単位で該当警告が出ます。

`<vector_types.h>` のインクルード前にマクロ `__NV_NO_VECTOR_DEPRECATION_DIAG`
を定義すれば NVIDIA 公式の手順で警告がグローバルに抑止されます。

```cpp
#ifndef __NV_NO_VECTOR_DEPRECATION_DIAG
#define __NV_NO_VECTOR_DEPRECATION_DIAG
#endif
#include <vector_types.h>
```

**注意**: `benchmark.cpp` では `<cuda.h>` を経由して `<vector_types.h>` が
先に読み込まれる可能性があるため、**同マクロを benchmark.cpp の先頭でも
定義**しました。またビルド時に `-D__NV_NO_VECTOR_DEPRECATION_DIAG` を渡せば
より確実です。

### 2. `cuda_header.cu` — `[[nodiscard]]` 対応 & タイマ/エラーチェック再実装

**(a) Runtime API の戻り値** — CUDA 12 以降、`cudaError_t` には
`[[nodiscard]]` が付いているため、以下のように `(void)` キャストで
意図的に捨てます:

```cpp
/* 旧: 戻り値を捨てる (警告) */
#define CUDA_SAFE_CALL(function) function

/* 新 */
#define CUDA_SAFE_CALL(function) ((void)(function))
```

**(b) タイマ/エラーチェック** — 元のコードは CUDA Samples の
`helper_timer.h` (`StopWatchInterface`, `sdkCreateTimer`, …) と
`helper_cuda.h` (`getLastCudaError`) を使う設計でしたが、
CUDA 13 ではツールキット本体にこれらは含まれません。
そのため CUDA Events を使った自前実装を同ファイル内に
`static inline` で追加しました:

```cpp
typedef struct StopWatchInterface_s {
    cudaEvent_t start_event;
    cudaEvent_t end_event;
} StopWatchInterface;

static inline void  startTimer(StopWatchInterface **timer);
static inline float endTimer (StopWatchInterface **timer, const char *title);
static inline void  getLastCudaError(const char *msg);

#define cutilCheckMsg getLastCudaError
```

関数シグネチャを `const char*` に直したことで、
`startTimer(&t); endTimer(&t, "name");` のような文字列リテラル渡しで
`-Wwrite-strings` も出なくなります。

### 3. `common.cu` — `cudaThreadExit` → `cudaDeviceReset`

`cudaThreadExit()` は CUDA 4.0 以降の非推奨 API です。
機能は `cudaDeviceReset()` と同一なので置換します。

```cpp
void GDDEnd() {
    printf("GDD turns off...\n");
    CUDA_SAFE_CALL( cudaDeviceReset() );   // 旧: cudaThreadExit();
    printf("\tdone.\n");
}
```

併せて裸の `cudaSetDevice(device)` を `CUDA_SAFE_CALL(...)` で包みました。

### 4. `inline.cu` — `#ifdef GQD_FMS` 分岐のバグ修正

```cpp
/* 旧: 参照なのにポインタ逆参照 + マクロ名の誤記 QD_FMS */
*err = GQD_FMS(a, b, p);
*err = QD_FMS(a, a, p);

/* 新 */
err = GQD_FMS(a, b, p);
err = GQD_FMS(a, a, p);
```

通常ビルド(`GQD_FMS` 未定義)では無関係ですが、`-DGQD_FMS` 時に
コンパイルが通らないため修正しました。

### 5. `gdd_exp.cu` / `gqd_exp.cu` — ループ変数シャドウの解消

外側スコープの `int i = 0;` を内側の `for (int i = 0; ...)` が
隠蔽していたため、内側変数を `j` にリネーム。

### 6. `gqd_basic.cu` — デッドストアの除去

5 引数版 `renorm` 内で `s0 = c0; s1 = c1;` の直後に両方上書き
されていたため除去。動作は変化しません。

### 7. `gqdtest_kernel.cu` — `cudaThreadSynchronize` 置換(13 箇所)

```cpp
/* 旧: cudaThreadSynchronize() は CUDA 4.0 以降非推奨 */
cutilSafeCall(cudaThreadSynchronize());

/* 新 */
cutilSafeCall(cudaDeviceSynchronize());
```

また `StopWatchInterface` / `startTimer` / `endTimer` /
`getLastCudaError`（`cutilCheckMsg` 経由）は修正 2(b) により
`cuda_header.cu` 側から提供されるので、呼び出し側の変更は不要です。

### 8. `benchmark.cpp` — `include` 順を考慮した警告抑止

`benchmark.cpp` は `<cuda.h>` を `"test_util.h"`（=`"gqd_type.h"`）
より先に `include` しているため、`<vector_types.h>` が先に読み込まれ
`double4` に非推奨属性が付いてしまいます。
これを避けるためファイル先頭(他の `#include` より前)に

```cpp
#ifndef __NV_NO_VECTOR_DEPRECATION_DIAG
#define __NV_NO_VECTOR_DEPRECATION_DIAG
#endif
```

を追加しました。

---

## ビルド例

### nvcc 単体でビルドする場合

```shell
# 通常はこれで充分
nvcc -O2 -std=c++17 -arch=sm_75 \
     -Xcompiler="-fopenmp" \
     -D__NV_NO_VECTOR_DEPRECATION_DIAG \
     -o gqdtest gqdtest_kernel.cu benchmark.cpp test_util.cpp \
     -lqd -lgomp
```

### Makefile 例(nvcc + g++ 混在)

```makefile
NVCC      = nvcc
CXX       = g++
ARCH      = sm_75
CXXSTD    = -std=c++17
WARNFLAGS = -Wall -Wextra
# この 1 行で double4 非推奨警告をグローバル抑止
DEPFLAGS  = -D__NV_NO_VECTOR_DEPRECATION_DIAG

CXXFLAGS  = $(CXXSTD) $(WARNFLAGS) $(DEPFLAGS) -fopenmp -O2
NVCCFLAGS = $(CXXSTD) $(DEPFLAGS) -O2 -arch=$(ARCH) \
            -Xcompiler="$(WARNFLAGS) -fopenmp"

LIBS      = -lqd -lgomp -lcudart

gqdtest: gqdtest_kernel.o benchmark.o test_util.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

gqdtest_kernel.o: gqdtest_kernel.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

benchmark.o: benchmark.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test_util.o: test_util.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o gqdtest
```

### 備考

- CUDA 13 は最低サポート計算能力が 7.5 以上です。利用 GPU に応じて
  `-arch=` または `-gencode=` を適宜変更してください。
- `libqd` (David Bailey の QD ライブラリ) がシステムに必要です。
- 今回は **挙動を変えない** 最小修正で警告ゼロを目指しました。
  CUDA 14 で `double4` が完全に削除されるので、その時点では
  `typedef double4_16a gqd_real;` 等への本格的移行が必要になります。
