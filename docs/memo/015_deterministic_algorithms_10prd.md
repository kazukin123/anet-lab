# `backend.deterministic_algorithms` 導入（学習の同 seed 再現性確保）仕様書

## Context（背景・目的）

学習 Run の **同 seed → 同結果（再現性）が失われていた**。原因は **SDPA（`at::scaled_dot_product_attention`）が CUDA で Memory-Efficient backend（cutlass fmha）にディスパッチされ、その backward が gradient を atomic 加算するため非決定**だったこと。forward は決定的なので「eval は再現するが train（loss/weight）だけ run ごとに割れる」という症状になる。

`backend.cudnn_deterministic=true`（既定 ON、`init.cpp:101`）は **cuDNN 畳み込み専用**で、ATen の flash/mem-efficient カーネルである SDPA には効かない、という穴だった。

実機で **`ctx.setDeterministicAlgorithms(true, /*warn_only=*/false)` の 1 行追加だけで再現性が復帰**することを確認済み（mem-efficient は決定的 backward 変種を持ち、このフラグで atomic 回避経路に切替わる）。`CUBLAS_WORKSPACE_CONFIG` は当該環境では不要だった（throw せず再現）。SDPA 化は実時間ほぼ不変＝attention は実時間ボトルネックでないため、決定モードの低速化は無視できる。

この 1 行 fix を、既存 backend フラグ（`cudnn_deterministic` 等）と同じ作法で **config フラグ化**する。`cudnn_deterministic=true` が既定である＝本プロジェクトは「決定論的を既定」方針なので、SDPA の穴を塞ぐ本フラグも **既定 true** とする。将来「決定版が無い op」を踏んで throw した場合の退避用に `deterministic_warn_only`（既定 false）も公開する。

採用判断の経緯と棄却案は `docs/adr/0006-deterministic-algorithms.md` を参照。

## 1. 前提事実（調査済み・再調査不要）

### 1.1 API（`torch::globalContext()` が返す `at::Context`）

```cpp
// init.cpp:98 で取得済みの torch::Context& ctx に対して呼ぶ。追加 include 不要（torch/torch.h 経由）。
void at::Context::setDeterministicAlgorithms(bool mode, bool warn_only);
bool at::Context::deterministicAlgorithms() const;
// 既存利用の前例（同じ ctx の setter）: ctx.setDeterministicCuDNN(...) が init.cpp:101 にある。
```

- `mode=true` でグローバル決定論ポリシーが ON：(a) 決定版を持つ op（`index_add`/`scatter_add` 等）を決定版へ強制、(b) 決定版が無い op は `warn_only=false`→**op 名付きで throw**、`warn_only=true`→**警告（TORCH_WARN）して非決定のまま実行**。
- **`warn_only=false` が「真の決定モード」**。`warn_only=true` は再現性を保証しない（非決定 op を素通りさせる）診断・退避用。
- `mode=false` のとき `warn_only` は無視される（特別扱い不要）。本プロジェクトは現状この関数を呼んでいない＝既定状態は `mode=false`。

### 1.2 SDPA backend と非決定（症状の根拠）

- SDPA は backend（flash / mem-efficient / math / cuDNN）を入力 dtype・形状・SDP context から自動選択。本モデル（`nn_modules.cpp:1289` の `at::scaled_dot_product_attention`）は CUDA で **mem-efficient** が選ばれている（`setDeterministicAlgorithms(true, warn_only=true)` で `attention_backward.cu:902` の "Memory Efficient attention defaults to a non-deterministic algorithm" 警告を確認）。
- flash/mem-efficient の **forward は決定的、backward は atomic 加算で非決定**。mem-efficient は **決定的 backward 変種**を持ち、`deterministicAlgorithms()==true && !warn_only` で atomic 回避経路（遅いが決定的）に切替わる。→ **backend の固定（`setSDPUse*`）は不要**で、本フラグ 1 つで再現性が戻る（実測確認済み）。

### 1.3 cudnn_deterministic では不足な理由

`cudnn_deterministic`（`setDeterministicCuDNN`）は cuDNN 畳み込みの算法選択のみを決定化する狭いフラグ。SDPA は ATen の flash/mem-efficient カーネル経由で cuDNN を通らないため、別途 `setDeterministicAlgorithms` が要る。両者は別レイヤーで、併用する（`cudnn_deterministic` は据え置き）。

### 1.4 config 読み込み・出力の前提

- `BackendConfig`（`core/anet-core/include/anet/init.hpp:12-33`）は `ANET_READ_CONFIG(config_data, <field>)` で各フィールドを読む。`cudnn_deterministic` 等の bool は明示 `my_config_data_.Set` 無しで読み込み＝**`MetricsLogger::Instance()->Log(backend_config)`（`init.cpp:105`）に自動的に載る**。新フラグも同じ作法で出力に乗る。

## 2. 設計方針

- 実コードは **`setDeterministicAlgorithms` 1 本**（あなたの確認済み fix そのまま）。`setSDPUse*` / `CUBLAS_WORKSPACE_CONFIG` は **コードに入れず、参照コメントとして残す**（§4.2）。理由は ADR 0006 の Considered Options。
- 既定 **true**（`cudnn_deterministic=true` と同じ「決定論的を既定」方針に揃え、SDPA 非決定の穴を塞ぐ）。
- `deterministic_warn_only`（既定 false）を公開し、将来「決定版が無い op」追加で throw した時に**再ビルド無しで非決定運転へ退避**できる逃げ道を持たせる（`warn_only=true` は再現性を保証しない点をコメント・config 注釈で明示）。

## 3. BackendConfig 改修（`core/anet-core/include/anet/init.hpp:12-33`）

`cudnn_benchmark` の隣に 2 フィールド追加（Doxygen コメント付き）。コンストラクタに `ANET_READ_CONFIG` を 2 行追加。

```cpp
/// 全 ATen op を決定化して同 seed 再現性を確保する（cuDNN 外＝SDPA 等の非決定もカバー）。
/// 決定版が無い op に当たると warn_only に従い throw/警告する。既定 true（cudnn_deterministic と同方針）。
bool deterministic_algorithms = true;
/// true: 決定版が無い op を例外でなく警告で素通りさせる（再現性は保証されない／throw 退避・診断用）。
/// deterministic_algorithms=false のときは無視される。既定 false。
bool deterministic_warn_only = false;
```

コンストラクタ（`init.hpp:23-28` の `ANET_READ_CONFIG` 群に追記）：

```cpp
ANET_READ_CONFIG(config_data, deterministic_algorithms);
ANET_READ_CONFIG(config_data, deterministic_warn_only);
```

## 4. InitRL 改修（`core/anet-core/src/init.cpp:101-102` の直後）

### 4.1 呼び出し

`setBenchmarkCuDNN(...)`（`init.cpp:102`）の直後、`MetricsLogger ... Log(backend_config)`（`init.cpp:105`）の前に追加。

### 4.2 挿入する内容（コメント全文 — 本 PRD の主目的）

以下を **そのまま** 挿入する（仕様情報をコメントとして詳しく残すのが本件の要件）。

```cpp
// --- 決定論的アルゴリズム（同 seed 再現性） -------------------------------------------
// setDeterministicAlgorithms はグローバルな「決定論ポリシー」スイッチ：
//   (a) 決定版を持つ op（index_add / scatter_add 等）を決定版へ強制する。
//   (b) 決定版が無い op は warn_only=false なら op 名付きで throw、true なら警告して
//       非決定のまま実行する。→ warn_only=false が「真の決定モード」。true は再現性を
//       保証しない（throw 退避・診断用）。
//
// なぜ cudnn_deterministic と別に要るか：
//   cudnn_deterministic は cuDNN 畳み込み限定。SDPA（at::scaled_dot_product_attention）は
//   ATen の flash / mem-efficient カーネルで cuDNN を通らず、その backward が atomic 加算で
//   非決定。これが同 seed 非再現の真因だった（forward は決定的＝eval は再現・train だけ割れる）。
//   mem-efficient は決定的 backward 変種を持ち、本フラグ(=true, warn_only=false)で atomic 回避
//   経路に切替わる。よって SDPA backend の固定は不要、このフラグ 1 つで再現性が戻る（実測確認済み）。
//
// SDPA backend を明示制御したい場合（本コードでは未使用・将来用の参照）：
//   at::Context の setSDPUseFlash / setSDPUseMemEfficient / setSDPUseMath / setSDPUseCuDNN(bool) で
//   候補を on/off できる（選択は「有効 ∧ 入力対応」を優先順位 flash>efficient>cuDNN>math で先着）。
//   math 固定でも決定化できるが遅く、mem-efficient の決定経路で足りるため採らない。
//   将来 dtype/shape 変更で flash が選ばれ、その LibTorch 版に flash の決定 backward が無いと
//   throw し得る → その時は setSDPUseFlash(false) で mem-efficient(決定)/math へ退避する。
//   ※ これら setter 名は LibTorch 版差あり（新しめは setSDPPriorityOrder）。ATen/Context.h で要確認。
//
// CUBLAS_WORKSPACE_CONFIG：決定モードで cuBLAS GEMM が env(:4096:8 等)を要求する場合がある。
//   失敗モードは silent ではなく throw。本環境では不要だった（throw せず再現）。将来 CUDA/cuBLAS/
//   形状変更で要求 throw が出たら、CUDA 初期化前に env を設定する（ApplyCudaLaunchBlockingConfig と同じ枠）。
//
// コスト：決定 backward は atomic 回避で遅いが attention は本 Run のボトルネックでない（SDPA 前後で
//   実時間ほぼ不変）ため無視可。他 op の決定版も僅かに遅くなる場合がある。
ctx.setDeterministicAlgorithms(backend_config.deterministic_algorithms,
                               backend_config.deterministic_warn_only);
```

## 5. 外部仕様（config 追加）

| キー | 型 | 既定値 | 意味 |
|---|---|---|---|
| `backend.deterministic_algorithms` | bool | `true` | 全 ATen op を決定化し同 seed 再現性を確保（SDPA 等 cuDNN 外の非決定もカバー）。false で高速だが非再現 |
| `backend.deterministic_warn_only` | bool | `false` | true: 決定版が無い op を例外でなく警告で素通り（**再現性は保証されない**／将来 op 追加で throw した時の一時退避）。`deterministic_algorithms=false` 時は無視 |

`apps/runner/config/common.txt` の `cudnn_*`（`:83-84`）の隣に、既存行と同じ整形・日本語インラインコメントで追記：

```
backend.deterministic_algorithms = true   # true:SDPA等の非決定opも決定化し同seed再現性を確保(やや遅い) false:高速だが非再現
backend.deterministic_warn_only  = false  # true:非決定opを例外でなく警告で素通り(再現性は保証されない/将来op追加でthrowした時の一時退避)
```

未指定時は `ANET_READ_CONFIG` が構造体既定値（true / false）を採用（既存読み込みフローと同じ）。

## 6. 修正対象ファイル

| ファイル | 変更内容 |
|---|---|
| `core/anet-core/include/anet/init.hpp` | `BackendConfig` に `deterministic_algorithms`/`deterministic_warn_only` 2 フィールド + `ANET_READ_CONFIG` 2 行 |
| `core/anet-core/src/init.cpp` | `InitRL` に `setDeterministicAlgorithms` 呼び出し 1 行 + §4.2 の詳細コメント |
| `apps/runner/config/common.txt` | `backend.deterministic_algorithms` / `backend.deterministic_warn_only` の 2 行 |

## 7. 既存利用可能な部品（再利用先）

- `ANET_READ_CONFIG`（`BackendConfig` 既存パターン、`init.hpp`）。
- `torch::globalContext()` の `ctx`（`init.cpp:98` で取得済み）/ `ctx.setDeterministicCuDNN`（`init.cpp:101`、同じ setter の前例）。
- `MetricsLogger::Instance()->Log(backend_config)`（`init.cpp:105`、新フラグを自動出力）。

## 8. 検証方針

config 追加のみ（既存 backend フラグにユニットテストが無いのと同様、新規単体テストは不要）。

1. **再現性（核心・ユーザー実施）**: `backend.deterministic_algorithms=true` で同 seed 2 run の **train 側 loss/weight 列が一致**することを確認。`false` で従来（非再現）に戻ることも確認。eval はどちらでも再現するはず（forward は決定的）。
2. **throw 退避の妥当性**: 将来「決定版が無い op」で throw した場合に、`backend.deterministic_warn_only=true` で警告のみになり Run が継続することを確認（逃げ道として機能するか）。
3. **ビルド**: VsDevCmd 経由で x64-Debug をビルドし `core\anet-core\bin\Debug\anet-core-test.exe` を実行（AGENTS.md 必須）。
4. **ログ**: 起動時メトリクス（`Log(backend_config)`）に 2 フラグが出ることを確認。

## 9. Out of Scope

- `setSDPUse*` の実コード投入（§4.2 の参照コメントのみ。実装は `setDeterministicAlgorithms` 1 本）。
- `CUBLAS_WORKSPACE_CONFIG` の自動設定（当該環境で不要確認済み。throw が出たら別途、env を CUDA 初期化前に設定）。
- cuDNN attention の決定経路採用（attention が実時間ボトルネックでないため見送り。重くなったら再検討）。
- SDPA 自体の見直し（速度メリットが出なかったが本 PRD の範囲外）。
- prefetch split revert（別件・独立。本件は `init.*` と config のみで無衝突）。
