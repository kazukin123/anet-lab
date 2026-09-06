# nn_util 実装の cpp 分離 仕様書（011 フォローアップ）

## Context（背景・目的）

011（FusedAdamW 導入、`docs/memo/011_fused_adamw_10prd.md`）で `core/anet-core/include/anet/nn_util.hpp` に ATen 内部ヘッダ 5 本と約 190 行の実装が追加され、ヘッダが肥大化した。実装を新設 `core/anet-core/src/nn_util.cpp` へ移し、以下を達成する。

1. **ATen 内部 op（アンダースコア API）依存の隔離** — `_fused_adamw_` 等の使用箇所が nn_util.cpp の 1 TU に閉じ、LibTorch 更新時の追従箇所が一点に集約される（`docs/adr/0003-fused-adamw-via-aten.md` の Consequences と整合）。
2. **ヘッダ API 面の縮小** — `FusedAdamWStepGroup` / `MakeFusedAdamWStepGroupKey` のような実装詳細をヘッダから消す。
3. 再コンパイル波及の削減（nn_util.hpp は `dqn_based_agent.hpp` 経由で広く伝播）。

**性能影響の評価（検討済み・実装時の再検討不要）**: 移動対象はすべて学習 step あたり高々数回呼ばれる関数で、本体はテンソル演算／カーネル起動（µs〜ms 級）。inline 喪失による関数呼び出しコスト（ns 級）は無視できる。挙動・性能を変えないリファクタリングであり、最適化の追加・削除はしない。

## 1. 移動対象（nn_util.hpp → 新設 nn_util.cpp）

| 対象 | 移し方 |
|---|---|
| `CollectDefinedGrads` / `ForeachGradNorm` / `ForeachClipGradNorm_` | 宣言をヘッダに残し、本体を cpp へ（`inline` 除去） |
| `FusedAdamWStepGroup` 構造体 / `MakeFusedAdamWStepGroupKey` | **ヘッダから削除し cpp 内へ完全移動**（実装詳細。無名 namespace は使わず、名前付き namespace に置く） |
| `FusedAdamW::step()` / `FusedAdamW::load()` | クラス定義はヘッダに残し、メソッド本体を cpp へ |
| `GradScaler::Unscale_` / `Step(optimizer)` / `Step(optimizer, bool)` / `Update` | 同上（クラス定義・ctor・`Scale` はヘッダ残留で可） |
| `Autocast` の ctor / dtor 本体 | cpp へ移し、ヘッダから `ATen/autocast_mode.h` を外す |

**ヘッダに残すもの**: `ApplyHeNormal` / `ApplyXavierUniform`（テンプレートのため言語制約で残留必須）、各クラス定義とメンバ変数、関数宣言。

## 2. include の整理

- **nn_util.hpp 側**: `ATen/ops/_amp_foreach_non_finite_check_and_unscale.h` / `_foreach_add.h` / `_foreach_mul.h` / `_foreach_norm.h` / `_fused_adamw.h`、`ATen/autocast_mode.h`、`anet/common.hpp`、`anet/profile.hpp`、`<map>`、`<string>` を削除。残るのは `torch/torch.h`、`<unordered_map>`、`<vector>` 程度（残す include は実際の使用に合わせて最小化）。
- **nn_util.cpp 側**: `#include "anet/nn_util.hpp"` を先頭に、上記で外したものを移す。
- include 削除によりヘッダの include 元（`nn_test.cpp` / `dqn_based_agent.hpp` / `rainbow_agent.cpp` / `default_dqn_agent.cpp` / `nn_impl.cpp`）で間接 include に依存していた箇所がビルドエラーになった場合は、その TU に必要な include を直接追加する。

## 3. nn_util.cpp の書き方（リポジトリ規約）

- AGENTS.md 準拠: `.cpp` は `namespace ... {}` で全体を囲まず `using namespace anet;` を使用。自由関数は `anet::` 修飾定義（using-directive では定義にならない点に注意）、メンバ関数は `FusedAdamW::step` 形式で可。既存の hpp/cpp ペア（`serialize.cpp` / `profile.cpp`）の流儀を参照。
- cpp 内限定ヘルパ（`FusedAdamWStepGroup` 等）は**無名 namespace 禁止**。名前付き namespace（`anet`）に置く。
- CMake 変更不要（`GLOB_RECURSE CONFIGURE_DEPENDS "src/*.cpp"` が自動で拾う）。

## 4. 併せて対応（011 レビュー指摘の修正）

1. **日本語コメントの追記**（AGENTS.md コメントルール違反の解消。移動と同時に書く）:
   - `FusedAdamW::step()`: ①int64 step（シリアライズ正本）と fp32 デバイス step tensor の並行管理 — 「+1 前の値でキャッシュ生成 → int64/デバイス両方 +1 → カーネルには +1 済みが渡る。両者は決定的に一致し同期不要」の不変条件、②(device, dtype) グループ化の理由（fused は同 device・同 dtype 前提）、③grads を毎 step 再収集する理由（set_to_none で実体が変わる）
   - `FusedAdamW::load()`: 親 load 後にキャッシュを破棄し次回 step で int64 値から再構築する旨
   - foreach ヘルパ 3 関数: 役割と CPU 同期なしの意図
   - `GradScaler::Unscale_` の既存コメント「勾配を scale で割る」を更新（inf/NaN 検出と `found_inf_tensor_` 更新の副作用を追記）
2. `core/anet-core/src/nn_test.cpp` の未使用 `#include <tuple>` を削除。
3. `core/anet-core/src/dqn_based_agent.cpp` 先頭に付与された BOM を除去（diff ノイズ）。

## 5. 変更しないもの（Out of Scope）

- 挙動・数値・性能特性（純リファクタリング。ロジック変更禁止）。
- `rainbow_agent.hpp` の `use_fused_optimizer = false` 固定（別途判断中。触らない）。
- テストのロジック（`nn_test.cpp` のテスト本体は無変更で全パスすること。`<tuple>` 削除のみ）。
- muzero / image_cls、zero_grad、PRD 011 の Out of Scope 全項目。

## 6. 検証

1. `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'`
2. `core\anet-core\bin\Debug\anet-core-test.exe` — 011 で追加したテスト含め全パス（テスト無変更で通ること自体が挙動不変の検証）
3. 全体ビルド `cmake --build --preset x64-Debug` で runner も通ること
4. `git diff --check`（行末・空白）
