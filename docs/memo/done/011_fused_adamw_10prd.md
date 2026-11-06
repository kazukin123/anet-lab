# FusedAdamW（ATen `_fused_adamw_` 直接呼び出し）導入 仕様書

## Context（背景・目的）

DQN Learner の `optimizer_->step()` が学習ループのボトルネック。Tracy 実測で optimizer_step 区間が支配的、CUDA_LAUNCH_BLOCKING 実測でカーネル起動律速と確認済み。LunarLander 級の小型 MLP では param テンソル ~10 個 × per-param 約 12 演算 ≈ **毎 step 100 回超のカーネル起動**が発生する。

PyTorch の `AdamW(fused=True)` 相当は LibTorch の C++ API に存在しないが、その実体である ATen op `at::_fused_adamw_` は使用中の **LibTorch 2.11.0+cu130 に CPU/CUDA 両ディスパッチで存在する**（確認済み）。これを直接呼ぶ `anet::FusedAdamW` を導入し、起動数を ~2/step へ削減する。併せて、同じホットパスに残る per-param ループ（grad_norm 手計算・手動 clip・GradScaler::Unscale_）を foreach 化する。

採用判断の経緯と棄却案は `docs/adr/0003-fused-adamw-via-aten.md` を参照。

**zero_grad は対象外と判明**: LibTorch 2.11 では `Optimizer::zero_grad(bool set_to_none = true)` がデフォルト true で、既存の `optimizer_->zero_grad()` は既にカーネル起動ゼロ（grad を undefined にリセットするだけ）。変更しない。

## 1. 前提事実（調査済み・再調査不要）

### 1.1 ATen op シグネチャ（`ATen/ops/_fused_adamw.h` で確認済み）

```cpp
// すべて torch/torch.h 経由で宣言済み（追加 include 不要）
at::_fused_adamw_(at::TensorList self, at::TensorList grads,
                  at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
                  at::TensorList max_exp_avg_sqs, at::TensorList state_steps,
                  double lr, double beta1, double beta2,
                  double weight_decay, double eps,
                  bool amsgrad, bool maximize,
                  const std::optional<at::Tensor>& grad_scale = {},
                  const std::optional<at::Tensor>& found_inf = {});
```

- `state_steps` は **fp32 scalar tensor（param と同デバイス）** のリスト。カーネルは渡された step 値（+1 済み）で bias correction を計算する。
- 全リストは同 device・同 dtype 前提（混在時は呼び出し側でグループ分けが必要）。
- `amsgrad = false` のとき `max_exp_avg_sqs` は空リストでよい。
- `maximize` は C++ の `AdamWOptions` に存在しないため **false 固定**。
- `grad_scale` / `found_inf` は今回未使用（`{}` を渡す）。grad clip 使用時は step 前 unscale が必須で利得が消えるため（ADR 参照）。

その他使用する op: `at::_foreach_add_`、`at::_foreach_norm`、`at::_foreach_mul_`、`at::_amp_foreach_non_finite_check_and_unscale_`（いずれも 2.11 に存在確認済み）。

### 1.2 LibTorch 2.11 の AdamW（`torch/csrc/api/include/torch/optim/adamw.h` で確認済み）

- `AdamW::step(LossClosure closure = nullptr)` / `save` / `load` は **virtual**。
- `AdamWParamState`（TORCH_ARG 生成アクセサ）: `step()` (int64_t) / `exp_avg()` / `exp_avg_sq()` / `max_exp_avg_sq()` (Tensor)。
- `AdamWOptions`: `lr` / `betas`（`std::tuple<double,double>`）/ `eps` / `weight_decay` / `amsgrad`。
- `Optimizer::state_` は protected。キーは `p.unsafeGetTensorImpl()`（void*）。

### 1.3 互換性の前提（このまま成立させること）

- 自作 `GradScaler`（`core/anet-core/include/anet/nn_util.hpp:68-135`）は `optimizer.param_groups()` 走査と仮想 `optimizer.step()` 呼び出しのみに依存 → **`torch::optim::AdamW` 派生ならそのまま動く**。
- チェックポイント: `Learner::Save/Load`（`core/anet-core/src/dqn_based_agent.cpp:1388-1400`）は `archive.WriteTorchObject(*optimizer_)` で仮想 save/load を呼ぶ → 派生で **旧 `torch::optim::AdamW` チェックポイントとの読み込み互換が自動成立**。
- メトリクス: `BatchUpdateResult::GetScalar`（`core/anet-core/src/dqn_based_agent.hpp:86-97`）は `grad_norm`（float）が無ければ `grad_norm_tensor` を遅延 `.item<float>()` するフォールバック実装済み → AMP パスを tensor 経路へ統一しても表示は壊れない。
- 設定読み込みは `ANET_READ_CONFIG(config_data, learner.xxx)` マクロ（`core/anet-core/include/anet/default_dqn_agent.hpp:42-`）。
- `nn_util.hpp` は `namespace anet` 直下・ヘッダオンリー・`#include <torch/torch.h>` 済み。

## 2. FusedAdamW クラス（`core/anet-core/include/anet/nn_util.hpp`、GradScaler の隣に追加）

```cpp
class FusedAdamW : public torch::optim::AdamW {
public:
    using torch::optim::AdamW::AdamW;   // コンストラクタ継承（3 箇所の構築コードと同形）
    torch::Tensor step(LossClosure closure = nullptr) override;
    void load(torch::serialize::InputArchive& archive) override;
private:
    // param TensorImpl* → デバイス上 fp32 step tensor（fused カーネル用の並行管理キャッシュ）
    std::unordered_map<void*, torch::Tensor> step_tensors_;
};
```

ヘッダ内 inline 実装（既存 GradScaler と同スタイル）。無名 namespace・フリーヘルパの無名化は禁止（プロジェクト規約）。

### 2.1 `step()` の処理手順

1. closure 対応: AdamW 本家と同形（`closure != nullptr` なら `at::AutoGradMode enable_grad(true)` 下で loss 評価）。以降は `torch::NoGradGuard`。
2. `ANET_PROFILE_FUNC()` を入れる（AGENTS.md の ProfileRange ルール。Learner 側の `optimizer_step` 区間とは別階層なので共存可）。
3. 各 param_group を走査。group の `AdamWOptions` から `lr` / `betas` / `eps` / `weight_decay` / `amsgrad` を取得。
4. grad が defined な param を収集。**sparse grad は `ANET_SYSTEM_ERROR` で明示的に落とす**（fused 非対応。暗黙フォールバック禁止の規約に従う）。
5. state lazy init: `state_[p.unsafeGetTensorImpl()]` が無ければ `AdamWParamState` を本家と同じ規約で生成（`step = 0`、`exp_avg` / `exp_avg_sq` = `torch::zeros_like(p, MemoryFormat::Preserve)`、amsgrad 時のみ `max_exp_avg_sq` も）。
6. `step_tensors_` に無い param は **現在の int64 step 値**から fp32 scalar tensor を param と同デバイスに生成:
   `torch::full({}, static_cast<float>(state.step()), torch::TensorOptions().dtype(torch::kFloat32).device(p.device()))`
   → load 後のキャッシュ再構築もこの経路で自動的に行われる。
7. `(device, dtype)` でグループ化（実運用は 1 グループだが、混在しても正しく動くようにする）。グループごとに params / grads / exp_avgs / exp_avg_sqs / (max_exp_avg_sqs) / state_steps の `std::vector<torch::Tensor>` を構築。grads のリストは **毎 step 再収集**（set_to_none により backward ごとに grad テンソルの実体が変わるため、キャッシュ不可）。
8. step 前進: `AdamWParamState` の int64 step を CPU で +1（シリアライズ用の正本）、`at::_foreach_add_(state_steps, 1)` でデバイス側 +1。**順序厳守**: キャッシュ生成（手順 6）は +1 前の値で行い、その後に両方を進める。両者は決定的に一致するため同期不要。
9. グループごとに `at::_fused_adamw_(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, /*maximize=*/false, {}, {})` を呼ぶ。
10. loss（closure 評価結果 or 空 Tensor）を返す。

### 2.2 `load()` の処理

親 `AdamW::load(archive)` を呼んだ後、`step_tensors_.clear()`。次回 step() の手順 6 で int64 step 値から再構築される。

## 3. foreach ヘルパ（同ファイル、`namespace anet` 直下の自由関数）

3 パス（FP32 / AMP+FP16 / AMP+BF16）から共用する。grads の `std::vector<torch::Tensor>` を受ける設計（AMP 展開の下地）。

| 関数 | 仕様 |
|---|---|
| `CollectDefinedGrads(parameters)` | params から defined な grad を収集して返す |
| `ForeachGradNorm(grads) -> torch::Tensor` | `at::_foreach_norm(grads, 2)` → `torch::stack` → `.norm(2)`。デバイス上 scalar を返し **CPU 同期なし**。空 vector ならゼロ tensor |
| `ForeachClipGradNorm_(grads, total_norm, tau)` | `scale = (tau_tensor / (total_norm + 1e-6)).clamp_max(1.0)` → `at::_foreach_mul_(grads, scale)`。既存 FP32 手動 clip（`dqn_based_agent.cpp:1140-1146`）と同セマンティクス（`+ 1e-6` / `clamp_max(1.0)` を踏襲） |

## 4. GradScaler 改修（`core/anet-core/include/anet/nn_util.hpp:68-135`)

- **`Unscale_(optimizer)`**: per-param `div_` ループを廃し、defined grads を収集して
  `at::_amp_foreach_non_finite_check_and_unscale_(grads, found_inf_tensor_, inv_scale_tensor)` の **1 呼び出し**に置換。
  `found_inf_tensor_`（fp32 `{1}`、grad と同デバイス、ゼロ初期化）をメンバに保持。`inv_scale_tensor` は `1.0 / scale_` から同デバイスに生成。
- **`Step(optimizer)` オーバーロード追加**: `found_inf_tensor_` を `.item<float>() != 0.0f` で読み（同期 1 回/step — 従来 AMP パスの `clip_grad_norm_` の CPU 同期が消えるため収支は改善）、inf 検出時は `optimizer.step()` をスキップ。`found_inf_` フラグに記録して `Update()` の backoff に繋ぐ。既存 `Step(optimizer, bool found_inf)` は互換のため残す。
- **`Update()`**: 変更なし（`found_inf_` は Step で設定済み）。
- **挙動変化（意図的・了承済み）**: 従来は `found_inf` 常時 false の簡易実装で inf 検出が no-op だった。本改修により FP16 overflow 時の step スキップ + scale backoff が本来の AMP の動作として機能し始める。

## 5. `Learner::Optimize` 改修（`core/anet-core/src/dqn_based_agent.cpp:1014-1156`）

3 パスすべてを grad_norm_tensor（tensor のまま・CPU 同期なし）経路へ統一する。

- **FP32 パス**: grad_norm 手計算ループ（1126-1130）→ `CollectDefinedGrads` + `ForeachGradNorm`。手動 clip（1140-1146）→ `ForeachClipGradNorm_`。
- **AMP+BF16 パス**（1029-1058）: `torch::nn::utils::clip_grad_norm_`（毎 step CPU 同期あり）と grad_norm 手計算ループを、FP32 と同形の foreach ヘルパ + `result.grad_norm_tensor` 経路に統一。`result.grad_norm`（float）は設定せず `GetScalar` の遅延フォールバックに任せる。
- **AMP+FP16 パス**（1063-1106）: 同様に foreach ヘルパへ統一。`bool found_inf = false;` の行を削除し、`grad_scaler_.Step(*optimizer_, found_inf)` を新オーバーロード `grad_scaler_.Step(*optimizer_)` に変更。
- `ANET_PROFILE_SCOPE` の既存区間名（zero_grad / backward / grad_norm / grad_clip / unscale / optimizer_step / scaler_step / scaler_update）は維持する。
- 既存の処理段階コメント（「AMP + BF16: …」等）は残し、改修意図のコメントを追補する。
- zero_grad（1024）は変更なし。

## 6. `Learner::SetupOptimizer` 改修（`core/anet-core/src/dqn_based_agent.cpp:963-970`）

```cpp
auto opts = torch::optim::AdamWOptions(config_.alpha).weight_decay(config_.weight_decay).eps(config_.adam_eps);
if (config_.use_fused_optimizer) {
    optimizer_ = std::make_unique<anet::FusedAdamW>(model_.GetPolicyParameters(), opts);
} else {
    optimizer_ = std::make_unique<torch::optim::AdamW>(model_.GetPolicyParameters(), opts);
}
```

`LOG::verbose()` の既存出力に `fused=` を追記（ログは英語）。

## 7. 外部仕様（config 追加）

既存フロー（構造体定義 → `ANET_READ_CONFIG` → 設定ファイル）に従う。

| キー | 型 | 既定値 | 意味 |
|---|---|---|---|
| `learner.use_fused_optimizer` | bool | `true` | ATen `_fused_adamw_` による高速 step を使う。false で従来 `torch::optim::AdamW`（fused は演算順序差で bitwise 非再現のため、挙動差の切り分け用） |

1. `core/anet-core/include/anet/agent.hpp` の `LearnerConfig`（`:160-201`）に `bool use_fused_optimizer = true;`（Doxygen コメント付き）を追加。
2. `core/anet-core/include/anet/default_dqn_agent.hpp` の learner.* 読み込み群に `ANET_READ_CONFIG(config_data, learner.use_fused_optimizer);` を追加。
3. `apps/runner/config/agent.txt` の DefaultDQN セクションに `DefaultDQNAgent.baseline.learner.use_fused_optimizer = true` + 用途コメントを追記。
4. Rainbow は `RainbowAgentConfig` が同じ `LearnerConfig` を使う場合のみ同様に読み込みを追加（TBO の前例にならい、公開しない判断でもよい。最小変更を優先）。

## 8. 修正対象ファイル

| ファイル | 変更内容 |
|---|---|
| `core/anet-core/include/anet/nn_util.hpp` | `FusedAdamW` クラス、foreach ヘルパ 3 関数、`GradScaler::Unscale_` 置換 + `Step(optimizer)` オーバーロード |
| `core/anet-core/src/dqn_based_agent.cpp` | `SetupOptimizer` フラグ分岐、`Optimize` の 3 パス foreach 化 |
| `core/anet-core/include/anet/agent.hpp` | `LearnerConfig` に `use_fused_optimizer` |
| `core/anet-core/include/anet/default_dqn_agent.hpp` | `ANET_READ_CONFIG` 追加 |
| `apps/runner/config/agent.txt` | 既定値追記 |
| `core/anet-core/src/nn_test.cpp` | テスト追加（§10） |

適用は DQN Learner のみ。`muzero_based_agent.cpp:937-938` / `image_cls_agent.cpp:76-77` は**変更しない**（将来の 1 行変更として残す）。

## 9. 既存利用可能な部品（再利用先）

- `ANET_READ_CONFIG` マクロ（`default_dqn_agent.hpp`）。
- `ANET_PROFILE_FUNC` / `ANET_PROFILE_SCOPE`（`anet/profile.hpp`、既存区間名は維持）。
- `ANET_SYSTEM_ERROR`（`anet/exception.hpp`、動作不能設定を明示的に落とす規約）。
- `BatchUpdateResult::GetScalar` の grad_norm_tensor 遅延フォールバック（`dqn_based_agent.hpp:86-97`、改修不要でそのまま効く）。
- `LOG::verbose()` / `LOG::warn()`（ログは英語）。

## 10. 検証方針

テストは `core/anet-core/src/nn_test.cpp` に追加（`anet-core-test` ターゲット、`*_test.cpp` 同居規約）。

1. **数値一致**: 同一初期 param 群 + 同一ランダム勾配列で `FusedAdamW` vs `torch::optim::AdamW` を 10 step、param / exp_avg / exp_avg_sq を `allclose`（rtol≈1e-5, atol≈1e-7。fused は bitwise 非一致のため）。weight_decay {0, 1e-2} × デバイス {CPU 必須, `torch::cuda::is_available()` 時 CUDA}。
2. **checkpoint round-trip**: `FusedAdamW` で数 step → save → 新インスタンスに load → 続き step が、save せず連続実行した結果と一致（デバイス上 step tensor 再構築の検証）。
3. **旧形式互換**: `torch::optim::AdamW` で N step → save → **`FusedAdamW` で load** → 以後の更新が AdamW 継続と数値一致。
4. **inf 検出**: GradScaler 使用時、grad に inf を注入 → `Unscale_` → `Step` で param 不変（スキップ）+ `Update` で scale が backoff（半減）。
5. **foreach ヘルパ単体**: `ForeachGradNorm` が per-param 手計算（`pow(2).sum()` 累積 → sqrt）と一致。`ForeachClipGradNorm_` の適用結果が既存手動 clip 実装と一致。
6. **ビルド**: VsDevCmd 経由で x64-Debug をビルドし `core\anet-core\bin\Debug\anet-core-test.exe` を実行（AGENTS.md 必須事項）。
7. **実測（ユーザー実施）**: LunarLander run で Tracy の optimizer_step 区間と steps/sec を前後比較。`use_fused_optimizer = false` で従来挙動に戻ることも確認。

## 11. Out of Scope

- muzero / image_cls への適用（将来のコンストラクタ 1 行変更）。
- `grad_scale` / `found_inf` の fused カーネル直渡し（grad clip 使用時に利得が消えるため。ADR 参照）。
- scale をデバイス tensor 化して `at::_amp_update_scale_` で完全同期レス化する GradScaler 再設計（将来課題。今回は found_inf 読み出しの同期 1 回/step を許容）。
- foreach 手組みの AdamW 実装（fused 一本。ADR 参照）。
- zero_grad の変更（2.11 で既に最適）。
- lr スケジューラ / amsgrad の新規利用（現状未使用。FusedAdamW は amsgrad をパススルー対応するが有効化はしない）。
