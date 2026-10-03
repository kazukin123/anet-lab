# Transformed Bellman Operator (TBO) 導入 仕様書

## Context（背景・目的）

DQN 系エージェントで Q 値が発散・暴走しやすい問題への対策として、**Transformed Bellman Operator**（Pohlen et al. 2018 "Observe and Look Further", R2D2/Ape-X 系）を導入する。Bellman ターゲットを可逆な圧縮関数 `h` で写像することで、報酬スケールに依存せず大きなターゲット値を抑え、学習を安定化させる。

将来的な Munchausen RL 導入（報酬に ≤0 の `α·τ·log π` 項が加わりターゲットが大きく負へ振れる）への事前整備でもあるが、**本仕様書のスコープは TBO 本体のみ**。

## 1. 変換関数の定義

```
h(x)    = sign(x)·(√(|x|+1) − 1) + ε·x
h⁻¹(x)  = sign(x)·( ( (√(1 + 4ε(|x| + 1 + ε)) − 1) / (2ε) )² − 1 )
```

- `ε·x` 項は可逆性・Lipschitz 性を保証する正則化（`ε = 1e-2` が文献標準）。
- Tensor（要素ごと）に作用する `TransformH(x)` / `TransformHInv(x)` を新設する。
- **配置: 共通基底クラス `anet::rl::dqn::Learner`（`core/anet-core/src/dqn_based_agent.hpp:433`）の共通処理（protected メンバ関数）として定義する。** `TDLearner` と `QuantileLearnerBase`(→`QRLearner`) は共にこの基底を継承しており（`dqn_based_agent.hpp:516, 495, 528`）、`config_`（`use_tbo`/`tbo_epsilon`）も基底が保持するため、メンバ関数内で `config_.tbo_epsilon` を直接参照できる。フリー関数や無名（匿名）namespace は使わない。

## 2. ターゲット計算への組込み

`h` は単調増加なので **argmax 行動選択は h 空間でも不変**（greedy / ε-greedy / UQE 単一分位点 `uqe_use_tail_mean=false`）。よって行動選択ロジックは変更不要で、ターゲット計算のみ改修する。

### 2.1 スカラ TD（`TDLearner::UpdateFromSamples`, `dqn_based_agent.cpp:1618-1623`）

現状:
```cpp
auto td_target = target_returns.detach()
               + not_terminal * gamma_n * max_next_q.detach();
```

改修（`use_tbo` 時）:
```cpp
auto bootstrap = config_.use_tbo
    ? TransformHInv(max_next_q.detach())   // 基底 Learner の共通処理。h⁻¹でブートストラップ値を実空間へ
    : max_next_q.detach();
auto raw_target = target_returns.detach() + not_terminal * gamma_n * bootstrap;
auto td_target = config_.use_tbo
    ? TransformH(raw_target)               // 再度 h 空間へ
    : raw_target;
```

- `target_returns`（n-step 累積報酬）は実空間のまま、`gamma^n` 乗算後に `h` を適用。
- ネット出力 `max_next_q` は h 空間なので、必ず `h⁻¹` で実空間へ戻してから Bellman 加算する。

### 2.2 分布型 QR-DQN（`QuantileLearnerBase::CalcTargetQuantiles`, `dqn_based_agent.cpp:1426-1428`）

現状:
```cpp
auto target_dist = returns + gamma_n * not_terminal * next_dist;
```

改修（`use_tbo` 時）: 各分位点ターゲットに同形を適用。
```cpp
auto next = config_.use_tbo ? TransformHInv(next_dist) : next_dist;   // 基底 Learner の共通処理
auto raw  = returns + gamma_n * not_terminal * next;
auto target_dist = config_.use_tbo ? TransformH(raw) : raw;
```

- 損失（Quantile Huber, `ComputeQuantileHuberLoss`）は h 空間の分位点で計算され、変更不要。

### 2.3 既存機能との整合（変更不要だが意味が変わる点）

- **Double DQN**: 次行動 argmax は h 空間でも不変。`h⁻¹` はブートストラップ値にのみ適用する。
- **N-step**: `gamma^n` は実空間の累積報酬に対し従来通り。
- **TD クリップ / PER 優先度**: h 空間の TD 誤差に作用する（自然に圧縮される）。`td_clip_value` 等は再チューニングが必要になりうる旨をコメントに記載する。

## 3. 報酬スケーラとの共存と warn

- 報酬は `DefaultDQNAgent::UpdateFromBatch`（`default_dqn_agent.cpp:506`）でリプレイ保存前にスケール済み。TBO は学習器内のスケール済み Q 空間に作用する（パイプライン変更なし）。
- **共存可とし、人間の設定を尊重する（エラーにはしない）。** ただしエージェント初期化時、`learner.use_tbo` かつ `reward_scaler.use_dynamic_scaling` または `use_auto_post_scale` が有効な場合は、二重圧縮の可能性を warn で通知する:

```cpp
anet::log::warn()
    << "learner.use_tbo is enabled together with reward_scaler.use_dynamic_scaling or "
    << "reward_scaler.use_auto_post_scale; targets may be double-compressed.";
```

`default_dqn_agent.hpp` 内では `LOG` エイリアスを置けないため、この warn は完全修飾の `anet::log::warn()` を使う。

（`LOG` は `anet::log` のエイリアス。`core/anet-core/include/anet/log.hpp`）

## 4. 外部仕様（config 追加）

既存の config 追加フロー（構造体定義 → `ANET_READ_CONFIG` → 設定ファイル）に従う。

| キー | 型 | 既定値 | 意味 |
|---|---|---|---|
| `learner.use_tbo` | bool | `false` | TBO の有効/無効 |
| `learner.tbo_epsilon` | float | `1e-2` | 変換関数の正則化項 ε |

1. `core/anet-core/include/anet/agent.hpp` の `LearnerConfig`（`:160`）に `bool use_tbo = false;` と `float tbo_epsilon = 1e-2f;` を追加。
2. `core/anet-core/include/anet/default_dqn_agent.hpp` の `DefaultDQNAgentConfig` で `ANET_READ_CONFIG(config_data, learner.use_tbo);` と `learner.tbo_epsilon` を読込み。
3. `RainbowAgentConfig`（`rainbow_agent.hpp`）では**当該キーを読まない**（default `false` のまま＝常に OFF 固定）。オリジナルの Rainbow に無い要素のため、Rainbow には公開しない。
4. `apps/runner/config/agent.txt` の DefaultDQN セクションに既定値（`use_tbo = false`, `tbo_epsilon = 0.01`）を追記。検証時は `apps/runner/config/DropMerge.txt` 側で `use_tbo = true` をオーバーライド。

## 5. Q 値メトリクス（学習 scalar の h 空間生値 ＋ h⁻¹ 逆変換値）

- 学習 scalar メトリクス集計で、`use_tbo` 時に追加で `h⁻¹` を適用した実空間 Q を**別 tag**で出力する。
- `MakeActionInfo` の `aux["q_values"]` は h 空間の生値のまま維持し、action-info aux への `q_values_real` 追加は行わない。
- h 空間の生値は従来 tag を維持する（学習が実際に見ている値）。tag 命名は既存規約（TensorBoard 由来のメトリクス識別子）に準拠し、`name`（Runner 等のインスタンス識別子）とは混同しない。
- メトリクスロガーの構造は変えず、`BatchUpdateResult` の学習 scalar 出力箇所の最小改修にとどめる。

## 6. 修正対象ファイル

| ファイル | 変更内容 |
|---|---|
| `core/anet-core/include/anet/agent.hpp` | `LearnerConfig` に `use_tbo`, `tbo_epsilon` |
| `core/anet-core/include/anet/default_dqn_agent.hpp` | `ANET_READ_CONFIG` 追加 |
| `core/anet-core/src/dqn_based_agent.hpp` | 基底 `anet::rl::dqn::Learner` に `TransformH`/`TransformHInv`（protected）宣言 |
| `core/anet-core/src/dqn_based_agent.cpp` | `TransformH`/`TransformHInv` 実装（基底 Learner 共通処理）、`TDLearner`/`QRLearner` ターゲット改修、学習 scalar の Q メトリクス h⁻¹ 出力 |
| `apps/runner/config/agent.txt` | 既定値追記 |
| `apps/runner/config/DropMerge.txt` | 検証用 `use_tbo = true` オーバーライド |
| `core/anet-core/src/*_test.cpp` | `h∘h⁻¹` 往復テスト追加 |

## 7. 既存利用可能な部品（再利用先）

- `ANET_READ_CONFIG` マクロ（config 読込み, `default_dqn_agent.hpp`）。
- `gamma_n` 計算（既存, `dqn_based_agent.cpp:1622, 1427`）。
- `LOG::warn()`（warn ログ。`LOG = anet::log` エイリアス, `anet/log.hpp`）。
- 既存メトリクス出力経路（`MakeActionInfo` aux, `metrics_logger.cpp`）。

## 8. 検証方針

1. **単体テスト**: 代表値（0, ±1, ±10, ±1e3, 終端含む）について `‖h(h⁻¹(x)) − x‖ < 1e-4`、`h⁻¹(h(x)) ≈ x`、`h` の単調性を確認する（`ε` を数点で）。
2. **DropMerge スモーク**: `use_tbo=false`/`true` を切替えて短時間学習。h 空間 Q が圧縮され発散しないこと、h⁻¹ 逆変換 Q が報酬規模に整合すること、greedy 評価が破綻しないことを確認。
3. **warn ログ確認**: `use_tbo=true` ＋ `use_dynamic_scaling=true` で warn が出ること。
4. **ビルド**: VsDevCmd 経由でビルド成功を確認（AGENTS.md 必須事項）。

## 9. Out of Scope

- Munchausen RL 本体の実装（別仕様。本件はその事前整備）。
- MuZero 系（独自の値変換を持つため対象外）。
- Rainbow への TBO 公開（OFF 固定）。
- `td_clip_value` 等の自動再チューニング（手動調整前提、コメントで注意喚起のみ）。
