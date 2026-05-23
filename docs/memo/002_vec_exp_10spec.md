# 【修正仕様】Ape-X型 空間探索（Vectorized Exploration）機能の追加

## 1. 概要

時間経過による探索パラメータ（$\epsilon$ や $\tau$）のアニーリングを無効化し、並列環境（`num_envs`）ごとに異なる固定のパラメータを空間的に割り当てる「空間的アニーリング」を導入する。これにより、Replay Bufferの多様性と学習の安定性を向上させる。
DefaultDQNAgentのみ対象。RainbowAgent は対応対象外。

## 2. 外部仕様（Config拡張）

`train_policy` 設定ブロックに以下の項目を追加する。

| パラメータ名 | 型 | デフォルト値 | 説明 |
| --- | --- | --- | --- |
| `use_spatial_exploration` | `bool` | `false` | `true` の場合、空間的アニーリングを有効にする。 |
| `spatial_scale_type` | `string` | `"log"` | パラメータ分配方式（`"log"` または `"linear"`）。 |

### 2.1. 既存パラメータの解釈

`use_spatial_exploration = true` の場合、以下の既存設定は空間分配の境界値として使用される。

* **`*_start`**: インデックス `0`（最大探索環境）のパラメータ値。
* **`*_end`**: インデックス `num_envs - 1`（最小探索環境）のパラメータ値。
* **`*_decay_steps`**: **無視する**（使用しない）。

### 2.2. 整合チェック

* use_spatial_exploration=true かつ num_envs<32 の場合、学習が不安定になる可能性がある。この場合、WARNログだけ出力して動作は継続。

## 3. 内部仕様

### 3.1. 空間パラメータテンソルの生成

ActionPolicy またはユーティリティクラスにて、サイズ `[num_envs]` のテンソルを初期化時に生成する。

* **引数:** `int num_envs`, `float start_val`, `float end_val`, `const std::string& scale_type`, `torch::Device device`
* **戻り値:** `torch::Tensor` (Shape: `[num_envs]`)
* **ゼロ除算・負値への安全対策ロジック:**
  1. `num_envs == 1` の場合は、`start_val` のスカラーテンソルを返す。
  2. `[0.0, 1.0]` の範囲を `num_envs` 等分した線形テンソル `base = torch::linspace(0.0f, 1.0f, num_envs, ...)` を生成。
  3. `scale_type == "log"` の場合:
     * `start_val <= 0.0f` または `end_val <= 0.0f` の場合、対数計算の崩壊を防ぐため、該当する値を一時的に `1e-4f` にクランプ（置換）して計算を行う。また、この際には `LOG::warn()` 等で自動クランプした旨を1度だけ通知する。
     * 式: $Value = end\_val\_clamped \times \left( \frac{start\_val\_clamped}{end\_val\_clamped} \right)^{1.0 - base}$
  4. `scale_type == "linear"`（または上記以外のフォールバック）の場合:
     * 式: $Value = start\_val + base \times (end\_val - start\_val)$

### 3.2. ActionPolicy クラスの改修（UQE / EpsilonGreedy共通）

時間減衰のスカラー値ではなく、事前生成したテンソルを用いてマスク処理等を行うようにし、同時に不要なスカラー値の出力（ロギング）を抑止する。

* **対象インスタンス:** `train_policy_` のみ（初期化時に `use_spatial_exploration` フラグを渡す）。
* **初期化時の処理:**
  * `use_spatial_exploration == true` の場合：
    1. 上記の `CreateSpatialEpsilon` 等を用いてテンソル（例: `spatial_eps_tensor_`, `spatial_tau_tensor_`）を生成し、メンバ変数として保持する。
    2. 既存のスカラ状態変数（`current_epsilon_` や `current_uqe_tau_`）には、**`std::numeric_limits<float>::quiet_NaN()`** を代入する。
* **推論時の処理 (`MakeAction` 等):**
  * `use_spatial_exploration == true` の場合、保持している空間パラメータテンソル（Shape: `[num_envs]`）を用いて行動選択（乱数テンソルとの比較やマスク生成）を一括で行う。

### 3.3. メトリクス出力と状態更新のスキップ制御

空間探索有効時は、時間経過によるパラメータの減衰（アニーリング）が存在しないため、無駄な計算をスキップする。

* **状態の更新 (`OnLearn` や `UpdateVariables`):**
  * メソッドの冒頭で `use_spatial_exploration == true` か判定し、`true` ならば**早期 return（スキップ）**する。
  * これにより、`current_epsilon_` 等は初期化時の `NaN` が永続的に維持される。
* **メトリクスの出力:**
  * エージェント側がロギングのために `ActionPolicy` のゲッター（`GetEpsilon()` 等）を呼び出した際、自然に `NaN` が返される形となる。
  * フレームワークの既存仕様（`NaN` は無視される）により、特別な分岐を書かなくとも自動的に不要なグラフ更新がスキップされる。
