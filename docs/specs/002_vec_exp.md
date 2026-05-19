# 【修正仕様】Ape-X型 空間探索（Vectorized Exploration）機能の追加

## 1. 概要

強化学習の探索パラメータ（$\epsilon$ や $\tau$）の時間経過によるアニーリング（減衰）を無効化し、並列環境（$N$環境）ごとに異なる固定のパラメータを空間的に割り当てる「空間的アニーリング（Ape-X型探索）」を導入する。
これにより、高探索な環境と完全に貪欲（Greedy）な環境を永続的に並存させ、Replay Bufferの多様性と学習の安定性を向上させる。

## 2. 外部仕様（Config拡張）

`train_policy` の設定パラメータに以下の項目を追加し、空間探索のON/OFFおよび分配方式を指定可能にする。

### 2.1. 追加パラメータ

| パラメータ名 | 型 | デフォルト値 | 説明 |
| --- | --- | --- | --- |
| `use_spatial_exploration` | `bool` | `false` | `true` の場合、時間アニーリングを無効化し空間的アニーリングを有効にする。 |
| `spatial_scale_type` | `string` | `"log"` | 並列環境へのパラメータ分配方式（`"log"` または `"linear"`）。通常は対数スケール(`log`)を推奨。 |

### 2.2. 既存パラメータの解釈変更

`use_spatial_exploration = true` の場合、既存の設定値は以下の意味を持つ。

* **`*_start`**: 最も探索を行う環境（インデックス `0`）に割り当てるパラメータの最大値。
* **`*_end`**: 全く探索を行わない、または最小限の探索を行う環境（インデックス `N-1`）に割り当てるパラメータの最小値。
* **`*_decay_steps`**: 一切使用されない（無視される）。

## 3. 内部仕様（ロジック追加・変更）

### 3.1. 空間パラメータテンソルの生成ユーティリティ

`ActionPolicy` 内部、または共通の数値計算ユーティリティとして、サイズ `[N]` のパラメータテンソルを生成する関数を追加する。

**【仕様】**

* 引数: `int num_envs` (環境並列数), `float start_val`, `float end_val`, `const std::string& scale_type`, `torch::Device device`
* 戻り値: `torch::Tensor` (Shape: `[num_envs]`)
* 計算ロジック:
1. `num_envs == 1` の場合は、`start_val` のスカラーテンソルを返す。
2. `[0.0, 1.0]` の範囲を `num_envs` 等分した線形テンソル `base` を生成する。
3. `scale_type == "log"` かつ `start_val > 0` かつ `end_val > 0` の場合：
$Value = end\_val \times \left( \frac{start\_val}{end\_val} \right)^{1.0 - base}$
4. 上記以外（`scale_type == "linear"` 等）の場合：
$Value = start\_val + base \times (end\_val - start\_val)$



### 3.2. ActionPolicy クラスの改修（UQE / EpsilonGreedy共通）

時間減衰のスカラー値ではなく、生成したテンソルを用いてマスク処理等を行うように変更する。

* **初期化 (Construct / Initialize):**
* `use_spatial_exploration == true` の場合、上記のユーティリティ関数を呼び出し、メンバ変数としてテンソル（例: `spatial_eps_tensor_`, `spatial_tau_tensor_` 等）を事前生成して保持しておく。


* **推論 (MakeAction等):**
* `use_spatial_exploration == true` の場合、引数で渡される `RuntimeVars` からの時間減衰スカラー値（`epsilon`, `uqe_tau` 等）を無視する。
* 代わりに、保持している事前生成テンソル（Shape: `[N]`）を用いて行動選択（一様乱数テンソルとの比較によるマスク生成など）を一括で行う。



### 3.3. メトリクス出力（MetricsLogger / RuntimeVars）の制御

空間探索有効時は、各環境で $\epsilon$ 等のパラメータが異なるため、単一のスカラー値としてTensorBoard等へロギングすることは不適切となる。

* **変更内容:**
`RuntimeVars` からメトリクス用の値を取得・出力する際、`use_spatial_exploration == true` の条件に合致する場合は、該当する探索パラメータ（`epsilon`, `uqe_tau` など）の戻り値を **`std::numeric_limits<float>::quiet_NaN()`** に設定する。
* **目的:** フレームワークの既存仕様（`nan` を返すとクリーンに無視される機能）を利用し、無意味な固定値のロギングを自動的に除外するため。

