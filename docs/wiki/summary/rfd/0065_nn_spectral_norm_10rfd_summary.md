---
title: NN Spectral Normalization（weight_norm.mode）PRD
page_kind: サマリ
material: "[[docs/memo/done/0065_nn_spectral_norm_10rfd.md|0065_nn_spectral_norm_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/phenomena/plasticity.md|可塑性指標と成績]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/subjects/breakout.md|Breakout]]"
updated: 2026-11-28
---
# NN Spectral Normalization（weight_norm.mode）PRD

## 元の材料

[NN Spectral Normalization（weight_norm.mode）PRD](../../../memo/done/0065_nn_spectral_norm_10rfd.md) blob:097d0a6c60e900a97089eed9312d83a1f82b18c9

## 要点

- 高い replay ratio の Breakout で、重み・活性の成長と成績低下を切り分ける制御として Spectral Normalization を導入する。
- 従来の weight decay では weight norm を変えられた一方、10M までのスコア推移は単調にならなかった。
- 重みを持つ Linear、Conv、残差・Transformer 系など6型の登録ブロックに共通設定を設ける。
- power iteration で特異値を推定し、正規化後の重みで計算する。
- 適用する層や設定を変えた比較を可能にし、学習効果は対照 Run で確かめる。

## 内容

### 生の重みと実効重み

Breakout では weight decay によって重みのノルムを用量順に抑えられたが、成績はそれに伴って改善しなかった。Spectral Normalization では生の重み W をパラメータとして保持し、forward に W を最大特異値の推定値で割った重みを使う。生の重みが成長し続けても、forward が使う重みの成長を別に制御できる。

### モードと推定の更新

`spectral` は推定特異値 σ で常に割るため、σ が1未満なら重みを拡大する。σ がゼロで始まる残差枝には使えない。`spectral_cap` では分母を `max(1, |σ|)` とし、ゼロ初期化した枝が恒等写像から学習を始められる。σ は反復による推定値なので、全 forward で真の最大特異値が1以下になる保証はない。

保存するのは power iteration のベクトル u と v で、σ は各 forward の重みから求める。初期化では専用の乱数を使い、15回の反復で推定を立ち上げる。以後は訓練モードかつ勾配有効の forward で1回だけ更新するので、Actor・target・probe の推論で内部状態が変わらない。σ の計算と除算は FP32 で行い、畳み込みや Linear 自体には既存の autocast を使う。

### 同期と診断

Actor への重みのコピーと保存・復元には、u・v も名前付き buffer として含める。soft update で SN を使う場合は混合係数が有限で0〜0.1または1であることを更新前に確認する。通常の `spectral` では算出した σ が有限かつ正であること、`spectral_cap` では有限であることを確認する。層の最大値だけでは一つの不正な層を見落とすので、不正値の有無を別に集約し、異常時だけ層名を特定する。

既存の feature/readout のノルムは引き続き生の重みを表す。実効重みのノルムと σ を併せて測り、対象の層が存在しない側は NaN と件数0で示す。測定・同期・診断が学習用乱数や既存メトリクスに影響しないことを同一 seed で確認する。

### 効果を判定する比較

weight decay は0でそろえる。短いスクリーニングで選んだ構成を運用条件の RR1 で50Mまで評価し、主指標には終盤区間の eval1 を使う。比較対象は SN を入れない8月29日の Run で実測した約427〜454。生と実効の重みの乖離や特徴のランクは、スコアの差を説明するための指標とする。重みの指標だけが改善した構成を成績の改善と取り違えない。

## 主要な概念

- [可塑性指標と成績](../../concepts/phenomena/plasticity.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [Breakout](../../concepts/subjects/breakout.md)
