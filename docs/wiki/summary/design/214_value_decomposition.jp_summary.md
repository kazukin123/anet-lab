---
title: 外的と内的の価値の分解
page_kind: サマリ
material: "[[docs/design/214_value_decomposition.jp.md|214_value_decomposition.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
updated: 2026-11-28
---
# 外的と内的の価値の分解

## 元の材料

[外的と内的の価値の分解](../../../design/214_value_decomposition.jp.md) blob:d5ff3626e00c8b1fcf086da7e0616ffb22ac50ef

## 要点

- DefaultDQN は外的と内的の価値を別々に学び、行動選択に使う値を β で結合できる。
- ReplayBuffer は成分ごとの報酬・割引率・収益を保持し、外的・内的の意味や β を扱うのは DQN 側である。
- 外的報酬と RND 由来の内的報酬は、それぞれのスケーラと終端の扱いを通す。
- 目標値・損失・優先度をどう構成するかは、分離した Head、最適化の方法と対応する。
- 診断は Head 別の値、結合値と行動への影響を区別する。選択した構成で成立しない値は NaN になる。

## 内容

### 2. 構成と報酬の流れ

- 外的報酬は、既存の外的報酬スケーラを通す。内的報酬は、RNDの倍率から1を引き、`intrinsic.use_done_mask`によるdoneの扱いを適用したあと、内的報酬スケーラを通す。
- 分解時は Body を共有し、TD・QR・IQN の外的・内的 Head に別のパラメータを持たせる。Replay には β を掛ける前の両成分を保存する。行動時に使う値は実空間の `Q_ext + β Q_int` で、価値変換を使う場合は各 Head の値を逆変換してから足す。

### 3. 学習

- Munchausen項とエントロピー項は外的Headだけに置く。内的Headは次状態の内的Qの期待値だけを使う。
- 通常の更新は成分ごとの n-step 収益と割引率を使い、内的 Head の次の行動を結合 Q から1つ選ぶ。外的 Head は設定に応じて結合 Q または外的 Q から選ぶ。Munchausen の更新はそれぞれの方策による期待値を使う。損失と PER 優先度には別々の重みを適用する。
- `non_episodic` でも n-step の和は終端で閉じ、終端のあとのエピソードの報酬は `R_int` に入らない（§3.1）。

### 4. 設定

- `intrinsic.value_mode` の既定は `combined`。`decomposed` を選ぶには内的報酬が有効で、skip が無効である必要がある。

### 5. 診断

- Actor は各 Head の実空間の Q とその Actor が自分の step で評価した行動用の β で行動選択への寄与を比較する。分布型は各分位点を逆変換してから平均する。比べるのは外的 Q と結合 Q の1位の不一致、行動を変えたときの外的 Q の差、外的 Q と β を掛けた内的 Q それぞれの1位と2位の差である。

### 6. 保存と復元

- Networkの保存には両Headを含む。splitは二つのAdam状態を保存する。保存側と読込み側のvalue_modeが異なる場合、または分解時のfeature_gradientが異なる場合は拒否する。報酬スケーラは外的・内的とも、種類と動的な統計、return_stdのlane別Rを保存する。

### 7. テストと拡張時の確認事項

- `testdata/rfd0121` には価値分解の受入手順と再実行の入口がある。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
