---
title: 可塑性メトリクス（srank / dormant unit 率）PRD
page_kind: サマリ
material: "[[docs/memo/done/0062_plasticity_metrics_10rfd.md|0062_plasticity_metrics_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/phenomena/plasticity.md|可塑性指標と成績]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
updated: 2026-11-26
---
# 可塑性メトリクス（srank / dormant unit 率）PRD

## 元の材料

[可塑性メトリクス（srank / dormant unit 率）PRD](../../../memo/done/0062_plasticity_metrics_10rfd.md) blob:9b1ad312091153993265fb76fa3fcb2d66d68478

## 要点

- Breakout の学習中に表現の有効ランクと dormant unit 率を測り、成績低下と表現変化の関係を追う。
- 学習 forward 上の branch capture は Learner が実際に使用したバッチの特徴を取る。
- 独立の probe は ReplayBuffer から一様・非復元で抽出し、PER の偏りを含む学習バッチとは別に測る。
- 測定対象は branch 名で指定し、学習更新の間隔はメトリクスの購読設定から決める。
- 二つの測定には異なる標本と動作モードを使うため、数値を同じ母集団の値として扱わない。

## 内容

### 成績の落ち込みと表現の変化

Breakout の replay ratio（環境から取り込む経験あたりの更新量）を変えたとき、過渡的な成績の谷は RR8、RR4、RR2 の順に深く、RR1 には現れなかった。Q 値の差の縮小や報酬の曲線からだけでは、学習済みの特徴が一部の方向に集中したのか、出力しなくなったユニットが増えたのかは判別しづらい。そこで読み出し直前の指定 branch に対して二種類の指標を用意する。

### 指標の計算

`srank` は、バッチの各行をサンプル、各列を特徴とする行列の特異値を大きい順に足し、合計の99%に達するのに必要な本数。全特徴がゼロなら0とする。ランクが落ちると、線形読み出し層が行動ごとの価値差を表すために使える方向も減る。dormant 率はユニットごとの平均絶対活性を全ユニット平均に対して測る。比が0なら完全に不活性、0.025以下なら低活性として割合を出す。全ユニットの活性がゼロなら、どちらの割合も1とする。

### 学習バッチと独立の probe

学習側は実際の更新に使った forward から特徴を取り出す。Dropout やバッチ統計の影響もその更新と同じで、追加 forward は要らない。一方、PER の抽出確率は優先度の α 乗に比例し、同一経験が何回引かれるか、優先度がどこまで古くなるかが replay ratio に応じて変わる。従って、学習バッチの変化には表現自体とサンプル分布の両方が含まれる。

独立の probe は ReplayBuffer から一様・非復元で抽出し、評価モードで指定 branch とその依存先だけを実行する。IQN では測りたい branch が最終出力に残らないので、融合後の出力を代用できない。訓練モードで追加 forward すると Dropout などが学習用の乱数を消費するため、比較する Run の条件も変わってしまう。

### 比較と受け入れ

`srank` の上限はバッチ件数と特徴次元の小さい方。Atari の学習バッチ256件・特徴512次元では、表現が健全でも天井に張り付くため、独立 probe の既定512件を主に読む。warmup 直後は経験の多様性が低い可能性もあり、絶対値だけで判断せず、各 Run の初期から過渡期までの変化と、RR4 の seed 違い2本のブレを比較する。

測定を購読しないときは特徴取り出し・probe・サンプリングを実行しない。測定の有無だけを変えた同一 seed で学習系列と保存重みの一致を調べ、測定負荷は Atari と軽量環境で別に測る。

## 主要な概念

- [可塑性指標と成績](../../concepts/phenomena/plasticity.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
