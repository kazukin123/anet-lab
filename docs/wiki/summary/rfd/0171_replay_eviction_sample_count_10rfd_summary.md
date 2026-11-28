---
title: Replay から消えるまでの抽選回数を経験の種類別に測る案
page_kind: サマリ
material: "[[docs/memo/0171_replay_eviction_sample_count_10rfd.md|0171_replay_eviction_sample_count_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/subjects/kung_fu_master.md|kung_fu_master（KFM）]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
updated: 2026-11-28
---
# Replay から消えるまでの抽選回数を経験の種類別に測る案

## 元の材料

[RFD0171　Replay から消えるまでの抽選回数を経験の種類別に測る案](../../../memo/0171_replay_eviction_sample_count_10rfd.md) blob:5de828716dec9e1a448fd501d3d3bb074d310341

## 要点

- 経験を区分して抽選・排出を計測する暫定案であり、詳細設計と区分の選択をグリルに残した。
- Kung-Fu Master のボス撃破のような珍しい経験が、消えるまでに学習へ何度使われたかを調べたい。
- 優先度そのもので分類せず、報酬の大きさや終端など遷移の中身で区分する。
- 各区分で追い出された遷移の抽選回数を集計し、未抽選の経験も数える。
- 同じ区分の raw priority も集計し、珍しい経験に TD 誤差の差が付くかを別に確かめる。
- 高い予測誤差と、方策を変えるうえで重要な経験であることは区別して結果を読む。

## 内容

### 目的と問題

PER が効いているかを、PER の内側の量（優先度、TD 誤差）ではなく、遷移の中身で分けた区分ごとに「消えるまでに何回、学習用抽選で引かれたか」で測れるようにする。知りたいのは、報酬の大きい遷移や終端の遷移のような珍しい経験が、ふつうの遷移より多く引かれているか、どれだけ引かれずに消えているか、である。
PER あり・なし（一様抽選）のどちらでも同じ指標を出し、一様の場合と比べられるようにする。

### 起草時点の現状

遷移ごとの抽選の記録は、`sampled_once_`（uint8 のフラグ、replay_buffer_impl.hpp）だけである。`Sample` の `MarkSampledOnce` で 1 にし、`Push` で slot を再利用するときに 0 に戻す。PER の有無によらず保守されている。回数は持たない。

### 提案の骨子

`sampled_once_` のフラグを、slot ごとの抽選回数に置き換える。`MarkSampledOnce` の書き込みが加算に変わるだけで、hot path の仕事量は今と同じ。1 回の抽選の中で同じ slot が 2 回選ばれたら 2 と数える。未抽選群・抽選済み群は「回数が 0 か、1 以上か」で今と同じに決まる。
`Push` のとき、その slot の区分を決めて slot ごとに持つ。区分は 2 つの軸で、PER の優先度を使わない。

### User Stories

研究者として、ボス撃破のような珍しい報酬の遷移が、消えるまでに何回学習に使われたかを、Run の指標で確かめたい。
研究者として、`per_alpha` を変えた群どうしで、報酬のある遷移の持ち上げ率がどう変わったかを比べたい。
研究者として、PER を切った Run で同じ指標が 1 付近になることを見て、計器が正しいことを確かめたい。
研究者として、報酬クリップをかけたままの Run でも、クリップ前の得点の大きさで遷移を分けて数えたい（段階 2）。

### 候補を採る場合の成功の測り方と受入の目安

PER を切った Run（一様抽選）で、全区分の平均抽選回数が「1 回の更新で引く件数 × 寿命中の更新回数 ÷ sampleable range の件数」に近く、どの区分の持ち上げ率も 1 付近になる。
kung_fu_master の報酬クリップを外した構成で、`per_alpha` を 0.2 から 0.6 にすると、撃破を分けた区分（境界 5 の `r2`）の持ち上げ率が上がる。

## 主要な概念

- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [kung_fu_master（KFM）](../../concepts/subjects/kung_fu_master.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
