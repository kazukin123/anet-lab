---
title: AtariのRND性能測定
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-11-12_rnd_performance.md|default-dqn/atari/2026-11-12_rnd_performance]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
  - "[[docs/wiki/concepts/subjects/montezuma_revenge.md|Montezuma's Revenge]]"
updated: 2026-11-26
---
# AtariのRND性能測定

## 元の材料

[AtariのRND性能測定](../../../../../experiments/default-dqn/atari/2026-11-12_rnd_performance.md) blob:4be05891816ac5989491d55baf227c8a1a7487b7

## 要点

- Montezuma・CUDA・64 lane で、89,984経験の実時間中央値は RND なし19.019秒、既定31.584秒、Taiga 設定280.616秒だった。
- 設定と実行体は測定前に複製した。3群の設定解決の全キー比較で、差はRNDの設定、群56の診断定義、Run名だけだった（なしと既定で24キー、既定とTaigaで9キー）。全Runの実効設定は下表と機械可読データから参照できる。
- 試走の8,192経験は、なし1.822秒、既定2.980秒、Taiga 25.804秒だった。試走3本とTracy用2本を通常比較の中央値へ混ぜない。
- 1巡ごとの比（各巡の群の中央値の比）は、既定/なしが1.665、1.698、1.533、Taiga/なしが14.754、15.793、13.774。3巡目は3群とも1・2巡目より速かった。1.661と14.754は全6本ずつの中央値の比であり、巡ごとの比の平均ではない。各群の学習系列は報酬の違いで変わり得るため、全体の時間差だけをRND内部の純粋な計算時間とは呼ばない。
- 18本とも終了0。RNDありの全Runで補助学習の進行を確認し、記録したlearn_countの最大値は既定958、Taiga 122,688だった。試走とTracy用の実効設定は、pilot_off・pilot_on・pilot_taiga・profile_on・profile_taiga。
- Tracy で完結した区間に限ると Taiga 設定の RND 内部では Learn が93.20%を占めた。

## 内容

### Tracyによる内訳

既定RNDとTaiga設定の別Runで、時計が32,768経験以上になったことを確かめてから公式Tracy 0.13.1のcaptureを8秒接続した。通常比較と同じRelease実行体を使った。計測中のCPU側の区間時間であり、GPU kernel時間ではない。割合は、完結した `intrinsic_reward` の区間の合計（RND全体）を分母にしている。

既定RND。 完結した `intrinsic_reward` は388 intake（24,832観測）、`Estimate` は388回、`Learn` は194回だった。RND全体の合計は5,081.5ms、1 intakeあたり13.097msで、捕捉区間8.121秒の62.57%を占めた。

`Estimate` は合計4,201.2ms（RND全体の82.68%）、`Learn` は842.4ms（16.58%）。正規化は1回あたり平均1.041msで、582回のうち194回は `Learn` の入力の正規化だった。

## 主要な概念

- [内的報酬による探索の比較](../../../../concepts/methods/intrinsic_novelty.md)
- [計測の負荷と比較条件](../../../../concepts/measurement/diagnostic_cost.md)
- [Montezuma's Revenge](../../../../concepts/subjects/montezuma_revenge.md)
