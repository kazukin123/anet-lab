---
title: GridMazeのRND性能測定
page_kind: サマリ
material: "[[docs/experiments/default-dqn/gridmaze/2026-11-07_rnd_performance.md|default-dqn/gridmaze/2026-11-07_rnd_performance]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/subjects/gridmaze.md|GridMaze]]"
updated: 2026-11-28
---
# GridMazeのRND性能測定

## 元の材料

[GridMazeのRND性能測定](../../../../../experiments/default-dqn/gridmaze/2026-11-07_rnd_performance.md) blob:58223682c8c9fcee0c229c26cadeeebf1804ebd9

## 要点

- GridMaze の165,632経験で6本ずつの実時間中央値は RND なし23.261秒、あり31.148秒だった。
- 設定と実行体は測定前に複製した。設定解決の全キー比較はRND設定、群56の診断定義、Run名の差だけだった。実際のRunの設定も `inspect_run.py config --effective-only --diff` で照合し、Run名とRNDの有効化・Network設定以外の差はなかった。実効設定の写しは下表と機械可読データから参照できる。
- 試走は各群8,192経験の測定区間で1.057秒/1.485秒だった。遅い群で約30秒を測るよう本測定の件数を選んだ。試走2本とTracy用1本を通常比較の中央値へ混ぜない。
- 12本とも終了0。RNDありの全Runで補助学習の進行を確認し、記録したlearn_countの最大値は1,322だった。
- 「転送名の区間」はobservation/input/error/result transferの合計で、CPU上のTensorの扱いや誤差の有限性検査、診断値の設定などを含む。GPUへの転送量・GPU待ちを測った値ではない。統計の区間には二重正規化の2回目も入るが、誤差・target出力の統計と一緒なので、その費用だけを分離した値ではない。

## 内容

### 通常比較の結果

1巡ごとのON/OFF比は1.4084、1.3177、1.3057で、RNDなし側にもばらつきがある。1.3391は全6本ずつの中央値の比であり、巡回ごとの比の平均ではない。各群の学習系列は報酬の違いで変わり得るため、全体の時間差だけをRND内部の純粋な計算時間とは呼ばない。

### Tracyによる内訳

別のRND有効Runで、時計が16,384経験以上になったことを確認してから公式Tracy 0.13.1のcaptureを8秒接続した。通常比較と同じRelease実行体を使った。計測中のCPU側の区間時間であり、GPU kernel時間ではない。

forwardが56.47%、backwardとoptimizerが35.10%を占めた。`Estimate` 全体は平均6.476ms、`Learn` 全体は1回平均3.535msだった。

### 確認できた範囲と残件

GridMazeのCPU・256 lane・既定RNDについて、§6.3の3巡以上の中央値とTracy内訳を記録した。成績・新しさ判別能力の評価ではない。

## 主要な概念

- [計測の負荷と比較条件](../../../../concepts/measurement/diagnostic_cost.md)
- [内的報酬による探索の比較](../../../../concepts/methods/intrinsic_novelty.md)
- [GridMaze](../../../../concepts/subjects/gridmaze.md)
