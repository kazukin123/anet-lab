---
title: メトリクスの横軸と間隔を数える単位を分ける RFD
page_kind: サマリ
material: "[[docs/memo/0096_metrics_step_axis_interval_10rfd.md|0096_metrics_step_axis_interval_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
updated: 2026-11-26
---
# メトリクスの横軸と間隔を数える単位を分ける RFD

## 元の材料

[メトリクスの横軸と間隔を数える単位を分ける RFD](../../../memo/0096_metrics_step_axis_interval_10rfd.md) blob:86571caa0c982f576cdbfc65e4347254e71c0217

## 要点

- 学習 step を使う診断を、バッチサイズや replay ratio が異なる Run で経験数に合わせて比較するための暫定仕様を記す。
- 従来は記録の横軸と計測間隔の単位が同じため、同じ経験数でもグラフの長さが異なって見えた。
- 推奨案は、学習 step で従来どおり発火した記録に、その時点の経験数も別欄で保存する方式である。
- Viewer の既定は定義された横軸とし、利用者が経験数軸に切り替える。
- 過去 Run の軸を推測で補わず、新しい記録と区別して扱う。

## 内容

### 目的と問題

メトリクス定義の `$learn_step` / `$exp_step` / `$train_step` は、「記録の横軸」と「`interval:` を数える単位」を兼ねている。この 2 つを別々に指定できない。

### 提案の骨子

案 A を推す。学習ステップの軸の記録だけに経験数を添えるので、ファイルはほとんど太らない。全記録に全座標を書く形は、1 行あたり 40〜50 バイト増え、jsonl が 5 割近く太るので取らない。

### 作らないもの・先送り

測定を経験数で数えて決めること（案 B の後半）: 経験数あたりの測定密度と EMA の時定数を群どうしでそろえられるが、学習側の判断の仕組みに触れる。段階 1・2 で読めるようになってから、要るかを決める。

## 主要な概念

- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
