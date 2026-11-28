---
title: Run 色は比較ごとの役割とし、分離は選択中 Run の集合に対してだけ保証する
page_kind: サマリ
material: "[[docs/adr/0041-run-color-is-comparison-role-not-run-identity.md|0041-run-color-is-comparison-role-not-run-identity]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
updated: 2026-11-28
---
# Run 色は比較ごとの役割とし、分離は選択中 Run の集合に対してだけ保証する

## 元の材料

[ADR 0041](../../../adr/0041-run-color-is-comparison-role-not-run-identity.md) blob:ff7b8c8e6b2b3461781fc529af705f94e9bd3ed1

## 要点

- Run の色は表示中の比較で見分けるために割り当てる。
- workspace 全体の列挙順だけで配ると、比較する2本が似た色になることがある。
- 分離する対象は選択中の Run に限り、色を Run の永続属性にはしない。
- 手動 Recolor は選んだ Run 全体を再配色する。
- 後の ADR 0050 では閾値0.16を廃止し、自動配色で既選択の色を保ちながら新規選択に最良の色を与える。

## 内容

### 比較と永続化

同じ Run を常に同じ色にすると、偶然近い色の組を比較するときに見分けにくい。一方、選択のたびに全件を再配色すると比較を続けている Run の見た目まで変わる。

### 自動と手動

色の分離は今回選んだ集合に限り、手動ボタンで全体を再最適化できるようにした。後の変更で、既存色の自動変更を許した距離閾値は廃止された。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
