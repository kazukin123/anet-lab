---
title: Metricsキャッシュは破棄可能な従属導出物とする(マスタ=JSONL、PK=序数)
page_kind: サマリ
material: "[[docs/adr/0015-metrics-cache-disposable-derivative.md|0015-metrics-cache-disposable-derivative]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
updated: 2026-11-26
---
# Metricsキャッシュは破棄可能な従属導出物とする(マスタ=JSONL、PK=序数)

## 元の材料

[ADR 0015](../../../adr/0015-metrics-cache-disposable-derivative.md) blob:9d2081e60f3e237fab47d2376334345376f65f4f

## 要点

- Run ごとの SQLite キャッシュは metrics.jsonl またはその圧縮版から再構築できる。
- 形式や原本の指紋が合わなければ再生成し、旧キャッシュの移行は行わない。
- 同じ step の複数点を保てるよう、各タグ内の出現順で全点を識別する。
- 短命な接続で Run フォルダの移動を妨げず、再構築時には generation を変える。
- 統計は確定した全点から計算する。step 逆行は当該タグを隔離する。保存 LOD の倍率は後に16から4に改めた。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
